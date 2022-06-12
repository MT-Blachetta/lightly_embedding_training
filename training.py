import torch
import copy
import numpy as np
import nltk
from nltk.cluster.kmeans import KMeansClusterer

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



class Trainer_clPcl(object):

    def __init__(self,p,criterion):
        self.num_clusters = p['num_classes']
        self.alpha = p['loss_alpha']
        self.lamb = p['loss_lambda']
        self.criterion = criterion.cuda()
        self.best_loss = 10000
        self.best_model = None

    def train_one_epoch(self, train_loader, model, optimizer, epoch, cluster_module=None):
        
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(len(train_loader),[losses],prefix="Epoch: [{}]".format(epoch))
        alpha = self.alpha
        iloss = torch.nn.CrossEntropyLoss()
        iloss = iloss.cuda()
        model = model.cuda()
        model.train()
        

        for i, batch in enumerate(train_loader):
            originImage_batch = batch['image']
            augmentedImage_batch_list = batch['image_augmented']
            indices_batch = batch['index']
            originImage_batch = originImage_batch.cuda(non_blocking=True)
            group_loss = 0
            instance_loss = 0
            
            for augmentedImage_batch in augmentedImage_batch_list:
                
                augmentedImage_batch = augmentedImage_batch.cuda(non_blocking=True)

                logits, labels = model(originImage_batch,augmentedImage_batch)
                instance_loss += iloss(logits,labels)


                original_view = model.group(originImage_batch) # € [batch_size ,feature_dim]
                augmented_view = model.group(augmentedImage_batch) # € [batch_size ,feature_dim]
                feature_dim = len(original_view[0])
                batch_size = len(original_view)

                #alpha = 0.1
                divzero = 0.1
                ov = original_view.cpu().detach().numpy()
                #print(ov.shape)
                av = augmented_view.cpu().detach().numpy()
                k = self.num_clusters

                if cluster_module:
                    labels_, cluster_centers = cluster_module.cluster_batch(indices_batch)
                    labels_I, cluster_centers_I = cluster_module.cluster_batch_I(indices_batch)
                else:      
                    clusterer = KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance, repeats=20, normalise=True, avoid_empty_clusters=True)
                    clusterer_I = KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance, repeats=20, normalise=True, avoid_empty_clusters=True)
                    labels_ = clusterer.cluster(ov,True)
                    labels_I = clusterer_I.cluster(av,True)
                    cluster_centers = torch.Tensor( np.array(clusterer.means()) ) 
                    cluster_centers_I = torch.Tensor( np.array(clusterer_I.means()) )


                #cluster_centers
                #MI_kmeans_results.append( cluster_centers_I )
                    # c -> k
                center = [ cluster_centers[i] for i in range(k) ]
                center_I = [ cluster_centers_I[i] for i in range(k) ]
                cdat = [ x.unsqueeze(0).expand(batch_size,feature_dim) for x in center]
                cmatrix = torch.cat(cdat,1)
                cdat_I = [ x.unsqueeze(0).expand(batch_size,feature_dim) for x in center_I]
                cmatrix_I = torch.cat(cdat_I,1)

                original_cpu = original_view.cpu()
                augmented_cpu = augmented_view.cpu()          
                fmatrix = torch.Tensor(copy.deepcopy(ov))
                fmatrix_I = torch.Tensor(copy.deepcopy(av))

                for _ in range(1,k): fmatrix = torch.cat((fmatrix,original_cpu),1)
                for _ in range(1,k): fmatrix_I = torch.cat((fmatrix_I,augmented_cpu),1)
                        
                cmatrix = cmatrix.cuda()
                fmatrix = fmatrix.cuda()
                cmatrix_I = cmatrix_I.cuda()
                fmatrix_I = fmatrix_I.cuda()
                    
                zmatrix = fmatrix-cmatrix
                zmatrix = zmatrix*zmatrix
                result = zmatrix.flatten(0).view(batch_size,k,feature_dim)
                result = torch.sum(result,2)
                result = torch.sqrt(result)

                zmatrix_I = fmatrix_I-cmatrix_I
                zmatrix_I = zmatrix_I*zmatrix_I
                result_I = zmatrix_I.flatten(0).view(batch_size,k,feature_dim)
                result_I = torch.sum(result_I,2)
                result_I = torch.sqrt(result_I)
                    
                assign = torch.zeros(batch_size,k)
                assign_I = torch.zeros(batch_size,k)

                for i in range(batch_size):
                    assign[i][ int(labels_[i]) ] = 1
                    assign_I[i][ int(labels_I[i]) ] = 1
                        
                assign = assign.cuda()
                assign_I = assign_I.cuda()
                    
                avgDistance = torch.sum(assign*result,0)
                Z = torch.sum(assign,0) + 1
                Zlog = torch.log(Z+alpha)
                divisor = Z*Zlog
                concentrations = (avgDistance/divisor) + divzero
                concentrations = concentrations.cpu()
                avgDistance_I = torch.sum(assign_I*result_I,0)
                Z_I = torch.sum(assign_I,0) + 1
                Zlog_I = torch.log(Z_I+alpha)
                divisor_I = Z_I*Zlog_I
                concentrations_I = (avgDistance_I/divisor_I) + divzero
                concentrations_I = concentrations_I.cpu()
                    
                group_loss += self.criterion( features = original_view, features_I = augmented_view, M_kmeans = cluster_centers , M_kmeans_I = cluster_centers_I, concentrations = concentrations, concentrations_I = concentrations_I, labels = labels, labels_I = labels_I, lb = self.lamb)
            
            loss = instance_loss + group_loss
            
            print('loss: ',str(loss))
            if loss < self.best_loss:
                self.best_loss = loss
                self.best_model = copy.deepcopy(model)
            
            losses.update(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 25 == 0:
                progress.display(i)