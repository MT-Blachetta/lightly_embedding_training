import torch
import copy
import numpy as np
import nltk
from nltk.cluster.kmeans import KMeansClusterer
from loss import SimCLRLoss
from lightly.models.utils import update_momentum
from lightly.models.modules import NNMemoryBankModule

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


class Trainer_nnclr(object):

    def __init__(self,p,criterion):
        self.criterion = criterion
        self.memory_bank = NNMemoryBankModule(size=8192).cuda()
        self.device = p['device']

    def train_one_epoch(self, train_loader, model, optimizer, epoch):

        model.train()
        model = model.to(self.device)

        for batch in train_loader:
            originImage_batch = batch['image']
            augmentedImage_batch_list = batch['image_augmented']

            originImage_batch = originImage_batch.to(self.device,non_blocking=True)
            z0, p0 = model(originImage_batch)

            loss = 0
            
            for augmentedImage_batch in augmentedImage_batch_list:
                augmentedImage_batch = augmentedImage_batch.to(self.device,non_blocking=True)
                z1, p1 = model(augmentedImage_batch)

                z0 = self.memory_bank(z0, update=False)
                z1 = self.memory_bank(z1, update=True)

                loss += 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"epoch: {epoch:>02}, loss: {loss:.5f}")

#------------------------------------------------------------------

class Trainer_barlowtwins(object):

    def __init__(self,p,criterion):
        self.criterion = criterion
        self.device = p['device']

    def train_one_epoch(self, train_loader, model, optimizer, epoch):

        model.train()
        model = model.to(self.device)

        for batch in train_loader:
            originImage_batch = batch['image']
            augmentedImage_batch_list = batch['image_augmented']

            originImage_batch = originImage_batch.to(self.device)
            z0 = model(originImage_batch)

            loss = 0
            
            for augmentedImage_batch in augmentedImage_batch_list:
                augmentedImage_batch = augmentedImage_batch.to(self.device)
                z1 = model(augmentedImage_batch)
                loss += self.criterion(z0, z1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"epoch: {epoch:>02}, loss: {loss:.5f}")



class Trainer_simsiam(object):

    def __init__(self,p,criterion):
        self.criterion = criterion
        self.device = p['device']
        

    def train_one_epoch(self, train_loader, model, optimizer, epoch):

        model.train()
        model = model.to(self.device)

        for batch in train_loader:
            originImage_batch = batch['image']
            augmentedImage_batch_list = batch['image_augmented']

            originImage_batch = originImage_batch.to(self.device)
            z0, p0 = model(originImage_batch)

            loss = 0
            
            for augmentedImage_batch in augmentedImage_batch_list:
                augmentedImage_batch = augmentedImage_batch.to(self.device)
                z1, p1 = model(augmentedImage_batch)

                loss += 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"epoch: {epoch:>02}, loss: {loss:.5f}")




#----------------------------------------------------------------


class Trainer_simclr(object):

    def __init__(self,p,criterion):
        self.criterion = criterion
        self.device = p['device']

    def train_one_epoch(self, train_loader, model, optimizer, epoch):

        model.train()
        model = model.to(self.device)

        for batch in train_loader:
            originImage_batch = batch['image']
            augmentedImage_batch_list = batch['image_augmented']

            originImage_batch = originImage_batch.to(self.device)
            z0 = model(originImage_batch)

            loss = 0
            
            for augmentedImage_batch in augmentedImage_batch_list:
                augmentedImage_batch = augmentedImage_batch.to(self.device)
                z1 = model(augmentedImage_batch)
                loss += self.criterion(z0, z1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"epoch: {epoch:>02}, loss: {loss:.5f}")


class Trainer_byol(object):

    def __init__(self,p,criterion):
        self.criterion = criterion
        self.device = p['device']

    def train_one_epoch(self, train_loader, model, optimizer, epoch):

        model.train()
        model = model.to(self.device)

        for batch in train_loader:
            update_momentum(model.backbone, model.backbone_momentum, m=0.99)
            update_momentum(model.projection_head, model.projection_head_momentum, m=0.99)
            originImage_batch = batch['image']
            augmentedImage_batch_list = batch['image_augmented']

            originImage_batch = originImage_batch.to(self.device)
            p0 = model(originImage_batch)
            z0 = model.forward_momentum(originImage_batch)

            loss = 0
            
            for augmentedImage_batch in augmentedImage_batch_list:
                augmentedImage_batch = augmentedImage_batch.to(self.device)
                p1 = model(augmentedImage_batch)
                z1 = model.forward_momentum(augmentedImage_batch)
                loss += 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))


            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"epoch: {epoch:>02}, loss: {loss:.5f}")





#----------------------------------------------------------------

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
                    cluster_module.batch_cluster_ids(indices_batch)
                    labels_, cluster_centers = cluster_module.cluster_batch(original_view)
                    labels_I, cluster_centers_I = cluster_module.cluster_batch(augmented_view,augmented=True)
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


class Trainer_proto(object):

    def __init__(self,p,criterion):
        self.num_clusters = p['num_classes']
        self.alpha = p['loss_alpha']
        self.lamb = p['loss_lambda']
        self.criterion = criterion.cuda()
        self.best_loss = 10000
        self.best_model = None
        self.nxt_criterion = SimCLRLoss(p['temperature'])
        self.version = 1
        self.phase = 0

    def train_one_epoch(self, train_loader, model, optimizer, epoch, cluster_module):
        
        if self.phase == 0:
            self.train_phase_A(train_loader, model, optimizer, epoch, cluster_module)
            self.configure_phase(self,epoch)

        elif self.phase == 1:
            self.train_phase_B(train_loader, model, optimizer, epoch, cluster_module)
            self.configure_phase(epoch)

        elif self.phase == 2:
            self.train_phase_C(train_loader, model, optimizer, epoch, cluster_module)            
            self.configure_phase(epoch)

        else: raise ValueError('invalid phase value')

    def configure_phase(self,epoch):
        
        if self.version == 1:
            if epoch > 50: self.phase = 2
            elif epoch > 20: self.phase = 1
        return      

    def train_phase_A(self, train_loader, model, optimizer, epoch, cluster_module=None):

        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(len(train_loader),[losses],prefix="Epoch: [{}]".format(epoch))
        #alpha = self.alpha
        iloss = torch.nn.CrossEntropyLoss()
        iloss = iloss.cuda()
        model = model.cuda()
        model.train()
        

        for i, batch in enumerate(train_loader):
            originImage_batch = batch['image']
            augmentedImage_batch_list = batch['image_augmented']
            #indices_batch = batch['index']
            originImage_batch = originImage_batch.cuda(non_blocking=True)
            #group_loss = 0
            instance_loss = 0
            
            for augmentedImage_batch in augmentedImage_batch_list:
                
                augmentedImage_batch = augmentedImage_batch.cuda(non_blocking=True)

                logits, labels = model(originImage_batch,augmentedImage_batch)
                instance_loss += iloss(logits,labels)

            losses.update(instance_loss.item())

            optimizer.zero_grad()
            instance_loss.backward()
            optimizer.step()

            if i % 25 == 0:
                progress.display(i)
        

    def train_phase_B(self, train_loader, model, optimizer, epoch, cluster_module=None):
        
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
                    cluster_module.batch_cluster_ids(indices_batch)
                    labels_, cluster_centers = cluster_module.cluster_batch(original_view)
                    labels_I, cluster_centers_I = cluster_module.cluster_batch(augmented_view,augmented=True)
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



    def train_phase_C(self, train_loader, model, optimizer, epoch, cluster_module):

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
                av = augmented_view.cpu().detach().numpy()
                k = self.num_clusters

                cluster_module.batch_cluster_ids(indices_batch)

                #print('clusterer.batch_cluster: ',cluster_module.batch_cluster.shape)
                #print('clusterer.batch_cluster_I: ',cluster_module.batch_cluster_I.shape)

                labels_, cluster_centers = cluster_module.cluster_batch(original_view)
                labels_I, cluster_centers_I = cluster_module.cluster_batch(augmented_view,augmented=True)
                #cluster_centers = torch.stack(cluster_centers,dim=0).cuda()
                #cluster_centers_I = torch.stack(cluster_centers_I,dim=0).cuda()

                mask_per_label = cluster_module.cluster_mask()
                prototype_list = []
                prototype_list_I = []

                for mask in mask_per_label:
                    resultant = torch.zeros([batch_size,feature_dim])
                    resultant_I = torch.zeros([batch_size,feature_dim])
                    for j in range(batch_size): 
                        resultant[j] = original_view[j,:]*mask[j]
                        resultant_I[j] = augmented_view[j,:]*mask[j]

                    #num_cluster_instances = torch.sum(mask)
                    instance_sum = torch.sum(resultant,dim=0)
                    instance_sum_I = torch.sum(resultant_I,dim=0)                  
                    prototype_list.append(instance_sum)
                    prototype_list_I.append(instance_sum_I)

                prototypes = torch.stack(prototype_list)
                prototypes_I = torch.stack(prototype_list_I)
                prototypes = torch.nn.functional.normalize(prototypes)
                prototypes_I = torch.nn.functional.normalize(prototypes_I)

                prototype_eval = torch.stack([prototypes,prototypes_I],dim=1)
                prototype_eval = prototype_eval.cuda()

                proto_loss = self.nxt_criterion(prototype_eval)

                #print('Proto loss: ',proto_loss)
                #print('labels_: len ',len(labels_),' type ',type(labels_), ' [0] ',labels_[0])
                #print('labels_I: len ',len(labels_I),' type ',type(labels_I), ' [0] ',labels_I[0])
                #print('cluster_centers: len ',len(cluster_centers),' type ',type(cluster_centers), ' [0] ',cluster_centers[0])
                #print('cluster_centers_I: len ',len(cluster_centers_I),' type ',type(cluster_centers_I), ' [0] ',cluster_centers_I[0])


                center = [ cluster_centers[i] for i in range(k) ]
                #print('center len: ',len(center))
                #print('center[0]: ',type(center[0]))
                #print('center[0]: ',center[0].shape)
                center_I = [ cluster_centers_I[i] for i in range(k) ]
                #print('center_I len: ',len(center_I))
                #print('center_I[0]: ',type(center_I[0]))
                #print('center_I[0]: ',center_I[0].shape)
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
            
            loss = instance_loss + 0.5*(group_loss + proto_loss)
            
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


