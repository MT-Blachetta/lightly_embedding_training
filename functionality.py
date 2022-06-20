import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
import collections
from torch._six import string_classes
import torchvision.transforms as transforms
from models import clpcl_model
int_classes = int


class wrapped_resnet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self,x):
        return self.backbone(x).flatten(start_dim=1)

def get_backbone(p):

    if p['backbone'] == 'scatnet':
        from scatnet import scatnet_backbone
        backbone = scatnet_backbone(J=p['model_kwargs']['J'], L=p['model_kwargs']['L'], input_size=p['model_kwargs']['input_size'], res_blocks=p['model_kwargs']['res_blocks'])
        out_dim = backbone.out_dim

    elif p['backbone'] == 'resnet18':
        resnet = torchvision.models.resnet18()
        backbone = nn.Sequential(*list(resnet.children())[:-1])
        out_dim = 512

    else: raise ValueError("invalid backbone ID")

    return {'backbone':backbone, 'out_dim':out_dim}

def get_model(p,backbone,backbone_dim):
    
    if p['base_model'] == 'scatnet':
        from scatnet import ScatSimCLR
        return ScatSimCLR(J=p['model_kwargs']['J'], L=p['model_kwargs']['L'], input_size=p['model_kwargs']['input_size'], res_blocks=p['model_kwargs']['res_blocks'], out_dim=p['model_kwargs']['out_dim'])
    
    elif p['base_model'] == 'barlow':
        from models import barlowtwins_model
        return barlowtwins_model(backbone, backbone_dim, p['model_kwargs']['hidden_dim'], p['model_kwargs']['out_dim'])

    elif p['base_model'] == 'simclr':
        from models import simclr_model
        return simclr_model(backbone, backbone_dim, p['model_kwargs']['hidden_dim'], p['model_kwargs']['out_dim'])

    elif p['base_model'] == 'byol':
        from models import byol_model
        return byol_model(backbone, backbone_dim, p['model_kwargs']['hidden_dim'], p['model_kwargs']['out_dim'])

    elif p['base_model'] == 'nnclr':
        from models import nnclr_model
        return nnclr_model(backbone, backbone_dim, p['model_kwargs']['hidden_dim'], p['model_kwargs']['out_dim'])

    elif p['base_model'] == 'simsiam':
        from models import simsiam_model
        return simsiam_model(backbone, backbone_dim, p['model_kwargs']['hidden_dim'], p['model_kwargs']['out_dim'])

    elif p['base_model'] == 'swav':
        from models import swav_model
        return swav_model(backbone, backbone_dim, p['model_kwargs']['hidden_dim'], p['model_kwargs']['out_dim'])

    elif p['base_model'] == 'clpcl':
        from models import clpcl_model
        return clpcl_model(backbone, backbone_dim, p['model_kwargs']['out_dim'])

def get_transform(p):

    augmentation_method = p['augmentation_strategy']
    #aug_args = p['augmentation_kwargs']
    # get_train_transformations(p):
    if augmentation_method == 'standard':
            # Standard augmentation strategy
        from transform import StandardAugmentation
        train_transformation = StandardAugmentation(p)

    elif augmentation_method == 'simclr':
            # Augmentation strategy from the SimCLR paper
        from transform import SimclrAugmentation
        train_transformation = SimclrAugmentation(p)

    elif augmentation_method == 'scan': # 'ours' -> 'scan'
            # Augmentation strategy from our paper 
        from transform import ScanAugmentation
        train_transformation = ScanAugmentation(p)

    elif augmentation_method == 'random':
        from transform import RandAugmentation
        train_transformation = RandAugmentation(p)

    elif augmentation_method == 'moco':
        from transform import MocoAugmentations
        train_transformation = MocoAugmentations(p)
    
    elif augmentation_method == 'barlow':
        from transform import BarlowtwinsAugmentations
        train_transformation = BarlowtwinsAugmentations(p)

    elif augmentation_method == 'multicrop':
        from transform import MultiCropAugmentation
        train_transformation = MultiCropAugmentation(p)
        
    else:
        raise ValueError('Invalid augmentation strategy {}'.format(p['augmentation_strategy']))

    return train_transformation


def get_val_dataset(p,transform):

    #split = p['split']
    #dataset_type = p['dataset_type']

    if p['val_db_name'] == 'cifar-10':
        from dataset import CIFAR10
        dataset = CIFAR10(train=False, transform=transform, download=True)
        #eval_dataset = CIFAR10(train=False, transform=val_transformations, download=True)

    elif p['val_db_name'] == 'cifar-20':
        from dataset import CIFAR20
        dataset = CIFAR20(train=False, transform=transform, download=True)
        #eval_dataset = CIFAR20(train=False, transform=transform, download=True)

    elif p['val_db_name'] == 'stl-10':
        from dataset import STL10
        dataset = STL10(split='test', transform=transform, download=False)
        #eval_dataset = STL10_trainNtest(path='/space/blachetta/data',aug=val_transformations)
        #eval_dataset = STL10(split='train',transform=val_transformations,download=False)

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))

    return dataset

def get_val_transformations(p):
    return transforms.Compose([
            transforms.CenterCrop(p['transformation_kwargs']['crop_size']),
            transforms.ToTensor(), 
            transforms.Normalize(**p['transformation_kwargs']['normalize'])])

def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)

def validation_loader(p):
    transformation = get_val_transformations(p)
    vds = get_val_dataset(p,transformation)
    vloader = get_val_dataloader(p,vds)

    return vloader


def get_dataset(p,transform):

    split = p['split']
    #dataset_type = p['dataset_type']

    if p['train_db_name'] == 'cifar-10':
        from dataset import CIFAR10
        dataset = CIFAR10(train=True, transform=transform, download=True)
        #eval_dataset = CIFAR10(train=False, transform=val_transformations, download=True)

    elif p['train_db_name'] == 'cifar-20':
        from dataset import CIFAR20
        dataset = CIFAR20(train=True, transform=transform, download=True)
        #eval_dataset = CIFAR20(train=False, transform=transform, download=True)

    elif p['train_db_name'] == 'stl-10':
        from dataset import STL10
        dataset = STL10(split=split, transform=transform, download=False)
        #eval_dataset = STL10_trainNtest(path='/space/blachetta/data',aug=val_transformations)
        #eval_dataset = STL10(split='train',transform=val_transformations,download=False)

    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    # Wrap into other dataset (__getitem__ changes)
    # Dataset returns an image and an augmentation of that image.
    from dataset import AugmentedDataset
    return AugmentedDataset(dataset)

def collate_custom(batch):
    if isinstance(batch[0], np.int64):
        return np.stack(batch, 0)

    if isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, 0)

    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch, 0)

    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)

    elif isinstance(batch[0], float):
        return torch.FloatTensor(batch)

    elif isinstance(batch[0], string_classes):
        return batch

    elif isinstance(batch[0], collections.abc.Mapping):
        batch_modified = {key: collate_custom([d[key] for d in batch]) for key in batch[0] if key.find('idx') < 0} # in den key namen darf sich 'idx' nicht als substring befinden
        return batch_modified

    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [collate_custom(samples) for samples in transposed]

    raise TypeError(('Type is {}'.format(type(batch[0]))))

def get_dataloader(p,dataset,collate=collate_custom):

    batch_loader = torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate,
            drop_last=True, shuffle=True)

    return batch_loader

def get_optimizer(p,model):

    params = model.parameters()
                
    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, **p['optimizer_kwargs'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, **p['optimizer_kwargs'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def get_criterion(p):
    #p['criterion_kwargs']['temperature']
    if p['criterion'] == 'clpcl':
        from loss import PclCldLoss_2
        return PclCldLoss_2()
    elif p['criterion'] == 'simclr':
        from lightly.loss import NTXentLoss
        return NTXentLoss()
    elif p['criterion'] == 'nnclr':
        from lightly.loss import NTXentLoss
        return NTXentLoss()
    elif p['criterion'] == 'barlowtwins':
        from lightly.loss import BarlowTwinsLoss
        return BarlowTwinsLoss()
    elif p['criterion'] == 'simsiam':
        from lightly.loss import NegativeCosineSimilarity
        return NegativeCosineSimilarity()
    elif p['criterion'] == 'byol':
        from lightly.loss import NegativeCosineSimilarity
        return NegativeCosineSimilarity()
    elif p['criterion'] == 'svaw':
        from lightly.loss import SwaVLoss
        return SwaVLoss()
    else: 
        raise ValueError('not ready yet')

def get_trainer(p,criterion):

    if p['train_method'] == 'clpcl':
        from training import Trainer_clPcl
        return Trainer_clPcl(p,criterion)

    if p['train_method'] == 'proto':
        from training import Trainer_proto
        return Trainer_proto(p,criterion)

    elif p['train_method'] == 'simclr':
        from training import Trainer_simclr
        return Trainer_simclr(p,criterion)

    elif p['train_method'] == 'nnclr':
        from training import Trainer_nnclr
        return Trainer_nnclr(p,criterion)

    elif p['train_method'] == 'simsiam':
        from training import Trainer_simsiam
        return Trainer_simsiam(p,criterion)

    elif p['train_method'] == 'barlowtwins':
        from training import Trainer_barlowtwins
        return Trainer_barlowtwins(p,criterion)
    
    elif p['train_method'] == 'byol':
        from training import Trainer_byol
        return Trainer_byol(p,criterion)

    else: raise ValueError('trainer not implemented')

def adjust_learning_rate(p, optimizer, epoch):
    lr = p['optimizer_kwargs']['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n # number of instances with features in MemoryBank
        self.dim = dim # feature dimension
        self.features = torch.FloatTensor(self.n, self.dim) # instance features in an array/matrix 
        self.targets = torch.LongTensor(self.n) # list/sequence of the instance labels
        self.ptr = 0 # STATE, memory bank size / pointing to the last element
        self.device = 'cpu'
        self.K = 100 # number of k nearest neighbours
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions): # perform weighted knn
        
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device) # for each K nearest neighbors, the class label as one-hot-Vektor
        batchSize = predictions.shape[0] # number of points the weighted_knn predeiction is performed
        
        correlation = torch.matmul(predictions, self.features.t()) # dot product of the input features with all labeled MemoryBank features
        # dot product equals the cosine similarity if feature vectors are normalized (unit length)
        
        set_zero = torch.zeros(correlation.shape[1])
        for i in range( correlation.shape[0] ):
            correlation[i] = torch.where( correlation[i] > 0.9999, set_zero, correlation[i] )
 
        # The K nearest neighbors in the memory bank for each batch instance
        # yd: topK highest similarity values - starting with the highest one
        # yi: topK INDEXes with highest similarity values - starting with the highest one
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)  
        
        # class labels of the memoryBank instances COPIED for each batch instance at dim=0
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        
        retrieval = torch.gather(candidates, 1, yi) # class labels of the memoryBank K_NN for each batch instance
        
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_() # class as one-hot-Vector: for all batches the K Nearest Neighbors stacked in one dimension
        
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1) # class labels of all kNN of all batches in one-hot encoding
        
        
        # div_ := alle Werte dividieren, exp_ := in die exponentialfunktion einsetzen
        yd_transform = yd.clone().div_(self.temperature).exp_() # [batch-size,self.K]  
        
        yd_sum = 1/torch.sum(yd_transform,1) # Summe der Ähnlichkeitswerte aller kNN's per batch
        #torch.mul(yd_transform,yd_sum.view(-1,1)) # relativer Anteil eines kNN an der Summe der Ähnlichkeitswerte aller kNN's 
        
        """retrieval_one_hot.view(batchSize, -1 , self.C)""" # Wahrscheinlichkeitswert für die Zugehörigkeit der kNN zu ihrer Klasse
        # One-Hot Repräsentation der Klassenangehörigkeit der k-NN's(dim) für jede batch instance(dim)
        
        """yd_transform.view(batchSize, -1, 1)"""
        # Der Distanz/Ähnlichkeitswert(probabilities) der k-NN's(dim) für jede batch instance(dim)
        # torch.mul trägt im one-hot-Vektor an der Stelle des Klassenlabels die distance/similarity des kNN ein (die Gewichtung des kNN)
        
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1) # Für jede Klasse werden die Gewichte aller KNN's dieser Klasse addiert
                          
        class_stats = torch.mul(probs,yd_sum.view(-1,1)) # Werte von [0,1] für jede Klasse
        # Die weighted_knn Summe aller kNN's einer Klasse befindet sich im one-hot-Vektor am Index der KlassenID   
        
        _, class_preds = probs.sort(1, True) # sortiert man die Indices nach Höhe der weighted_knn_sum erhält man die Klasse wo es die nahesten KNN's gibt 
        
        class_pred = class_preds[:, 0] # class assignments of the weighted_knn Algorithm for each batch instance

        return class_pred, class_stats

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t()) # [batch_size,self.n] Distanz/Ähnlichkeitswert zwischen batch(predictions) und Datensatz(MemoryBank)
        
        sample_pred = torch.argmax(correlation, dim=1) # [batch_size] die Indices der Batch 1-Nearest Neighbors 
        
        class_pred = torch.index_select(self.targets, 0, sample_pred) # [batch_size] Klassenlabels der 1-Nearest Neighbors pro batch
        
        return class_pred


    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0) # probably the batch-size of the features that updates the memory bank
        
        assert(b + self.ptr <= self.n) # der pointer darf nur bis zur Länge self.n der MemoryBank gehen
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach()) # füge die features aus der batch zur MemoryBank hinzu
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach()) # labels der Trainingsdaten als integer [0,9]
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')


class AverageMeter(object):
    def __init__(self,name, fmt=':f'):
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


def evaluate_knn(p,val_dataloader,model,device='cuda:0'):

    model.eval()
    i = 0
    features = []
    targets = []
    with torch.no_grad():
        model = model.to(device)
        for batch in val_dataloader:
            imgs = batch['image']
            labels = batch['target']

            imgs = imgs.to(device)
            labels = labels.to(device)

            if isinstance(model,clpcl_model):
                features.append(model.group(imgs))
            else:
                features.append(model(imgs))

            targets.append(labels)

        #features_tensor = torch.cat(features)
        targets_tensor = torch.cat(targets)
      
        
    dsize = len(targets_tensor) #[returns the first dimension size of targets]

    memory_bank = MemoryBank(int(dsize), p['feature_dim'], p['num_classes'], p['temperature'])
    memory_bank.reset()
    memory_bank.to(device)
    for f, l in zip(features,targets): memory_bank.update(f, l)
    memory_bank.to(device)
    topmeter = AverageMeter('Acc@1', ':6.2f')

    for output, target in zip(features, labels):

        w_nn, class_stats = memory_bank.weighted_knn(output)
        correct_classmask = torch.eq(w_nn, target).float()
        acc1 = 100*torch.mean(correct_classmask)
        topmeter.update(acc1.item(), output.size(0)) 

    print('Result of kNN evaluation is %.2f' %(topmeter.avg))

    return


        #p['temperature']

    



