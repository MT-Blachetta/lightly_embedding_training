import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
import collections
from torch._six import string_classes
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
        # TO DO

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
    else: 
        raise ValueError('not ready yet')

def get_trainer(p,criterion):

    if p['train_method'] == 'clpcl':
        from training import Trainer_clPcl
        return Trainer_clPcl(p,criterion)

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

