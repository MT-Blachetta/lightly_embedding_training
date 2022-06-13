import torchvision
import torch
from torch import nn
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.models.modules import BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.modules import NNCLRProjectionHead
from lightly.models.modules import NNCLRPredictionHead
from lightly.models.modules import SimSiamProjectionHead
from lightly.models.modules import SimSiamPredictionHead
from lightly.models.modules import SwaVProjectionHead
from lightly.models.modules import SwaVPrototypes
from lightly.models.modules import SimCLRProjectionHead
import copy

class barlowtwins_model(nn.Module):
    def __init__(self, backbone, backbone_outdim, hidden_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(backbone_outdim, hidden_dim, out_dim)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z
        
    def bf(self,x):
        return self.backbone(x).flatten(start_dim=1)


class byol_model(nn.Module):
    def __init__(self, backbone, backbone_dim, hidden_dim, out_dim):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(backbone_dim, hidden_dim, out_dim)
        self.prediction_head = BYOLProjectionHead(out_dim, hidden_dim, out_dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z
        
    def bf(self,x):
        return self.backbone(x).flatten(start_dim=1) 


class nnclr_model(nn.Module):
    def __init__(self, backbone,backbone_dim,hidden_dim,out_dim):
        super().__init__()

        self.backbone = backbone
        self.projection_head = NNCLRProjectionHead(backbone_dim,hidden_dim,out_dim)
        self.prediction_head = NNCLRPredictionHead(out_dim, hidden_dim, out_dim)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p
        
    def bf(self,x):
        return self.backbone(x).flatten(start_dim=1) 


class simsiam_model(nn.Module):
    def __init__(self, backbone, backbone_dim,hidden_dim,out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(backbone_dim, backbone_dim, out_dim)
        self.prediction_head = SimSiamPredictionHead(out_dim, hidden_dim, out_dim)

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p
        
    def bf(self,x):
        return self.backbone(x).flatten(start_dim=1)



class swav_model(nn.Module):
    def __init__(self, backbone, backbone_dim, hidden_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SwaVProjectionHead(backbone_dim, hidden_dim, out_dim)
        self.prototypes = SwaVPrototypes(out_dim, n_prototypes=512)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p
        
    def bf(self,x):
        return self.backbone(x).flatten(start_dim=1)


class simclr_model(nn.Module):
    def __init__(self, backbone, backbone_dim, hidden_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(backbone_dim, hidden_dim, out_dim)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z
        
    def bf(self,x):
        return self.backbone(x).flatten(start_dim=1) 


class clpcl_model(nn.Module): # Conditions: [+] key_encoder is copy(query_encoder) 
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, backbone ,backbone_dim ,dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(clpcl_model, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = backbone
        self.encoder_k = copy.deepcopy(backbone)
        self.instance_head_q = nn.Linear(backbone_dim,dim)
        self.instance_head_k = nn.Linear(backbone_dim,dim)
        self.group_head = nn.Linear(backbone_dim,dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

    
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient <- !!!!!!!!!!!!!!!!!

        for param_q, param_k in zip(self.instance_head_q.parameters(), self.instance_head_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient <- !!!!!!!!!!!!!!!!!
    
        
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
                

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.instance_head_q.parameters(), self.instance_head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        #print("IN MoCo->_dequeue_and_enqueue: batch_size = "+str(batch_size))

        ptr = int(self.queue_ptr)
        #print("queue_ptr = "+str(ptr))
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        #print("next ptr = "+str(ptr))
        self.queue_ptr[0] = ptr



    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = self.instance_head_q(q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            #im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = self.instance_head_k(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            #k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def group(self,im_q):

        q = self.encoder_q(im_q)  # queries: NxC
        q = self.group_head(q)
        q = nn.functional.normalize(q, dim=1)

        return q

    def get_backbone(self):
        return self.encoder_q

