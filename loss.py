import torch
from torch import nn

#@author: Michael Blachetta
class PclCldLoss_2(nn.Module):
    
    def __init__(self):
        super(PclCldLoss_2, self).__init__()
        #self.temperature = temperature
        
    def forward(self,features,features_I,M_kmeans,M_kmeans_I,concentrations,concentrations_I,labels,labels_I,lb):
        
        #M_num = len(concentrations)
        #print(M_num)
        batch_size = features.size()[0]

        #M_logits = []
        #M_logits_I = []

        #if k == 2: print()

        #for k in range(M_num):
        c = len(concentrations) # c = num_clusters of Mk
        M_cmatrix = torch.zeros(c,batch_size)
        MI_cmatrix = torch.zeros(c,batch_size)
        for i in range(c):
            M_cmatrix[i,:] = 1/concentrations[i]
            MI_cmatrix[i,:] = 1/concentrations_I[i]

          #if k == 2: print(M_cmatrix)          
        M_cmatrix = M_cmatrix.cuda()
        MI_cmatrix = MI_cmatrix.cuda()     
        centroids = M_kmeans.cuda()
        centroids_I = M_kmeans_I.cuda()
        gLoss_or = torch.mm(centroids,features_I.T) # OK 
        gLoss_au = torch.mm(centroids_I,features.T)
          #print("gLoss_or type: "+str(type(gLoss_or)) )
          #print("gLoss_or shape: "+str(gLoss_or.shape) )
        #--------------------------------------------------------
        summing_logits = gLoss_or * M_cmatrix # OK
        summing_logits_I = gLoss_au * MI_cmatrix

        exp_logits = torch.exp(summing_logits)
        exp_logits_I = torch.exp(summing_logits_I)
        log_sum = torch.sum(exp_logits,0)
          #print("log_sum type: "+str(type(log_sum)))
          #print("log_sum shape: "+str(log_sum.shape))
        log_sum_I = torch.sum(exp_logits_I,0)

        positive_pair = torch.zeros(batch_size)
        positive_pair_I = torch.zeros(batch_size)

        exlogCPU = exp_logits.cpu()
        exlogCPU_I = exp_logits_I.cpu()
          #lcpu = labels[k].cuda()
          #lcpu_ = labels_I[k].cuda()
        for l in range(batch_size):
            positive_pair[l] = exlogCPU[int(labels[l])][l]
            positive_pair_I[l] = exlogCPU_I[int(labels_I[l])][l]

        positive_pair = positive_pair.cuda()
        positive_pair_I = positive_pair_I.cuda()
                    #positive_pair = torch.exp(torch.mm(positive_pair,gLoss_or))
                    #positive_pair_I = torch.exp(torch.mm(positive_pair_I,gLoss_au))

        M_logits = torch.sum( torch.log(positive_pair/log_sum) ) # +0.0001 ),0).cpu()       ) 
        M_logits_I = torch.sum( torch.log(positive_pair_I/log_sum_I) ) # +0.0001 ),0).cpu() ) 

        return (1/batch_size)*-lb*0.5*(M_logits.cpu() + M_logits_I.cpu())

class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss