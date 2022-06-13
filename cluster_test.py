import faiss
import torch
import torch.nn as nn
import numpy as np


def run_kmeans(x, num_cluster, temperature=0.5, gpu_id=0):
    """
    Args:
        x: data to be clustered
    """
    
    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    seed = 0
    

    # intialize faiss clustering parameters
    d = x.shape[1] #  d = feature_dim
    k = int(num_cluster) # number of clusters for the run
        # CLUSTERING CONFIGURATION & PARAMETERS
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = 20
    clus.nredo = 5
    clus.seed = seed
    #clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 10

        # CLUSTERING PARALLEL EXECUTION
    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = gpu_id    
    index = faiss.GpuIndexFlatL2(res, d, cfg)  

        # START CLUSTERING
    clus.train(x, index)   

#D = cluster distance for each sample to its cluster_centroid
#I =  cluster assignment for each sample


    D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
    im2cluster = [int(n[0]) for n in I] # cluster assignments in sample-order

    print(len(im2cluster))
        
        # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
        
        # sample-to-centroid distances for each cluster 
    Dcluster = [[] for c in range(k)] # [[],[]...] for each cluster centroid          
    for im,i in enumerate(im2cluster): # im = sampleID | i = clusterID ( assigned for sample "im")
        Dcluster[i].append(D[im][0])
        
        # concentration estimation (phi)        
    density = np.zeros(k) # concentration parameter for each cluster
    for i,dist in enumerate(Dcluster): # for each cluster its distances dist
        if len(dist)>1: # at least 2 cluster members necessary
            d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
            density[i] = d # density(concentration) for  cluster i   
                
        #if cluster only has one point, use the max to estimate its concentration        
    dmax = density.max() # FOR EACH CLUSTER WITH ONE OR ZERO POINTS
    for i,dist in enumerate(Dcluster):
        if len(dist)<=1:
            density[i] = dmax 

        # SOULUTION FOR TOO LOW OR HIGH CONCENTRATION VALUES !!!!!!!!!!!!!!!!!
    density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
    density = temperature*density/density.mean()  #scale the mean to temperature 
        
        # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=2, dim=1) # normalize Centroids for unit-length   

    im2cluster = torch.LongTensor(im2cluster).cuda()               
    density = torch.Tensor(density).cuda()
        #------------------------------------------------------------
        
        # APPEND THE VALUES FOR EACH SINGLE CLUSTERING [25000,50000,100000]
    results['centroids'] = centroids
    results['density'] = density
    results['im2cluster'] = im2cluster    
        
    return results


dummy = torch.rand([5000,128])
results = run_kmeans(dummy.cpu().numpy(),10)
print(results['centroids'].shape)
print(results['density'].shape)
print(results['im2cluster'].shape)