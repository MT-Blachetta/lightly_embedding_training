import faiss
import torch
import torch.nn as nn
import numpy as np


class cluster_module(object):

    def __init__(self,num_clusters,temperature,gpu_id):
        self.temperature = temperature
        self.num_cluster = num_clusters
        self.gpu_id = gpu_id
        self.features = []
        self.features_I = []
        self.centroids = None
        self.im2cluster = None
        self.density = None
        self.im2cluster_I = None
        self.batch_cluster = None
        self.batch_cluster_I = None


    def run_kmeans(self, x):
        """
        Args:
            x: data to be clustered
        """
        
        print('performing kmeans clustering')
        #results = {'im2cluster':[],'centroids':[],'density':[]}
        seed = 0
        

        # intialize faiss clustering parameters
        d = x.shape[1] #  d = feature_dim
        k = int(self.num_cluster) # number of clusters for the run
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
        cfg.device = self.gpu_id    
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
        density = self.temperature*density/density.mean()  #scale the mean to temperature 
            
            # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids)
        centroids = nn.functional.normalize(centroids, p=2, dim=1) # normalize Centroids for unit-length   

        im2cluster = torch.LongTensor(im2cluster)               
        density = torch.Tensor(density)
            #------------------------------------------------------------
            
            # APPEND THE VALUES FOR EACH SINGLE CLUSTERING [25000,50000,100000]
        self.centroids = centroids
        self.density = density
        self.im2cluster = im2cluster    
            
        return im2cluster

    def cluster_data(self,x,augmented=False):
        """
        Args:
            x: data to be clustered
        """        
        print('performing kmeans clustering')

        seed = 0
        # intialize faiss clustering parameters
        d = x.shape[1] #  d = feature_dim
        k = int(self.num_cluster) # number of clusters for the run
            # CLUSTERING CONFIGURATION & PARAMETERS
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        #clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 2

            # CLUSTERING PARALLEL EXECUTION
        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id    
        index = faiss.GpuIndexFlatL2(res, d, cfg)  

            # START CLUSTERING
        clus.train(x, index)   
    #D = cluster distance for each sample to its cluster_centroid
    #I =  cluster assignment for each sample

        D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I] # cluster assignments in sample-order
            
        
        im2cluster = torch.LongTensor(im2cluster)               
        
        if augmented:
            self.im2cluster_I = im2cluster
        else:
            self.im2cluster = im2cluster
            
        return im2cluster

    def clustering(self):
        self.cluster_data(self.features,augmented=False)
        self.cluster_data(self.features_I,augmented=True)


    def batch_cluster_ids(self,batch):

        batch_cluster = torch.zeros(batch.shape)
        for j in range(len(batch_cluster)): batch_cluster[j] = self.im2cluster[int(batch[j])]
        self.batch_cluster = batch_cluster

        batch_cluster_I = torch.zeros(batch.shape)
        for j in range(len(batch_cluster_I)): batch_cluster_I[j] = self.im2cluster_I[int(batch[j])]
        self.batch_cluster_I


    def cluster_mask(self):

        mask_list = []
        for i in range(self.num_cluster):
            mask_list.append((self.batch_cluster == i).type(torch.LongTensor))

        return mask_list


    def cluster_batch(self,features,augmented=False):

        centers = []

        if augmented:
            bcl = self.batch_cluster_I
        else:
            bcl = self.batch_cluster

        for i in range(self.num_cluster):
            cluster_mask = bcl == i
            cluster_vectors = features[cluster_mask]
            centers.append(torch.sum(cluster_vectors,dim=0)/len(cluster_vectors))

        return bcl, centers




#dummy = torch.rand([5000,128])
#results = run_kmeans(dummy.cpu().numpy(),10)
#print(results['centroids'].shape)
#print(results['density'].shape)
#print(results['im2cluster'].shape)