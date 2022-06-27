import numpy as np
from sklearn.utils import shuffle
from sklearn import cluster
import sklearn
from sklearn.decomposition import IncrementalPCA
from tqdm import trange, tqdm
from scipy.optimize import linear_sum_assignment





def batches(l, n):
    for i in range(0, len(l), n): # step_size = n, [i] is a multiple of n
        yield l[i:i + n] # teilt das array [l] in batch sub_sequences der Länge n auf




def get_cost_matrix(y_pred, y, nc=1000): # C[ground-truth_classes,cluster_labels] counts all instances with a given ground-truth and cluster_label
    C = np.zeros((nc, y.max() + 1))
    for pred, label in zip(y_pred, y):
        C[pred, label] += 1
    return C 



def assign_classes_hungarian(C): # rows are (num. of) clusters and columns (num. of) ground-truth classes
    row_ind, col_ind = linear_sum_assignment(C, maximize=True) # assume 1200 rows(clusters) and 1000 cols(classes)
    ri, ci = np.arange(C.shape[0]), np.zeros(C.shape[0]) # ri contains all CLASS indexes as integer from 0 --> num_classes
    ci[row_ind] = col_ind # assignment of the col_ind[column nr. = CLASS_ID] to the [row nr. = cluster_ID/index]

    # for overclustering, rest is assigned to best matching class
    mask = np.ones(C.shape[0], dtype=bool)
    mask[row_ind] = False # True = alle cluster die nicht durch [linear_sum_assignment] einer Klasse zugeordnet wurden
    ci[mask] = C[mask, :].argmax(1) # Für weitere Cluster über die Anzahl Klassen hinaus, ordne die Klasse mit der größten Häufigkeit zu 
    return ri.astype(int), ci.astype(int) # at each position one assignment: ri[x] = index of cluster <--> ci[x] = classID assigned to cluster


def assign_classes_majority(C):
    col_ind = C.argmax(1) # assign class with the highest occurence to the cluster (row)
    row_ind = np.arange(C.shape[0]) # clusterID at position in both arrays (col_ind and row_ind)

    # best matching class for every cluster
    mask = np.ones(C.shape[0], dtype=bool)
    mask[row_ind] = False

    return row_ind.astype(int), col_ind.astype(int)



#cluster_idx,class_idx = assign_classes_hungarian(C_train)
#rid,cid = assign_classes_majority(C_train)

def accuracy_from_assignment(C, row_ind, col_ind, set_size=None):
    if set_size is None:
        set_size = C.sum()
    cnt = C[row_ind, col_ind].sum() # sum of all correctly (class)-assigned instances that contributes to the Cluster's ClassID decision
    # (that caused the decision)
    return cnt / set_size # If all clusters would have only instaces of one unique class, this value becomes = 1

def get_best_clusters(C, k=3):
    Cpart = C / (C.sum(axis=1, keepdims=True) + 1e-5) # relative Häufigkeit für jedes Cluster label
    Cpart[C.sum(axis=1) < 10, :] = 0 # Schwellwert für die Mindestanzahl Instanzen mit ground-truth_class
    # setzt bestimmte relative Häufigkeiten auf 0 (aus der Bewertung entfernt)
    # print('as', np.argsort(Cpart, axis=None)[::-1])
    
    # np.argsort(Cpart, axis=None)[::-1] # flattened indices in umgekehrt_absteigender Abfolge (sonst aufsteigender Reihenfolge)
    # Cpart.shape = (1000,1000)
    ind = np.unravel_index(np.argsort(Cpart, axis=None)[::-1], Cpart.shape)[0]  # first-dimension indices in C of good clusters (highest single frequency correlation)
    _, idx = np.unique(ind, return_index=True) # index of the first occurence of the unique element in $[ind]
    # idx = 1000 aus einer Million indices (höchst-erst-bestes aus jeder ground-truth), keine Duplikate
    cluster_idx = ind[np.sort(idx)]  # unique indices of good clusters (von groß nach klein)
    # nimmt den ersten Wert eines auftauchenden classIndex value von [ind] und notiert sich nur die Indexposition in [ind] dabei
    # die Werte werden von Beginn bis Ende in der Reihenfolge von [ind] ausgewählt; somit ist der kleinste Wert von idx auch 
    # der erste Wert von [ind], weitere Werte mit dem gleichen classIndex werden übersprungen und der zweite Werte ist somit der
    # nächsthöchste classIndex von [ind], somit hat man die besten classID's in absteigender Reihenfolge    
    accs = Cpart.max(axis=1)[cluster_idx] # die accuracies (höchste Wahrscheinlichkeit von Cpart) der besten classes/cluster (als ID)
    good_clusters = cluster_idx[:k] # selects the k best clusters
    best_acc = Cpart[good_clusters].max(axis=1)
    best_class = Cpart[good_clusters].argmax(axis=1)
    #print('Best clusters accuracy: {}'.format(best_acc))
    #print('Best clusters classes: {}'.format(best_class))
    outstring = ''
    for i in range(k):
        outstring += str(i)
        outstring += ' ,'
        outstring += str(good_clusters[i])
        outstring += ','
        outstring += str(best_class[i])
        outstring += ','
        outstring += str(best_acc[i])        
        outstring += '\n'
        
    print(outstring)
    
    return {'best_clusters': good_clusters, 'classes': best_class, 'accuracies': best_acc}


def train_pca(X_train,n_comp):
    bs = max(4096, X_train.shape[1] * 2)
    transformer = IncrementalPCA(batch_size=bs,n_components=n_comp)  #
    for i, batch in enumerate(tqdm(batches(X_train, bs), total=len(X_train) // bs + 1)):
        transformer = transformer.partial_fit(batch)
        # break
    print(transformer.explained_variance_ratio_.cumsum())
    return transformer

def transform_pca(X, transformer):
    n = max(4096, X.shape[1] * 2)
    n_comp = transformer.components_.shape[0]
    X_ = np.zeros((X.shape[0],n_comp))
    for i in trange(0, len(X), n):
        X_[i:i + n] = transformer.transform(X[i:i + n])
        # break
    return X_

def kmeans_eval(X_train,y_train,num_classes,n_components):

    #data = np.load(numpy_features)
    #X_train, y_train = data['features'], data['labels']
    #batch_size = 512
    #epochs = 2

    #num_classes = 10
    #n_components = 128

    transform = train_pca(X_train,n_components)
    train_features = transform_pca(X_train,transform)

    km = cluster.KMeans(n_clusters=num_classes).fit(train_features)

    #minib_k_means = cluster.MiniBatchKMeans(n_clusters=10, batch_size=batch_size, max_no_improvement=None)
        
    pred = km.predict(train_features)

    C_train = get_cost_matrix(pred, y_train, num_classes)

    message = 'val'
    y_pred = pred
    y_true = y_train
    train_lin_assignment = assign_classes_hungarian(C_train)
    train_maj_assignment = assign_classes_majority(C_train)

    acc_tr_lin = accuracy_from_assignment(C_train, *train_lin_assignment)
    #acc_tr_maj = accuracy_from_assignment(C_train, *train_maj_assignment)

    result_dict = {} # get_best_clusters(C_train,k=5)


    ari = sklearn.metrics.adjusted_rand_score(y_true, y_pred)
    v_measure = sklearn.metrics.v_measure_score(y_true, y_pred)
    ami = sklearn.metrics.adjusted_mutual_info_score(y_true, y_pred)
    fm = sklearn.metrics.fowlkes_mallows_score(y_true, y_pred)

    #headline = 'method,ACC,ARI,AMI,FowlkesMallow,'
    #print('\ncluster performance:\n')
    #print(eval_name+'  ,'+str(acc_tr_lin)+', '+str(ari)+', '+str(v_measure)+', '+str(ami)+', '+str(fm))

    result_dict['ACC'] = acc_tr_lin
    result_dict['ARI'] = ari
    result_dict['V_measure'] = v_measure
    result_dict['fowlkes_mallows'] = fm
    result_dict['AMI'] = ami

    #print("\n{}: ARI {:.5e}\tV {:.5e}\tAMI {:.5e}\tFM {:.5e}".format(message, ari, v_measure, ami, fm))

    return result_dict