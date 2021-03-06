'''
Created on Jul 9, 2015

@author: alxcoh
'''
import numpy as np
import cPickle as pickle
from multiprocessing import Pool

data_matrix = data_matrix_np = np.loadtxt(base_path+"data/intel_dat.txt")

with open(base_path+"data/intel_feat.txt", "r") as feats:
    features = [l[:-2] for l in feats]

with open(base_path+"data/intel_items.txt", "r") as items:
    items = [l[:-2] for l in items]
    
full_probability_matrix_goodmethod = [[ 0.70385265,  0.31086353,  0.18540700,  0.12579383,  0.06243025],
                                      [ 0.09839965,  0.27167072,  0.12949217,  0.08246418,  0.03858618],
                                      [ 0.09718542,  0.21443453,  0.37983041,  0.20084763,  0.09890584],
                                      [ 0.05635820,  0.11671840,  0.17166805,  0.31256788,  0.17611042],
                                      [ 0.04420408,  0.08631282,  0.13360238,  0.27832648,  0.62396730]]

def prob_to_index(prob):
    return (prob+1)*2

def index_to_prob(index):
    return (float(index)/2) - 1
    

with open(base_path+'data/all_clusts_divisive2.pickle') as ac:
    all_clusts = pickle.load(ac)
    
del all_clusts[-1]
all_clusts.append(range(len(items)))

def prob_resp_from_clust(clust, feat, val):
    sumtot = 0
    prior = 1.0/float(len(clust))
    
    for o in clust:
        #print features[feat], items[o], full_probability_matrix_goodmethod[val][int(prob_to_index(data_matrix[o, feat]))]
        sumtot += full_probability_matrix_goodmethod[val][int(prob_to_index(data_matrix[o, feat]))] * prior
    #print sumtot
    return sumtot


def dat_probs(clust):
    clusts = [np.where(np.array(clust) == val)[0] for val in list(set(clust))]
    num_clusts = len(clusts)
    prob_matrix = []
    for val in range(5):
        new_matrix = []
        for c in clusts:
            new_row = []
            for f in range(len(features)):
                new_row.append(prob_resp_from_clust(c, f, val))
            new_matrix.append(new_row)
        prob_matrix.append(new_matrix)
        
    return prob_matrix

def prob_resps_oneclust(cluster):
    prob_matrix = np.zeros((5, len(features)))
    for val in range(5):
        for f in range(len(features)):
            prob_matrix[val, f] = prob_resp_from_clust(cluster, f, val)
        
    return prob_matrix
        
def new_tup(clust):
    #print "STARTING:", len(set(clust))
    dat_matrx = dat_probs(clust)
    #print "FINISHED:", len(set(clust))
    return (clust, dat_matrx)

p = Pool()
'''
all_clusts_matrix = []
for c in all_clusts:
    all_clusts_matrix.append(new_tup(c))
'''
'''
all_clusts_matrix = p.map(new_tup, all_clusts)

    
with open(base_path+'data/all_clusts_matrices5.pickle', 'w') as acm2:
    pickle.dump(all_clusts_matrix, acm2)
    
'''
clust = all_clusts[0]
clusts = [np.where(np.array(clust) == val)[0] for val in list(set(clust))]
#print prob_resp_from_clust(clusts[0], features.index("WOULD YOU FIND IT IN A ZOO?"), 1)
#print prob_resp_from_clust(clusts[1], features.index("WOULD YOU FIND IT IN A ZOO?"), 1)