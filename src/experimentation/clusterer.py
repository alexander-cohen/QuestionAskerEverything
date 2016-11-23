'''
Created on Nov 23, 2015

@author: alxcoh
'''
import numpy as np
import data_loader
import scipy.stats as scistats
from runner_clust import ClustPlayer
from multiprocessing import Pool
data_matrix, _, features, objects = data_loader.get_data()

full_probability_matrix_goodmethod = [[ 0.70385265,  0.31086353,  0.18540700,  0.12579383,  0.06243025],
                                      [ 0.09839965,  0.27167072,  0.12949217,  0.08246418,  0.03858618],
                                      [ 0.09718542,  0.21443453,  0.37983041,  0.20084763,  0.09890584],
                                      [ 0.05635820,  0.11671840,  0.17166805,  0.31256788,  0.17611042],
                                      [ 0.04420408,  0.08631282,  0.13360238,  0.27832648,  0.62396730]]

logfile = open("clustmaker_logfile.txt", 'w')

def get_clusters_given_posterior(n_clusters, current_clustering, posterior = np.repeat(0.01, (1000)), knowledge = []):
    #probget = np.all(0.2, (n_clusters, 218, 5), dtype=np.float64)
 
    did_shift = True
    clustering = current_clustering[:]
    iterations = 0
    while did_shift:
        iterations += 1
        print "\n\n*****************\nIterations:", iterations, "\n", "Clustering:", clustering, "\n*********************\n\n\n"
        did_shift = False
        #curval = cluster_validity(clustering)
        p = Pool()
        new_clustering = p.map(dofor, zip(range(1000), [clustering]*1000, [n_clusters]*1000, [knowledge]*1000, [posterior]*1000))
        if new_clustering != clustering: did_shift = True
        clustering = new_clustering
        logfile.write(str(clustering) + '\n')   
           
    print "\n\n\n\nCOMPLETED\n\n\n\n", clustering
    return clustering


    '''
    def test_kp(clustering):
        orders = [np.size(clustering == n)[0] for n in range(n_clusters)]
        new_posterior = [orders[clustering[i] for i in range(1000) ]
    
    def m():
    '''

def e(clustering, n_clusters):
    probget = np.zeros((218, n_clusters, 5), dtype=np.float128)
    for item in range(1000):
        for feature in range(218):
            probget[feature, clustering[item], data_matrix[item, feature]] += 1.0
    #print np.shape(np.sum(probget, 2))
    for cluster in range(n_clusters):
        for feature in range(218):
            probget[feature, cluster] /= np.sum(probget[feature, cluster, :])
    '''c
        
    '''
            
    new_probget = np.zeros((218, n_clusters, 5), dtype=np.float128)
    
    for cluster in range(n_clusters):
        for feature in range(218):
            for answer in range(5):
                #print probget[feature, cluster, 0], probget[feature, cluster, 1], probget[feature, cluster, 2]
                #print [probget[feature, cluster, val] for val in range(5)]
                new_probget[feature, cluster, answer] = np.sum( [full_probability_matrix_goodmethod[answer][val]* probget[feature, cluster, val] for val in range(5)] )
                
    return np.swapaxes(new_probget, 0, 2)

def cluster_validity(tup):
    clustering, posterior, knowledge, n_clusters = tup
    myplayer = ClustPlayer(0)
    myplayer.knowledge = knowledge
    myplayer.set_custom_prob_matrix(clustering, e(clustering, n_clusters))
    myplayer.update_all()
    return scistats.entropy(posterior, myplayer.probabilities)

def newclustering(index, new, oldclustering, n_clusters, knowledge, posterior):
    newclustering = oldclustering[:]
    newclustering[index] = new
    return newclustering, posterior, knowledge, n_clusters

def dofor(tup):
    i, clustering, n_clusters, knowledge, posterior = tup
    #old = clustering[i]
    c_i = np.argmax( [cluster_validity(newclustering(i, c, clustering, n_clusters, knowledge, posterior)) for c in range(n_clusters)] )
    clustering[i] = c_i
    print "Finished:", i
    return c_i


import cPickle as pickle
with open(base_path+"data/all_clusts_matrices5.pickle") as acm:
    all_clust_matrix = pickle.load(acm)
get_clusters_given_posterior(8, all_clust_matrix[2][0])
logfile.close()
