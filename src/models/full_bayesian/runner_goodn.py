'''
Created on Aug 12, 2015

@author: alxcoh
'''

from runner_clust import *
import random
from multiprocessing import Pool
from kmedoids import kMedoids
from load_similarity_data import get_data
from runner_randomN import *

class GoodN_old(RandomN):
    def __init__(self, n = 20):
        self.data_matrix, self.items, self.features, self.sim_items, self.sim_feats = get_data()

        k = n
        if k > 200: k = 200

        for i in range(500):
            try:
                medoids, clusters = kMedoids(self.sim_items, k)
                break
            except:
                continue
                
        else:
            print "medoids failed"
            medoids = [0]
            clusters = {0:range(len(self.items))}

        super(GoodN, self).__init__(9, k, medoids)



    
        
    
