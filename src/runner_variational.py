'''
Created on Aug 12, 2015

@author: alxcoh
'''
from runner_clust import *
import copy

class VariationalPlayer(ClustPlayer):
    def build_from(self, clust_amount):
        return int(math.log(clust_amount, 2)) + 1
        return int(math.log(clust_amount, 2)) + 1
        return 9
        
    
    def build_clusts(self, base, amt):
        clusts = copy.deepcopy(base)
        
        temp = ClustPlayer(9)
        temp.knowledge = self.knowledge
        temp.features_left = self.features_left
        temp.update_all()
        temp_probs = temp.probabilities.copy()
        del temp
        
        new_clusts = []
        while len(clusts) > amt:
            for c in clusts:
                
            
        return clusts
        
        
    
    def load_new_prob_matrix_amt(self, clusts_amt):
        build_from = self.build_from(clusts_amt)
        self.clusts_index = all_clust_matrix[build_from][0]
        self.clusts = [np.where(np.array(self.clusts_index) == val)[0] for val in list(set(self.clusts_index))]
        
        
        
        self.data_probs = np.zeros((5, len(items), len(features)))
        
        for c in self.clusts:
            pass
            #print [items[i] for i in c]

        self.clusts_index = np.array(self.clusts_index)
        self.clusts_index -= np.min(self.clusts_index)
        self.data_probs_clust = np.array(all_clust_matrix[clust_indx][1])
        for i, c in zip(range(len(items)), self.clusts_index):
            self.data_probs[:,i,:] = self.data_probs_clust[:,c,:]
    
    def load_new_prob_matrix(self, clust_indx):
        
        raise Exception("should not be here")

        