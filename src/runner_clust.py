'''
Created on Jul 28, 2015

@author: alxcoh
'''
from runner_numpy import *
    
with open("../data/all_clusts_matrices2.pickle") as acm:
    all_clust_matrix = pickle.load(acm)

feature_blacklist = []

import operator as op
def ncr(n, r):
    if n == 0 or r == 0 or n == r: return 1
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

def prob_to_index(prob):
    return (prob+1)*2

def index_to_prob(index):
    return (float(index)/2) - 1

cluster_amounts = {2:0, 4:1, 8:2, 16:3, 32:4, 64:5, 128:6, 256:7, 512:8, 999:9}

class ClustPlayer(OptimalPlayer): #ALL itemS AND FEATURES WILL BE REFERRED TO BY INDEX
    def __init__(self, clusts):
        self.clusts = None
        self.load_new_prob_matrix(clusts)
        super(ClustPlayer, self).__init__()
        
        self.update_all()
        
    def load_new_prob_matrix(self, clust_indx):
        self.clusts_index = all_clust_matrix[clust_indx][0]
        self.clusts = [np.where(np.array(self.clusts_index) == val)[0] for val in list(set(self.clusts_index))]
        self.data_probs = np.zeros((5, len(items), len(features)))
        for c in self.clusts:
            print [items[i] for i in c]
        self.clusts_index = np.array(self.clusts_index)
        self.clusts_index -= np.min(self.clusts_index)
        self.data_probs_clust = np.array(all_clust_matrix[clust_indx][1])
        for i, c in zip(range(1000), self.clusts_index):
            self.data_probs[:,i,:] = self.data_probs_clust[:,c,:]
        
 
    def prob_knowledge_from_clust(self, indx):
        clust_probs = self.data_probs_clust[:, indx] #a 2D slice of the array, rows being prob and column being feature
        prior = float(len(self.clusts[indx])) / float(len(items))
        return np.prod( np.fromiter((clust_probs[prob_to_index(r)][f] for f, r in self.knowledge), np.float64)) * \
            prior
 
    def get_prob_knowledge_from_items(self):
        prob_from_clusts = np.fromiter( ((self.prob_knowledge_from_clust(c)) for c in range(len(self.clusts)) ), np.float64)
        prob_from_items = np.zeros(len(items))
        for i, c in zip(range(1000), self.clusts):
            prob_from_items[c] = prob_from_clusts[i] / float(len(c))
        return prob_from_items

    def prob_with_new_knowledge(self, new_knowledge):
        feature, response = new_knowledge
        
        prob_response_for_clust = self.data_probs_clust[prob_to_index(response), :, feature] #2d slice, row is prob col is animal
        multr = np.zeros(len(items))
        for i, c in zip(range(1000), self.clusts):
            multr[c] = prob_response_for_clust[i]
        
        new_prob_knowledge_from_items = self.prob_knowledge_from_items * multr
                                                
        new_prob_knowledge_overall = np.sum(new_prob_knowledge_from_items)
        
        return new_prob_knowledge_from_items/new_prob_knowledge_overall
 
    def update_all(self):
        self.num_items_left = float(len(self.clusts))
        self.prior_prob = 1.0 / self.num_items_left
        
        self.prob_knowledge_from_items = self.get_prob_knowledge_from_items()
        self.prob_knowledge_overall = np.sum(self.prob_knowledge_from_items)
        
        self.probabilities = self.prob_knowledge_from_items / self.prob_knowledge_overall
    
        self.entropy = entropy(self.probabilities)

        #if len(self.items_guessed) > 0: self.prob_knowledge_from_items[np.array(self.items_guessed)] = 0
        
        
    
'''  
player = OptimalPlayer()
player.iterate()
'''
'''
player = ClustPlayer(3)
player.iterate()
'''
