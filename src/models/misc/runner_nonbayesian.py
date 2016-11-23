'''
Created on Jan 30, 2016

@author: alxcoh
'''


from runner_clust import *
import random
from multiprocessing import Pool
import time
with open(base_path+"data/all_clusts_matrices5.pickle") as acm:
    all_clust_matrix = pickle.load(acm)

print "beginning clustfile"

feature_blacklist = []

class NonBayesianPlayer(ClustPlayer):
    def __init__(self):
        super(NonBayesianPlayer, self).__init__(9)
        self.last_player = ClustPlayer(9)
        self.last_player.update_all()
        self.last_probs = self.last_player.probabilities
    
    def expected_gain(self, feature):
        if self.knowledge == []:
            if features[int(feature)] == "IS IT MANUFACTURED?": return 1.0
            else: return 0.0
        else:
            sum_last = 0
            sum_current = 0
            if len(self.knowledge) - len(self.last_player.knowledge) > 1:
                self.last_player.knowledge = self.knowledge[:-1]
                self.last_player.update_all()
                self.last_probs = self.last_player.probabilities
                
            for k in self.knowledge:
                if k[0] == feature: return 0
                
            for o in range(1000):
                alternate_scale = 2*(data_matrix[o, feature]+1)
                sum_last += alternate_scale * self.last_probs[o]
                sum_current += alternate_scale * self.probabilities[o]
            return sum_current - sum_last
            
            '''
            for k in self.knowledge:
                if k[0] == feature: return 0
            sumtot = 0
            sumall = 0
            top10 = np.argsort(self.probabilities)[::-1][:10]
            for o in top10:
                sumtot += 2*(data_matrix[o, feature]+1)
            for o in range(1000):
                sumall += 2*(data_matrix[o, feature]+1)
            return sumtot / sumall
            '''

'''
p = NonBayesianPlayer()
p.play_game()
'''