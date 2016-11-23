'''
Created on Aug 12, 2015

@author: alxcoh
'''

from runner_clust import *

class MaxprobPlayer(ClustPlayer):
    def expected_gain(self, feature):
        eig = 0
        for i in range(5):
            print "Prob", i, self.prob_response(feature, index_to_prob(i)), np.max(self.prob_with_new_knowledge((feature, index_to_prob(i))))
            eig += self.prob_response(feature, index_to_prob(i)) * np.max(self.prob_with_new_knowledge((feature, index_to_prob(i))))
        return  eig