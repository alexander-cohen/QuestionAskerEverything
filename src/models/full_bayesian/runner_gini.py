from runner_clust import *

class GiniPlayer(ClustPlayer):
    def expected_gain(self, feature):
        eig = 0
        for i in range(5):
            print "Prob", i, self.prob_response(feature, index_to_prob(i)), np.max(self.prob_with_new_knowledge((feature, index_to_prob(i))))
            eig += self.prob_response(feature, index_to_prob(i)) * np.sum(self.prob_with_new_knowledge((feature, index_to_prob(i))) ** 2)
        return  eig