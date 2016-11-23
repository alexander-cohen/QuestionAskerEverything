from runner_clust import *

class PositiveBiasModel(OptimalPlayer): 

    def expected_gain(self, feature):
        for k in self.knowledge:
            if k[0] == feature: return 0
            
        prob = self.prob_response(feature, 1.0)
        #print features[feature], "{:.3}".format(prob)
        return prob
        
