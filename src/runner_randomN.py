'''
Created on Aug 12, 2015

@author: alxcoh
'''

from runner_clust import *
import random
from multiprocessing import Pool

class RandomN(ClustPlayer):
    def __init__(self, clusts, n=20, rands = [], strict_only_consider = True):
        if rands == []: self.randomN = np.array(random.sample(range(1000), n))
        else: self.randomN = np.array(rands)
        self.strict_only_consider = strict_only_consider
        super(RandomN, self).__init__(clusts)
        
    def get_prob_knowledge_from_items_new(self):

        arr = np.zeros(len(self.items_left), np.float64)
        for item in self.randomN:
            arr[item] = self.prob_knowledge_from_item(item)
        return arr
        
    def update_all(self):
        self.num_items_left = float(len(self.clusts))
        self.prior_prob = 1.0 / self.num_items_left
        
        self.prob_knowledge_from_items = self.get_prob_knowledge_from_items_new() if self.strict_only_consider else self.get_prob_knowledge_from_items()
        self.prob_knowledge_overall = np.sum(self.prob_knowledge_from_items)
        
        self.probabilities = self.prob_knowledge_from_items / self.prob_knowledge_overall
    
        self.entropy = entropy(self.probabilities[self.randomN])
        
    def entropy_with_new_knowledge(self, new_knowledge):
        ent = entropy(self.prob_with_new_knowledge(new_knowledge)[self.randomN]) 
        return self.entropy if math.isinf(ent) else ent
    
    def info_gain_probs(self, probs):
        return self.info_gain_ent(entropy(probs[self.randomN]))

    def get_randomN(self):
        return self.randomN
'''    
player = RandomN(9, n=20)
player.play_game()
'''

def updater(simulated_person):
    simulated_person.update_all()

class RandomN_averaged():
    def __init__(self, clusts, n_objects, n_simulated_people, rands = [], strict_only_consider = True):
        self.knowledge = []
        self.simulated_people = \
            [RandomN(clusts, n_objects, \
                rands[i] if len(rands) >= n_objects else [], \
                strict_only_consider) for i in range(n_simulated_people)]

    def update_all(self):
        #pool = Pool()

        for p in self.simulated_people:
            p.knowledge = self.knowledge

        map(updater, self.simulated_people)
        #pool.close()

    def expected_gain(self, question):
        return np.average(  [p.expected_gain(question) for p in self.simulated_people] )

    def expected_gains(self):
        return np.average( np.array( [p.expected_gains() for p in self.simulated_people ] ), 0 )

    def get_simulated_people(self):
        return [list(p.get_randomN()) for p in self.simulated_people]

class CompletelyRandom():
    def update_all(self):
        pass

    def expected_gain(self, question):
        return 1.0/float(len(features))

    def expected_gains(self):
        return [1.0/float(len(features))]*len(features)