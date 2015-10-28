'''
Created on Jul 28, 2015

@author: alxcoh
'''
from runner_numpy import *
import random
from multiprocessing import Pool
    
with open("../data/all_clusts_matrices5.pickle") as acm:
    all_clust_matrix = pickle.load(acm)

print "beginning clustfile"

feature_blacklist = []


cluster_amounts = {2:0, 4:1, 8:2, 16:3, 32:4, 64:5, 128:6, 256:7, 512:8, 999:9}

class ClustPlayer(OptimalPlayer): #ALL itemS AND FEATURES WILL BE REFERRED TO BY INDEX
    def __init__(self, clusts):
        self.clusts = None
        self.load_new_prob_matrix(clusts)
        super(ClustPlayer, self).__init__()
        self.load_new_prob_matrix(clusts)
        self.update_all()
        
    def load_new_prob_matrix(self, clust_indx):
        self.clusts_index = all_clust_matrix[clust_indx][0]
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
        
 
    def prob_knowledge_from_clust(self, indx):
        clust_probs = self.data_probs_clust[:, indx] #a 2D slice of the array, rows being prob and column being feature
        prior = float(len(self.clusts[indx])) / float(len(items))
        #print '\nprob knowledge from', indx
        #for f, r in self.knowledge:
            #print clust_probs[prob_to_index(r)][f]
        #print '\n'
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
        #print 
        #if len(self.items_guessed) > 0: self.prob_knowledge_from_items[np.array(self.items_guessed)] = 0
        
        
    
'''  
player = OptimalPlayer()
player.iterate()
'''
'''
player = ClustPlayer(9)
player.play_game()
'''
'''
itms = random.sample(range(len(items)), 50)

def getgamelist(item):
    print item, items[item]
    player = ClustPlayer(9)
    this_game = [item, items[item]]
    questions = []
    for x in range(20):
        choice, infogains = player.computer_iterate(items[item])
        
        this_question = [choice, infogains, np.argsort(infogains), [player.features_left[f] for f in np.argsort(infogains)[::-1]]]
        #print len(this_question[3])
        questions.append(this_question)
        player.features_left.remove(choice)
        
    this_game.append(questions)
    return this_game
        
        
p = Pool()
all_games = p.map(getgamelist, itms)
        
with open("../data/all_games_for_experiment.pickle", 'w') as agfe:
    pickle.dump(all_games, agfe)
        


'''
#print_for_test = False
#player = ClustPlayer(0)
'''
expected = player.get_unnormed_gains()
print expected[features.index("WOULD YOU FIND IT IN A ZOO?")]
'''
'''
player.knowledge = [(features.index("WOULD YOU FIND IT IN A ZOO?"), 1.0)]
print "knowledge set 1"
print player.prob_knowledge_from_clust(0)
print player.prob_knowledge_from_clust(1)

player.knowledge = [(features.index("WOULD YOU FIND IT IN A ZOO?"), -1.0)]
print "knowledge set -1"
print player.prob_knowledge_from_clust(0)
print player.prob_knowledge_from_clust(1)
player.knowledge = []
'''
'''
print player.prob_response("WOULD YOU FIND IT IN A ZOO?", 1.0)
print player.prob_response("WOULD YOU FIND IT IN A ZOO?", -1.0)

#print expected[features.index("DOES IT MAKE A SOUND?")]
'''