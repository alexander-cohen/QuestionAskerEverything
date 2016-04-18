'''
Created on Jul 28, 2015

@author: alxcoh
'''

from runner_numpy import *
import random
from multiprocessing import Pool
import time
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
    
    def set_custom_prob_matrix(self, clustering, prob_resp, is_real_index = False, clusts_index = None, byIndex = True):
        #t = time.time()
        if byIndex and not is_real_index: 
            self.clusts_index = np.array(clustering)
            self.clusts = [np.where(np.array(self.clusts_index) == val)[0] for val in list(set(self.clusts_index))]
            
        elif is_real_index:
            t = time.time()
            self.clusts = clustering
            self.clusts_index = clusts_index
            #print "\npart 1:", time.time() - t
            
        else: 
            #t = time.time()
            self.clusts_index = [ which_cluster(clustering, item) for item in range(1000) ]
            #print "\npart 1:", time.time() - t
            #t = time.time()
            self.clusts = np.array(clustering)
            #print "part 2:", time.time() - t
            
        #print self.clusts_index
        #print self.clusts_index
        
        
        #t = time.time()
        #should be 5x|c|x218
        t = time.time()
        self.data_probs_clust = np.array(prob_resp)
        #print np.shape(self.data_probs_clust)
        #print np.shape(self.data_probs_clust[0])
        #print np.shape(self.data_probs_clust[:,1])
        #print np.shape(self.data_probs_clust[:,:,0])
        #print '\n'
        '''
        print "part 2:", time.time() - t
        t = time.time()
        self.data_probs = np.zeros((5, len(items), len(features)))
        print "part 3:", time.time() - t
        '''
        
        t = time.time()
        self.data_probs = np.array([self.data_probs_clust[:,c] for c in self.clusts_index])
        self.data_probs = np.swapaxes(self.data_probs, 0, 1)
        #print np.shape(self.data_probs)
        #print "part 1:", time.time() - t
        #prob resp should be 5x1000x218
        '''
        t = time.time()
        for i, c in zip(range(len(items)), self.clusts_index):
            #print i, c, np.shape(self.data_probs_clust)
            self.data_probs[:,i,:] = self.data_probs_clust[:,c]
        #print "shape:", np.shape(self.data_probs), np.shape(other)
        print "part 5:", time.time() - t
        '''
    def load_new_prob_matrix(self, clust_indx):
        self.clusts_index = all_clust_matrix[clust_indx][0]
        self.clusts = [np.where(np.array(self.clusts_index) == val)[0] for val in list(set(self.clusts_index))]
        self.data_probs = np.zeros((5, len(items), len(features)))
        
        for c in self.clusts:
            pass
            #print len(c), [items[i] for i in c]

        self.clusts_index = np.array(self.clusts_index)
        self.clusts_index -= np.min(self.clusts_index)
        self.data_probs_clust = np.array(all_clust_matrix[clust_indx][1])
        for i, c in zip(range(len(items)), self.clusts_index):
            self.data_probs[:,i,:] = self.data_probs_clust[:,c,:]
        
 
    def prob_knowledge_from_clust(self, indx):
        clust_probs = self.data_probs_clust[:, indx] #a 2D slice of the array, rows being prob and column being feature
        prior = float(len(self.clusts[indx])) / float(len(items))
        '''
        if len(self.knowledge_indexfrom) != len(self.knowledge):
            self.knowledge_indexfrom = np.array( [[ prob_to_index(prob), f] for f, prob in self.knowledge], dtype = np.int16) 
        '''
        #print '\nprob knowledge from', indx
        #for f, r in self.knowledge:
            #print clust_probs[prob_to_index(r)][f]
        #print '\n'
        '''
        print '\n', self.knowledge
        for f, r in self.knowledge:
            print r, f, prob_to_index(r), clust_probs[prob_to_index(r)][f]
        '''

        return np.prod( np.fromiter((clust_probs[prob_to_index(r)][f] for f, r in self.knowledge), np.float64)) * \
            prior
        '''
        print self.knowledge_indexfrom.T, np.shape(clust_probs), clust_probs[[[0], [78]]]
        return 1.0 if len(self.knowledge_indexfrom) == 0 else np.prod( clust_probs[list(self.knowledge_indexfrom.T)] ) * \
           prior
        '''
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
def which_cluster(clusters, item):
    for clust_index, c in zip(range(len(clusters)), clusters):
        if item in c: return clust_index
    if item == 0: print "Could not find:", items[item], item, item in clusters[2], '\n', '\n'.join([repr(list(elem)) for elem in clusters])
 


'''
player = ClustPlayer(9)
player.play_game()
'''
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
'''
player = ClustPlayer(9)
player.play_game()
'''