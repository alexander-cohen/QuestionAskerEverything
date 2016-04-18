'''
Created on Aug 12, 2015

@author: alxcoh
'''
from runner_clust import *
from tree_variational import *
import scipy.stats as scistats
import time

#print "starting loading variational tree"

with open('../data/clusttree.pickle', 'r') as clusttree_pickle_file:
    clusttree = pickle.load(clusttree_pickle_file)
    
#print "done loading variational tree"



import copy

with open("../data/all_clusts_matrices5.pickle") as acm:
    all_clust_matrix = pickle.load(acm)

logfile = open("variational_clust_log", 'w')


class VariationalPlayer(ClustPlayer):
    
    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.amount_retreat = 0
        self.can_write = True
        
    def start(self, clusts):
        self.numclusts = clusts
        self.clusts = None
        self.items_left = range(len(items)) #list of items that can be guessed
        self.items_guessed = [] #list of items that have been guessed
        self.question_num = 1
        self.num_items_left = len(self.items_left) #this variable should always refer to the number of items left
        self.prior_prob = 1.0 / self.num_items_left #prior distribution over items (uniform for now)
        
        self.features_left = range(len(features)) #features are referred to by index, this is the features we cn still choose
        
        for f in feature_blacklist:
            self.features_left.remove(features.index(f)) #remove bad features
        
        self.probabilities = np.tile(self.prior_prob, self.num_items_left) #for now, everything has prior prob
        self.prob_knowledge_from_items = np.tile(1.0, self.num_items_left) #empty knowledge set, so 1.0 prob
        self.data_probs = data_probs_temp
        self.entropy = entropy(self.probabilities) #entropy of distribution
        
        self.cur_money = 2.0
        self.money_deplete = 0.5
        self.num_item_rand = 20
        self.num_item_end_left = self.num_item_rand-1
        
        self.update_all() #updates all information

    
    def find_for_situation(self, open_set, num_clusts, realprob):
        #print "Finding situation, total clusts:", num_clusts, " clusts on:", (len(open_set)+1)
        all_leaf = True
        best_index = 0
        KL_mode = True
        want_big_entropy = True
        best_gain = 0 if want_big_entropy else 10**10
        best_clustering = None
        best_prob_resp = None
        best_prob_clust = None
        t = time.time()

        #if self.can_write: logfile.write("\n=========================\n=========================\n")
        for option, index in zip(open_set, range(1000)): #each option is a ClustTree node
            t = time.time()
            if option.leaf: continue
            leftclusts = None if option.left == None else option.left.val
            rightclusts = None if option.right == None else option.right.val
           
            #print leftclusts
            #print rightclusts
            
            
            cluster_indexes = np.zeros(1000)
            
            new_clusts = [leftclusts, rightclusts]
            for node in (open_set[:index] + open_set[index+1:]):
                new_clusts += [node.val]
                
        
            for c, i in zip(new_clusts, range(1000)):
                for elem in c:
                    cluster_indexes[elem] = i
            
            #print np.shape(np.array(new_clusts))
            #print '\n'
            #print "\n".join([str([i for i in c]) for c in new_clusts])
            #print '\n'
            
            prob_resp = [option.left.prob_resp, option.right.prob_resp]
            #print "beep", np.shape(prob_resp[0])
            for node in (open_set[:index] + open_set[index+1:]):
                #print "Shape:", np.shape(node.prob_resp)
                prob_resp += [node.prob_resp]
            prob_resp = np.array(prob_resp)
            prob_resp = np.swapaxes(prob_resp, 0, 1) #make it the correct shape
            #print np.shape(prob_resp)
            #print np.shape(np.array(all_clust_matrix[option.depth+1][1])[:,option.left.lindex,:])
            #print "\npart 1:", time.time() - t
            t2 = time.time()
            newplayer = ClustPlayer(0)
            t = time.time()
            newplayer.set_custom_prob_matrix(new_clusts, prob_resp, True, cluster_indexes)
            #print "Total:", time.time() - t
            t = time.time()
          
            newplayer.knowledge = self.knowledge
            newplayer.update_all()
            #print "Part 2:", time.time() - t
            #print "part 1:", time.time() - t2
            t3 = time.time()
            
            probs_clusts = np.array([newplayer.probabilities[c[0]] * len(c) for c in new_clusts])
            #print scistats.entropy(newplayer.probabilities, realprob)
            
            if KL_mode: gain = scistats.entropy(newplayer.probabilities, realprob)
            else: 
                #probs_clusts = np.array([newplayer.probabilities[c[0]] * len(c) for c in new_clusts])
                gain = scistats.entropy(probs_clusts)
                #print gain, list(probs_clusts)
                
                

                
                
            if index == 0:
                best_gain = gain
                best_index = index
                best_clustering = new_clusts
                best_prob_resp = prob_resp
                best_prob_clust = probs_clusts
                
            elif (index > (num_clusts - self.amount_retreat)) and (gain > best_gain and want_big_entropy) or (gain < best_gain and not want_big_entropy):
                best_gain = gain
                best_index = index
                best_clustering = new_clusts
                best_prob_resp = prob_resp
                best_prob_clust = probs_clusts
                
            #logfile.write( "{:0.4}, {:0.4}   {!s}".format(gain, best_gain, list(probs_clusts) ))

            elif len(option.val) >= len(open_set[best_index].val): #if there is a tie, use depth so you prefer smaller ones
                best_gain = gain
                best_index = index
                best_clustering = new_clusts
                best_prob_resp = prob_resp
                best_prob_clust = probs_clusts


            #print "part 2:", time.time() - t3
        #logfile.write( '\n\n\n' )
        #print best_index, best_gain
        #print best_prob
        #print "\nClusters:", (len(open_set)+1), "\ntime taken:", time.time() - t
        #print best_gain
        #logfile.write( '\n\n' )
        
        for p, c in zip(best_prob_clust, best_clustering):
            pass
            #logfile.write( "{:.4f}".format(p).ljust(6) + str(len(c)).rjust(4) + "  " + str( [items[elem] for elem in c] ) + '\n')
        
        if num_clusts == len(open_set) + 1: 
            #print [(items[i], best_prob[i]) for i in range(1000)]
            #print [(items[i], realprob[i]) for i in range(1000)]
            
            if self.can_write:
                logfile.write( '\n\n*****************\n')
                logfile.write( "Num clusts: " + str(num_clusts))
               
                logfile.write( "\nKnowledge: " + str([[features[f], r] for f, r in self.knowledge]) )
                logfile.write( "\nprobability, amount, items")
                logfile.write( '\n\n' )
                for p, c in zip(probs_clusts, best_clustering):

                    logfile.write( "{:.4f}".format(p) + str(len(c)).rjust(4) + "  " + str( [items[elem] for elem in c] ) + '\n')
                
            #logfile.write( "\n\n" + "\n".join([str(len(c)) + " " + str([items[i] for i in c]) for c in best_clustering]) + '\n')
            return best_clustering, best_prob_resp, best_gain
        else: return self.find_for_situation( [open_set[best_index].left, open_set[best_index].right] + \
                                        open_set[:best_index] + open_set[best_index+1:], num_clusts, realprob)

            
    def set_situation(self, clusts):
        self.amount_retreat = clusts
        t = time.time()
        logfile = open("variational_clust_log", 'a')
        #logfile.write('\n\n')
        
        full_player = ClustPlayer(9)
        full_player.knowledge = self.knowledge
        full_player.update_all()
        
        best_clusters, best_prob_resp, new_entropy = self.find_for_situation([clusttree], clusts, full_player.probabilities)
        #print "Num clusters:", len(best_clusters)
        self.set_custom_prob_matrix(best_clusters, best_prob_resp, False, None, False)
        logfile.close()
        #print "\nClusts:", clusts
        #print "Time:", time.time() - t
        
    def update_all(self):
        t = time.time()
        self.set_situation(self.numclusts)
        #print "\n\n**********\nCompleted updating:", self.numclusts
        #print "Time:", time.time() - t
        #print "Knowledge:", self.knowledge
         
        self.num_items_left = float(len(self.clusts))
        self.prior_prob = 1.0 / self.num_items_left
        
        self.prob_knowledge_from_items = self.get_prob_knowledge_from_items()
        self.prob_knowledge_overall = np.sum(self.prob_knowledge_from_items)
        
        self.probabilities = self.prob_knowledge_from_items / self.prob_knowledge_overall
    
        self.entropy = entropy(self.probabilities)

myplayer = VariationalPlayer([[features.index('IS IT MANMADE?'), 1.0], \
                              [features.index('CAN YOU HOLD IT IN ONE HAND?'), 1.0], \
                              [features.index('IS IT MADE OF METAL?'), 0.0]])

'''
myplayer.start(64)
print_for_test = True
myplayer.play_game_computer('desk')
'''