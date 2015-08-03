'''
Created on Jul 4, 2015

@author: alxcoh
'''
import numpy as np
from scipy.stats import entropy
import pickle
import math
from matplotlib import pyplot as plt

data_matrix = np.loadtxt("../data/intel_dat.txt")


with open("../data/intel_feat.txt", "r") as feats:
    features = [l[:-2] for l in feats]

with open("../data/intel_items.txt", "r") as items:
    items = [l[:-2] for l in items]
   
import operator as op
def ncr(n, r):
    if n == 0 or r == 0 or n == r: return 1
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom

#from data bad method
full_probability_matrix_badmethod =  [[ 0.86536632,  0.41078120,  0.25688982,  0.15429598,  0.04290900],
                                      [ 0.04755257,  0.19152609,  0.13691758,  0.08676840,  0.01680109],
                                      [ 0.05165025,  0.18489911,  0.32072590,  0.20624163,  0.09186447],
                                      [ 0.01979656,  0.11932672,  0.17985315,  0.24463726,  0.17145633],
                                      [ 0.01563430,  0.09346688,  0.10561355,  0.30805673,  0.67696911]]

#from data good method
full_probability_matrix_goodmethod = [[ 0.70385265,  0.31086353,  0.18540700,  0.12579383,  0.06243025],
                                      [ 0.09839965,  0.27167072,  0.12949217,  0.08246418,  0.03858618],
                                      [ 0.09718542,  0.21443453,  0.37983041,  0.20084763,  0.09890584],
                                      [ 0.05635820,  0.11671840,  0.17166805,  0.31256788,  0.17611042],
                                      [ 0.04420408,  0.08631282,  0.13360238,  0.27832648,  0.62396730]]

full_probability_matrix_madeup = [[0.60, 0.25, 0.09, 0.04, 0.02 ],
                                  [0.25, 0.38, 0.22, 0.10, 0.05 ],
                                  [0.09, 0.22, 0.38, 0.22, 0.09 ],
                                  [0.05, 0.10, 0.22, 0.38, 0.25 ],
                                  [0.02, 0.04, 0.09, 0.25, 0.60 ]]

all_probs = [np.zeros(np.shape(data_matrix)) for x in range(5)]

for i in range(5):
    p = (float(i)/2.0) - 1.0
    
    for r in range(np.shape(data_matrix)[0]):
        for c in range(np.shape(data_matrix)[1]):
            all_probs[i][r][c] = full_probability_matrix_goodmethod[i][int((data_matrix[r][c] + 1)*2)]


all_probs = np.array(all_probs)
all_probs.dump("../pickled_data/data_probs.pickle")

with open("../pickled_data/data_probs.pickle", 'r') as d_probs:
    data_probs = pickle.load(d_probs)

print np.shape(data_probs)

num_items_tot = len(items)

feature_blacklist = []



def prob_to_index(prob):
    return (prob+1)*2

def index_to_prob(index):
    return (float(index)/2) - 1



class Player(object): #ALL itemS AND FEATURES WILL BE REFERRED TO BY INDEX
    def __init__(self):
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
        
        self.knowledge = [] #will add feature/response pairs
        
        self.entropy = entropy(self.probabilities) #entropy of distribution
        self.cur_money = 2.0
        self.money_deplete = 0.5
        self.num_item_rand = 20
        self.num_item_end_left = self.num_item_rand-1
        self.update_all() #updates all information
        
    def prob_knowledge_from_item(self, item):
        item_probs = data_probs[:, item] #a 2D slice of the array, rows being prob and column being feature
        return np.prod( np.fromiter((item_probs[prob_to_index(r)][f] for f, r in self.knowledge), np.float64)) * \
                self.prior_prob
        
    def get_prob_knowledge_from_items(self):
        return np.fromiter( ((self.prob_knowledge_from_item(item)) for item in self.items_left ), np.float64)
        
    def update_all(self):
        self.num_items_left = len(self.items_left)
        self.prior_prob = 1.0 / self.num_items_left
        
        self.prob_knowledge_from_items = self.get_prob_knowledge_from_items()
        self.prob_knowledge_overall = np.sum(self.prob_knowledge_from_items)
        
        
#         if len(self.items_guessed) > 0: self.prob_knowledge_from_items[np.array(self.items_guessed)] = 0
        
        
        
        self.probabilities = self.prob_knowledge_from_items / self.prob_knowledge_overall
        
        self.entropy = entropy(self.probabilities)
        
    def entropy_with_new_knowledge(self, new_knowledge):
        feature, response = new_knowledge

        prob_response_for_animal = data_probs[prob_to_index(response), :, feature] #2d slice, row is prob col is animal

        new_prob_knowledge_from_items = self.prob_knowledge_from_items * \
                                               np.fromiter(  (prob_response_for_animal[o]  for o in self.items_left), np.float64)
                                                
        new_prob_knowledge_overall = np.sum(new_prob_knowledge_from_items)

        ent =  entropy(new_prob_knowledge_from_items/new_prob_knowledge_overall) 
        return self.entropy if math.isinf(ent) else ent
        
    def prob_response(self, feature, val):
        prob_of_val = data_probs[prob_to_index(val), self.items_left, feature]
        return np.sum(self.probabilities * prob_of_val)
        
    def expected_info_gain(self, feature):
        eig = 0
        for i in range(5):
            eig += self.prob_response(feature, index_to_prob(i)) * self.info_gain_ent(self.entropy_with_new_knowledge((feature, index_to_prob(i))))
        return  eig
                                                            
    def expected_info_gains(self):
        return_arr = []
        for f in self.features_left:
            return_arr.append(self.expected_info_gain(f))
        return np.nan_to_num(np.array(return_arr))                                                 
                                                            
    def get_best_feature(self):
        return self.features_left[np.argmax(self.expected_info_gains())] 
    
    def get_best_feature_and_gain(self):
        gains = self.expected_info_gains()
        best_indx = np.argmax(gains)
        return self.features_left[best_indx], gains[best_indx] 
    
    def add_knowledge(self, feature, response):
        self.knowledge.append((feature, float(response)))
        self.features_left.remove(feature)
        self.update_all()
        
    def ordered_features_name_and_gain_str(self):
        info_gains = self.expected_info_gains()
        ordered_names = [features[self.features_left[f]] for f in np.argsort(info_gains)[::-1]]
        return ":".join(ordered_names) + "," + ":".join([str(elem) for elem in list(np.sort(info_gains)[::-1])])
              
    def finish(self, choices):
        dummie = Player()
        for k in self.knowledge:
            dummie.knowledge.append(k)
            dummie.update_all()
            print '\n\n***************\nFeature:', features[k[0]], ', Your response:', k[1]
            for o in choices:
                try:
                    print 'For', o.upper(), ':', data_matrix[items.index(o.lower()), k[0]], 'prob:', dummie.probabilities[items.index(o.lower())]
                except:
                    pass
              
    def info_gain_ent(self, new_entropy):
        return self.entropy - new_entropy
    
    def info_gain_probs(self, probs):
        return self.info_gain_ent(entropy(probs))
              
    def guess_object(self, item):
        while True:
            response = raw_input("Is it a " + items[item] + "? (y/n): ")
            if response == 'y':
                print "Yay! I win!"
                self.finish([items[item]])
                return True
            
            elif response == 'n':
                print self.items_left, item
                self.items_left.remove(item)
                self.items_guessed.append(item)
                self.num_items_left -= 1
                self.prior_prob = 1.0 / self.num_items_left
                self.update_all()
                return False
                
            else:
                continue
    
    def prob_with_new_knowledge(self, new_knowledge):
        feature, response = new_knowledge
        
        prob_response_for_animal = data_probs[prob_to_index(response), :, feature] #2d slice, row is prob col is animal
        
        new_prob_knowledge_from_items = self.prob_knowledge_from_items * \
                                               np.fromiter(  (prob_response_for_animal[o]  for o in self.items_left), np.float64)
                                                
        new_prob_knowledge_overall = np.sum(new_prob_knowledge_from_items)
        
        return new_prob_knowledge_from_items/new_prob_knowledge_overall
    
    def expected_prob_win(self, feature):
        epw = 0
        for i in range(5):
            epw += self.prob_response(feature, index_to_prob(i)) * self.prob_win_if_end(self.prob_with_new_knowledge((feature, index_to_prob(i))))
        return  epw 
    
    def iterate(self):
        print self
        #best_feature, gain = self.get_best_feature_and_gain()
        gains = self.expected_info_gains()
        best_feature = self.features_left[np.argmax(self.expected_info_gains())] 
        gain = np.max(gains)
    
        fig, ax = plt.subplots()
        plt.hist(self.probabilities, np.arange(0, 21.0*np.max(self.probabilities)/20.0 + 0.001, np.max(self.probabilities)/20.0), log=True)
        plt.show()
        
        print "Best gain: ", gain
        prob_win = self.prob_win_if_end(self.probabilities)
        ''' #percentage threshold
        for o_indx, p in zip(range(1000), self.probabilities):
            probs_without = np.array(list(self.probabilities)[:o_indx] + list(self.probabilities)[o_indx+1:])
            probs_without /= np.sum(probs_without)
            eig = p*self.entropy + (1-p)*self.info_gain_probs(probs_without)
           
            if p >= 0.9:
                print "Better to choose object, info gain best question:", gain, "for object total:", eig, \
                    "if yes:", p*self.entropy, "if no:", (1-p)*self.info_gain_probs(probs_without)        
                self.question_num += 1
                if self.guess_object(o_indx): return
                else: self.iterate()
        '''
        ''' #prob win limit
        if prob_win * self.cur_money > \
            self.expected_prob_win(best_feature) * (self.cur_money - self.money_deplete):
            self.question_num += 1
            if self.guess_object(self.items_left[np.argmax(self.probabilities)]): return
            else: 
                self.update_all()
                self.iterate()
        '''
        prob_responses = [self.prob_response(best_feature, 1.0),
                          self.prob_response(best_feature, 0.5),
                          self.prob_response(best_feature, 0.0),
                          self.prob_response(best_feature, -0.5),
                          self.prob_response(best_feature, -1.0)]
        
    
        info_gains = [self.info_gain_ent(self.entropy_with_new_knowledge((best_feature, 1.0))),
                      self.info_gain_ent(self.entropy_with_new_knowledge((best_feature, 0.5))),
                      self.info_gain_ent(self.entropy_with_new_knowledge((best_feature, 0.0))),
                      self.info_gain_ent(self.entropy_with_new_knowledge((best_feature, -0.5))),
                      self.info_gain_ent(self.entropy_with_new_knowledge((best_feature, -1.0)))]
        
        
        expected_gains = list(np.array(prob_responses)*np.array(info_gains))
        
        
        helper_str = "eig = " + str(self.expected_info_gain(best_feature)) + ':\n'\
                        '   y = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[0], info_gains[0], expected_gains[0]) + '\n' \
                        '  py = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[1], info_gains[1], expected_gains[1]) + '\n' \
                        '   u = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[2], info_gains[2], expected_gains[2]) + '\n' \
                        '  pn = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[3], info_gains[3], expected_gains[3]) + '\n' \
                        '   n = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[4], info_gains[4], expected_gains[4]) + '\n'                                                                                                                                                                                         
                                                                    
        while True:
                        
            response = raw_input(helper_str + "\n" + str(self.question_num) + "). " + features[best_feature].upper() + \
                                 " (y/py/u/pn/n or end,item1,item2,item3...): ")
            if response == 'y':
                self.knowledge.append((best_feature, 1))
                
            elif response == 'n':
                self.knowledge.append((best_feature, -1))
                
            elif response == 'py':
                self.knowledge.append((best_feature, 0.5))
                
            elif response == 'pn':
                self.knowledge.append((best_feature, -0.5))
            
            elif response == 'u':
                self.knowledge.append((best_feature, 0.0))
            
            elif response.split(',')[0] == 'end':
                return self.finish(response.split(',')[1:])
                return
            
            else:
                continue
            break
        self.features_left.remove(best_feature)
        self.question_num += 1
        self.update_all()
        self.iterate()
        
    def num_nongreater_probs(self, p, pdist):
        return len(np.where(pdist <= p)[0])
    
    def num_lower_probs(self, p, pdist):
        return len(np.where(pdist < p)[0])
    
    def num_equal_probs(self, p, pdist):
        return len(np.where(pdist == p)[0])
    
    def prob_guess_item(self, item, p, pdist, numties, numless):
        tot_prob = 0
        len_dist = len(pdist)
        
        possible_ties = range(max(1, self.num_item_rand - (numless)), min(numties+1, self.num_item_rand+1))
        try:
            ways_to_get_ties = [float(ncr(numties, i) * ncr(numless, self.num_item_rand - i)) for i in possible_ties]
        except:
            print numties, i, self.num_item_rand, numless
        ways_tot = float(sum(ways_to_get_ties))
        for ways, i in zip(ways_to_get_ties, possible_ties):
            tot_prob += (1.0 / float(i)) * (ways / ways_tot)
            
        return tot_prob
        
    def prob_win_if_item(self, item, p, pdist):
        
        try: 
            numties = self.num_equal_probs(p, pdist)
            numless = self.num_lower_probs(p, pdist)
            numnongreater = self.num_nongreater_probs(p, pdist)
            return self.prob_guess_item(item, p, pdist, numties, numless) * \
                    (float(ncr(numnongreater, self.num_item_end_left)) / \
                        float(ncr(len(pdist), self.num_item_end_left)))
        except:
            return 0
        
    def prob_win_if_end(self, probdistr): #probability that a higher prob object is there
        prob_win = 0
        num = len(probdistr)
        for i, p in zip(range(1000), probdistr): #in case of tie, guess random
            #print ncr(len(np.where(self.probabilities < p)[0]), 19)
            prob_win += p * self.prob_win_if_item(i, p, probdistr)
        return prob_win
        
    def ordered_features_indx(self):
        return np.array([self.features_left[f] for f in np.argsort(self.expected_info_gains())[::-1]])
        
    def ordered_features_name(self):
        return [features[f] for f in self.ordered_features_indx()]
    
    def __str__(self):
        ordered = sorted([(items[self.items_left[i]], prob) for i, prob in zip(range(10000), self.probabilities)], key=lambda x: -x[1])
        to_print_probs = repr([(item, "{:.3}%".format(prob*100)) for item, prob in ordered][:10])
        return "\nProbabilities: " + to_print_probs + "\n" \
                "Entropy: " + str(self.entropy) + '\n' +\
                "Questions asked: " + str(self.question_num) + '\n' + \
                "Prob win: " + str(self.prob_win_if_end(self.probabilities))
        
class ComputerPlayer(Player):
    def __init__(self, item):
        super(ComputerPlayer, self).__init__()
        self.item = item
        self.item_indx = items.index(item)
        self.item_row = data_matrix[self.item_indx]
    
    def iterate(self):
        #print self
        best_feature, gain = self.get_best_feature_and_gain()
        #print "Best gain: ", gain

                
        prob_responses = [self.prob_response(best_feature, 1.0),
                          self.prob_response(best_feature, 0.5),
                          self.prob_response(best_feature, 0.0),
                          self.prob_response(best_feature, -0.5),
                          self.prob_response(best_feature, -1.0)]
        
    
        info_gains = [self.info_gain_ent(self.entropy_with_new_knowledge((best_feature, 1.0))),
                      self.info_gain_ent(self.entropy_with_new_knowledge((best_feature, 0.5))),
                      self.info_gain_ent(self.entropy_with_new_knowledge((best_feature, 0.0))),
                      self.info_gain_ent(self.entropy_with_new_knowledge((best_feature, -0.5))),
                      self.info_gain_ent(self.entropy_with_new_knowledge((best_feature, -1.0)))]
        
        
        expected_gains = list(np.array(prob_responses)*np.array(info_gains))
        
        
        helper_str = "eig = " + str(self.expected_info_gain(best_feature)) + ':\n'\
                        '   y = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[0], info_gains[0], expected_gains[0]) + '\n' \
                        '  py = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[1], info_gains[1], expected_gains[1]) + '\n' \
                        '   u = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[2], info_gains[2], expected_gains[2]) + '\n' \
                        '  pn = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[3], info_gains[3], expected_gains[3]) + '\n' \
                        '   n = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[4], info_gains[4], expected_gains[4]) + '\n'                                                                                                                                                                                         
                            
                                                
        resp = self.item_row[best_feature]         
        print '------>', self.item, features[best_feature], resp                                                                                                                           
        self.knowledge.append((best_feature, resp))
   
        
        self.features_left.remove(best_feature)
        self.question_num += 1
        self.update_all()
        self.iterate()
        
def get_for_depth_and_obj(obj, depth):
    player = ComputerPlayer(obj)
    for i in range(depth): player.iterate()
    
    k = [[k1, k2] for k1, k2 in player.knowledge]
   
    return '{},{},{}\n'.format(items.index(obj), depth, str(k))

player = Player()
print player.ordered_features_name_and_gain_str()
player.iterate()
