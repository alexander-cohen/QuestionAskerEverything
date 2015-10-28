'''
Created on Jul 4, 2015

@author: alxcoh
'''
import numpy as np
from scipy.stats import entropy
import cPickle as pickle
import math

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

print_for_test = True

def prob_to_index(prob):
    return (prob+1)*2

def index_to_prob(index):
    return (float(index)/2) - 1


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
    data_probs_temp = pickle.load(d_probs)


num_items_tot = len(items)

feature_blacklist = []


print "beginning full file"

class OptimalPlayer(object): #ALL itemS AND FEATURES WILL BE REFERRED TO BY INDEX
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
        
        self.data_probs = data_probs_temp
        
        self.entropy = entropy(self.probabilities) #entropy of distribution
        self.cur_money = 2.0
        self.money_deplete = 0.5
        self.num_item_rand = 20
        self.num_item_end_left = self.num_item_rand-1
        
        self.update_all() #updates all information
        
    def prob_knowledge_from_item(self, item):
        item_probs = self.data_probs[:, item] #a 2D slice of the array, rows being prob and column being feature
        return np.prod( np.fromiter((item_probs[prob_to_index(r)][f] for f, r in self.knowledge), np.float64)) * \
                self.prior_prob
        
    def get_prob_knowledge_from_items(self):
        return np.fromiter( ((self.prob_knowledge_from_item(item)) for item in self.items_left ), np.float64)

    def update_all(self):
        self.num_items_left = len(self.items_left)
        self.prior_prob = 1.0 / self.num_items_left
        
        self.prob_knowledge_from_items = self.get_prob_knowledge_from_items()
        self.prob_knowledge_overall = np.sum(self.prob_knowledge_from_items)
        
    
        if len(self.items_guessed) > 0: self.prob_knowledge_from_items[np.array(self.items_guessed)] = 0
        
        
        
        self.probabilities = self.prob_knowledge_from_items / self.prob_knowledge_overall
        
        self.entropy = entropy(self.probabilities)
        
            
    def prob_with_new_knowledge(self, new_knowledge):
        feature, response = new_knowledge
        
        prob_response_for_animal = self.data_probs[prob_to_index(response), :, feature] #2d slice, row is prob col is animal
        
        new_prob_knowledge_from_items = self.prob_knowledge_from_items * \
                                               np.fromiter(  (prob_response_for_animal[o]  for o in self.items_left), np.float64)
                                                
        new_prob_knowledge_overall = np.sum(new_prob_knowledge_from_items)
        
        return new_prob_knowledge_from_items/new_prob_knowledge_overall
        
    def entropy_with_new_knowledge(self, new_knowledge):
        ent =  entropy(self.prob_with_new_knowledge(new_knowledge)) 
        return self.entropy if math.isinf(ent) else ent
        
    def prob_response(self, feature, val):
        prob_of_val = self.data_probs[prob_to_index(val), self.items_left, feature]
        return np.sum(self.probabilities * prob_of_val)
        
    def expected_gain(self, feature):
        eig = 0
        for i in range(5):
            eig += self.prob_response(feature, index_to_prob(i)) * self.info_gain_ent(self.entropy_with_new_knowledge((feature, index_to_prob(i))))
        return  eig
                                                            
    def expected_gains(self):
        return_arr = []
        for f in self.features_left:
            return_arr.append(self.expected_gain(f))
        return np.nan_to_num(np.array(return_arr))                                                 
                                                            
    def get_best_feature(self):
        return self.features_left[np.argmax(self.expected_gains())] 
    
    def get_best_feature_and_gain(self):
        gains = self.expected_gains()
        best_indx = np.argmax(gains)
        return self.features_left[best_indx], gains[best_indx] 
    
    def add_knowledge(self, feature, response):
        self.knowledge.append((feature, float(response)))
        self.features_left.remove(feature)
        self.update_all()
        
    def ordered_features_name_and_gain_str(self):
        info_gains = self.expected_gains()
        ordered_names = [features[self.features_left[f]] for f in np.argsort(info_gains)[::-1]]
        return ":".join(ordered_names) + "," + ":".join([str(elem) for elem in list(np.sort(info_gains)[::-1])])
              
    def finish(self, choices):
        for k in self.knowledge:
            print '\n\n***************\nFeature:', features[k[0]], ', Your response:', k[1]
            for o in choices:
                try:
                    print 'For', o.upper(), ':', data_matrix[items.index(o.lower()), k[0]]
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

    
    def helper_str(self, best_feature):
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
        
        
        helper_str = "eig = " + str(self.expected_gain(best_feature)) + ':\n'\
                        '   y = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[0], info_gains[0], expected_gains[0]) + '\n' \
                        '  py = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[1], info_gains[1], expected_gains[1]) + '\n' \
                        '   u = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[2], info_gains[2], expected_gains[2]) + '\n' \
                        '  pn = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[3], info_gains[3], expected_gains[3]) + '\n' \
                        '   n = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[4], info_gains[4], expected_gains[4]) + '\n'   
        
        return helper_str
        
    def query_person_indx(self, feat, itm = None):
        while True:
            resp = raw_input(str(self.question_num) + "). " + features[feat] + ' (y/py/u/pn/n/end): ')
            if resp == 'y': return 1.0
            elif resp == 'py': return 0.5
            elif resp == 'u': return 0.0
            elif resp == 'pn': return -0.5
            elif resp == 'n': return -1.0
            elif resp.split(',')[0] == 'end': 
                self.finish(resp.split(',')[1:])
                continue
            
    def query_dat_indx(self, feat, itm = None):
        resp = data_matrix[itm, feat]
        resp_char = {-1.0:'n', -0.5:'pn', 0.0:'u', 0.5:'py', 1.0:'y'}[resp]
        if not print_for_test:
            print str(self.question_num) + "). " + features[feat] + ' (y/py/u/pn/n/end): ' + resp_char
        return resp
    
    def query_dat_name(self, feat, itm = None):
        return self.query_dat_indx(feat, items.index(itm))
        
    def guess_threshes(self):
        ''' 
        #percentage threshold
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
        ''' 
        #prob win limit
        if prob_win * self.cur_money > \
            self.expected_prob_win(best_feature) * (self.cur_money - self.money_deplete):
            self.question_num += 1
            if self.guess_object(self.items_left[np.argmax(self.probabilities)]): return
            else: 
                self.update_all()
                self.iterate()
        '''
        
    def iterate(self, query_func = None, itm = None):
        if query_func == None: query_func = self.query_person_indx
        if not print_for_test:
            print self
        #best_feature, gain = self.get_best_feature_and_gain()
        gains = self.expected_gains()
              
        best_feature = self.features_left[np.argmax(gains)] 
  
        gain = np.max(gains)
        if not print_for_test:
            print "Best gain: ", gain
            print self.helper_str(best_feature)
            
        resp = query_func(best_feature, itm)
        self.knowledge.append((best_feature, resp))    
        
        if not print_for_test:
            self.features_left.remove(best_feature)
            
        self.question_num += 1
        self.update_all()
        return best_feature, gains
        
    def play_game(self):
        while True:
            self.iterate()
    
    def play_game_computer(self, itm = 'desk', depth = 20):
        for i in range(depth):
            self.iterate(self.query_dat_name, itm)
            
    def computer_iterate(self, itm = 'desk'):
        return self.iterate(self.query_dat_name, itm)
        
    def get_ordered_feats(self):
        return [self.features_left[f] for f in reversed(np.argsort(self.expected_gains()))] 
    
    def get_normed_gains(self):
        gains = self.expected_gains()
        return gains / np.sum(gains)
    
    def get_unnormed_gains(self):
        gains = self.expected_gains()
        return gains
        
    def ordered_features_indx(self):
        return np.array([self.features_left[f] for f in np.argsort(self.expected_gains())[::-1]])
        
    def ordered_features_name(self):
        return [features[f] for f in self.ordered_features_indx()]
    
    def __str__(self):
        ordered = sorted([(items[self.items_left[i]], prob) for i, prob in zip(range(10000), self.probabilities)], key=lambda x: -x[1])
        to_print_probs = repr([(item, "{:.3}%".format(prob*100)) for item, prob in ordered][:10])
        if print_for_test:
            return ""
            #return "\n\nQuestions asked: " + str(self.question_num)
        else:
            return "\nProbabilities: " + to_print_probs + "\n" \
                    "Entropy: " + str(self.entropy) + '\n' +\
                    "Questions asked: " + str(self.question_num)
'''
player = OptimalPlayer()
print player.ordered_features_name_and_gain_str()
player.iterate()
'''