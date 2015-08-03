'''
Created on Mar 31, 2015

@author: alxcoh
'''
import time
import math
import numpy as np
import sys
import scipy.spatial.distance as ssd
import data_loader
import scipy.stats
#import sklearn
#import cv2

feature_blacklist = ["DOES IT HAVE A FRONT AND A BACK?", "DOES IT CAST A SHADOW?", "DOES IT HAVE AT LEAST ONE HOLE?"]


data_logger = open("../logs/data_logger.txt", "w")
prob_logger = open("../logs/prob_logger.txt", "w")

data_logger.truncate(0)
prob_logger.truncate(0)



# full_probability_dict = {1.0 : {1.0 : 0.60, 0.5 : 0.25, 0.0 : 0.09, -0.5 : 0.04, -1.0 : 0.02 },
#                          0.5 : {1.0 : 0.37, 0.5 : 0.38, 0.0 : 0.14, -0.5 : 0.07, -1.0 : 0.04 },
#                          0.0 : {1.0 : 0.10, 0.5 : 0.20, 0.0 : 0.40, -0.5 : 0.20, -1.0 : 0.10 },
#                         -0.5 : {1.0 : 0.04, 0.5 : 0.07, 0.0 : 0.14, -0.5 : 0.38, -1.0 : 0.37 },
#                         -1.0 : {1.0 : 0.02, 0.5 : 0.04, 0.0 : 0.09, -0.5 : 0.25, -1.0 : 0.60 }}

# full_probability_dict = {1.0 : {1.0 : 1.0, -1.0 : 0.0 },
#                         -1.0 : {1.0 : 0.0, -1.0 : 1.0 }}



full_probability_dict = {1.0 : {1.0 : 0.60, 0.5 : 0.25, 0.0 : 0.09, -0.5 : 0.04, -1.0 : 0.02 },
                         0.5 : {1.0 : 0.25, 0.5 : 0.38, 0.0 : 0.22, -0.5 : 0.10, -1.0 : 0.05 },
                         0.0 : {1.0 : 0.09, 0.5 : 0.22, 0.0 : 0.38, -0.5 : 0.22, -1.0 : 0.09 },
                        -0.5 : {1.0 : 0.05, 0.5 : 0.10, 0.0 : 0.22, -0.5 : 0.38, -1.0 : 0.25 },
                        -1.0 : {1.0 : 0.02, 0.5 : 0.04, 0.0 : 0.09, -0.5 : 0.25, -1.0 : 0.60 }}

# full_probability_dict =  {-1.0: {-1.0: 0.86536632, -0.5: 0.04755257, 0.0: 0.05165025, 0.5: 0.01979656, 1.0: 0.01563430},
#                           -0.5: {-1.0: 0.4107812,  -0.5: 0.19152609, 0.0: 0.18489911, 0.5: 0.11932672, 1.0: 0.09346688},
#                           -0.0: {-1.0: 0.25688982, -0.5: 0.13691758, 0.0: 0.32072590, 0.5: 0.17985315, 1.0: 0.10561355},
#                            0.5: {-1.0: 0.15429598, -0.5: 0.08676840, 0.0: 0.20624163, 0.5: 0.24463726, 1.0: 0.30805673},
#                            1.0: {-1.0: 0.04290900, -0.5: 0.01680109, 0.0: 0.09186447, 0.5: 0.17145633, 1.0: 0.67696911}}

new_dict = {}


prior_probs = {-1.0: 0.539,
               -0.5: 0.110,
                0.0: 0.161,
                0.5: 0.070,
                1.0: 0.120}

for pA in [-1, -0.5, 0.0, 0.5, 1.0]:
    new_dict[pA] = {}
    sum_row = 0
    for pB in [-1, -0.5, 0.0, 0.5, 1.0]:
        print "prob", pA, "such that", pB, "=",full_probability_dict[pA][pB], '=?=', full_probability_dict[pB][pA]*prior_probs[pA]/prior_probs[pB]        
        sum_row += full_probability_dict[pB][pA]*prior_probs[pA]/prior_probs[pB]
    for pB in [-1, -0.5, 0.0, 0.5, 1.0]:
        new_dict[pA][pB] = full_probability_dict[pB][pA]*prior_probs[pA]/(prior_probs[pB]*sum_row)

for k, v in sorted(new_dict.items(), key=lambda x: x[0]):
    print '\n',str(k) + ": ",
    for k, v in sorted(v.items(), key=lambda x: x[0]):
        print str(k) + ':' + str(v),
'''
                            -1.0: 53.9%
                            -0.5: 11.0%
                            0.0 : 16.1%
                            0.5 :  7.0%
                            1.0 : 12.0%
'''
class Player:
    def __init__(self):     
        
        _, self.data_dict, all_features, all_objects = data_loader.get_data()

        self.unexamined_features = all_features[:]
        for f in feature_blacklist: self.unexamined_features.remove(f)
        self.known_features = {}
        self.probabilities = {}
        self.probability_features_all_objects = {}
        self.questions_asked = 0
        self.unexamined_objects = all_objects[:]
        self.ordered_probs = []
        
        for o in self.unexamined_objects:
            self.probability_features_all_objects[o] = 1.0
            self.probabilities[o] = self.prior_probability(o)
            
        self.cur_entropy = self.entropy(self.probabilities)


    def prior_probability(self, the_object):
        return 1.0/float(len(self.unexamined_objects))
    
        
    def probability_response_generated(self, response, object_data, feature):
        #if abs(response) != 1 or abs(object_data) != 1: return 1
        return full_probability_dict[response][object_data]
    
        
    def probability_features_from_object(self, feature_set, the_object):
        total_product = 1
        for feature in feature_set:
            feature_value = feature_set[feature]
            object_value = self.data_dict[the_object][feature]
            #print probability_response_generated(feature_value, object_value, feature), feature_value, object_value, feature
            total_product *= self.probability_response_generated(feature_value, object_value, feature)
            
        #print total_product*prior_probability(the_object), the_object
            
        return total_product*self.prior_probability(the_object)
    
    def probability_features_from_all_objects(self, feature_set):
        return_dict = {}
        for o in self.unexamined_objects:
            return_dict[o] = self.probability_features_from_object(feature_set, o) / self.prior_probability(o)
        
        return return_dict
    
    def probability_feature_set(self, probability_features_all_objects):

        return sum(probability_features_all_objects.itervalues())
    
    def probability_object(self, the_object, probability_features_all_objects, probability_of_feature_set = None):
        try:
            numerator = probability_features_all_objects[the_object]
            denominator = self.probability_feature_set(probability_features_all_objects) if \
                            probability_of_feature_set == None else probability_of_feature_set
            return numerator/denominator
        except:
            return 0
        
    def probability_feature_overall(self, value, feature, object_probabilities):
        sum_total = 0
        for o, p in object_probabilities.iteritems():
            sum_total += p*self.probability_response_generated(value, self.data_dict[o][feature], feature)
        return sum_total
        
    def entropy(self, probability_distribution):
#         sum_total = 0
#         for key, value in probability_distribution.iteritems():
#             sum_total += 0 if value == 0 else value*math.log(value, 2)
#         
#         return -sum_total
        return scipy.stats.entropy(np.array(probability_distribution.values()))
        
    
        
    def information_gain_value(self, feature, feature_value):
        new_probs = self.new_probabilities_prediction(feature, feature_value)
        info_gain = abs(self.cur_entropy - self.entropy(new_probs))
        return info_gain
    
    def information_gain_total(self, feature):
        self.cur_entropy = self.entropy(self.probabilities)

        probs = {1.0 : self.probability_feature_overall( 1.0, feature, self.probabilities),
                 0.5 : self.probability_feature_overall( 0.5, feature, self.probabilities),
                 0.0 : self.probability_feature_overall( 0.0, feature, self.probabilities),
                -0.5 : self.probability_feature_overall(-0.5, feature, self.probabilities),
                -1.0 : self.probability_feature_overall(-1.0, feature, self.probabilities)}
        
        expected_gain = 0
        for val, prob in probs.iteritems():
            expected_gain += prob*self.information_gain_value(feature, val)
                        
        data_logger.write("\n" + feature + " expected info gain: " + str(expected_gain))

        return expected_gain
                
    def best_feature(self):
        max_info_gain = 0
        best_feature = self.unexamined_features[0]
        for f, i in zip(self.unexamined_features, range(1000)):
            t = time.time()
            info_gain = self.information_gain_total(f)
            #print time.time() - t
            #print i, f, info_gain
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                best_feature = f
            

            
        return best_feature
    
    def new_probabilities_prediction(self, new_feature, new_value):
        t = time.time()
        new_probability_features_all_objects = {}
        for key in self.probability_features_all_objects:
            new_probability_features_all_objects[key] = \
                self.probability_features_all_objects[key]* \
                self.probability_response_generated(new_value, self.data_dict[key][new_feature], new_feature)
                
        #print time.time() - t,
        t = time.time()
                
        probabilities = {}
        prob_feature_set = self.probability_feature_set(new_probability_features_all_objects)
        for o in self.unexamined_objects:
            probabilities[o] = self.probability_object(o, new_probability_features_all_objects, prob_feature_set)
            
        #print time.time() - t,
        t = time.time()
            
        #data_logger.write("\nNew probabilities with " + str(new_feature) + " " + str(new_value) + " " + str(probabilities))
        
        #print time.time() - t
        
        return probabilities
    
    def add_feature(self, new_feature, new_value):
        for o in self.unexamined_objects:
                #if o == "bear": print self.probability_features_all_objects[o], probability_response_generated(new_value, data_dict[o][new_feature], new_feature)
                self.probability_features_all_objects[o] *= \
                    self.probability_response_generated(new_value, self.data_dict[o][new_feature], new_feature)
                
        self.update_probabilities()
    
    def update_probabilities(self):
        self.probabilities = {}
        for o in self.unexamined_objects:
            self.probabilities[o] = self.probability_object(o, self.probability_features_all_objects)
            
        self.cur_entropy = self.entropy(self.probabilities)
    
    def get_response(self, feature):
        while True:
            the_input = raw_input(feature + " (y/py/u/pn/n): ")
        
            if the_input == 'y': return 1.0
            elif the_input == 'n': return -1.0
            if the_input == 'py': return 0.5
            elif the_input == 'pn': return -0.5
            elif the_input == 'u': return 0 
            
            print "Issue with input"  
        
        
    def ask_question(self, feature):
        the_input = self.get_response(feature)
        
        self.unexamined_features.remove(feature)
        self.known_features[feature] = the_input
        
        self.add_feature(feature, the_input)
        prob_logger.write("\n" + str(self.questions_asked + 1) + " " + str(feature) + " " + str(the_input))
        
    def guess(self, choice):   
        answer = raw_input("I am " + str( int(self.probabilities[choice]*100) ) + "% sure it is a(n) " + choice + " am I right? (y/n): ") 
        if answer == 'y':
            print "Here are all the probabilities:", self.ordered_probs 
            return True
        else:
            self.unexamined_objects.remove(choice)
            self.probability_features_all_objects = self.probability_features_from_all_objects(self.known_features)
            self.update_probabilities()
            return False
    
    def play_game(self):
        while True:
            if len(self.unexamined_features) == 1: 
                print "found nothing with high enough confidence"
                return None
            
            best_prob = 0
            best_object = self.unexamined_objects[0]
            questions_left = 20 - self.questions_asked
            probs_tuple = []
            probs = []
            
            for o in self.unexamined_objects:
                prob = self.probabilities[o]
                probs_tuple.append((prob, o))
                probs.append(prob)
                if prob > best_prob:
                    best_prob = prob
                    best_object = o
                    
                    
                
            probs_tuple = sorted(probs_tuple, key = lambda x: x[0], reverse = True)
            self.ordered_probs = probs_tuple
            prob_logger.write("\n" + str(self.ordered_probs))
            #print self.ordered_probs
            probs = sorted(probs, reverse = True)
            #print probs_tuple
            #print sum(probs[:questions_left])
            print "Sum probs:", str(sum(probs)) + ", Sum top probs:", \
                    ("N/A" if self.questions_asked > 19 else str(sum(probs[:questions_left]))) + \
                    ", Best object:", best_object + ", with prob:", str(best_prob) + ", total entropy:", self.cur_entropy
            if best_prob > 0.9:
                if self.guess(best_object): return best_object
            
            elif self.questions_asked < 19:
                can_win = sum(probs[:questions_left]) > 0.9
                if can_win:
                    if self.guess(best_object): return best_object

            elif self.questions_asked == 19:
                if self.guess(best_object): return best_object
                
#             


#             print "\n\nProbabilities: ", self.probabilities
#             print "Entropy: ", self.cur_entropy
            print "Questions asked:", self.questions_asked, "\n\n"
            
            data_logger.write("\nQuestions asked: " + str(self.questions_asked) + 
                            "\nProbabilities: " + str(self.ordered_probs) + "\n\n")
            
            self.ask_question(self.best_feature())
            
            self.questions_asked += 1
     
class ComputerPlayer(Player):
    def __init__(self, the_object):
        Player.__init__(self)
        self.object = the_object
        
    def get_response(self, feature):
        #print feature, self.object
        the_input = self.data_dict[self.object][feature]
        
        print feature + " "
        
        #time.sleep(0.5)
        
        if the_input == 1.0: print 'y'
        elif the_input == 0.5: print 'py'
        elif the_input == 0.0: print 'u'
        elif the_input == -0.5: print 'pn'
        else: print 'n'
    
        return the_input

my_player = Player()
my_player.play_game()

data_logger.close()
