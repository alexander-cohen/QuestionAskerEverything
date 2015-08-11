'''
Created on Jul 17, 2015

@author: alxcoh
'''

import time
from multiprocessing import Pool
from runner_numpy import *
from runner_clust import ClustPlayer



class RandchoicePlayer(ClustPlayer): #ALL ITEMS AND FEATURES WILL BE REFERRED TO BY INDEX
    def __init__(self, clusts):
        super(RandchoicePlayer, self).__init__(clusts)
        self.cur_money = 2.0
        self.num_item_rand = 20
        self.num_item_end_left = self.num_item_rand-1
        
        self.update_all() #updates all information

    def num_nongreater_probs(self, p, pdist):
        return len(np.where(pdist <= p)[0])
    
    def num_lower_probs(self, p, pdist):
        return len(np.where(pdist < p)[0])
    
    def num_equal_probs(self, p, pdist):
        return len(np.where(pdist == p)[0])
    
    def prob_guess_item(self, p, numties, numless, numtot):
        tot_prob = 0
        
        possible_ties = range(max(1, self.num_item_rand - (numless)), min(numties+1, self.num_item_rand+1))
        try:
            ways_to_get_ties = [float(ncr(numties, i) * ncr(numless, self.num_item_rand - i)) for i in possible_ties]
        except:
            pass
            #print numties, i, self.num_aitem_rand, numless
        ways_tot = float(sum(ways_to_get_ties))
        for ways, i in zip(ways_to_get_ties, possible_ties):
            tot_prob += (1.0 / float(i)) * (ways / ways_tot)
            
        return tot_prob
        
    def prob_win_if_item(self, p, numties, numless, numgreater, numtot):
        
        try: 
            numnongreater = numties + numless
            return self.prob_guess_item(p, numties, numless, numtot) * \
                    (float(ncr(numnongreater, self.num_item_end_left)) / \
                        float(ncr(numtot, self.num_item_end_left)))
        except:
            return 0
    
    def prob_win_if_end(self, probdistr): #probability that a higher prob object is there
        t = time.time()
        prob_win = 0
        num_tot = len(probdistr)
        list_version = list(probdistr)
        equiv = []
        num_lower = 0
        for val in sorted(list(set(list_version))):
            num = list_version.count(val)
            if num*val < 0.00000001: continue
            equiv.append((val*num, num, num_lower, len(items) - num_lower - num, val))
            num_lower += num
            
        #print equiv
            
        for p, ties, lower, greater, _ in equiv: #in case of tie, guess random
            #print ncr(len(np.where(self.probabilities < p)[0]), 19)
            prob_win += p * self.prob_win_if_item(p, ties, lower, greater, num_tot)
        #self.calcs += 1
        #print self.calcs, '). Calc took', time.time() - t, 'seconds'
        return prob_win
        
 
    def expected_gain(self, feature):
        return expected_gain((feature, self))

    
    def expected_gains(self):
        t = time.time()
        self.calcs = 0
        p = Pool()

        return_arr = p.map(expected_gain, zip(self.features_left, [self]*1000))
        p.close()
        print time.time() - t
        return np.nan_to_num(np.array(return_arr))                                                 
                                                                           
    def helper_str(self, best_feature):
        
        prob_responses = [self.prob_response(best_feature, 1.0),
                          self.prob_response(best_feature, 0.5),
                          self.prob_response(best_feature, 0.0),
                          self.prob_response(best_feature, -0.5),
                          self.prob_response(best_feature, -1.0)]

       
        info_gains = [self.prob_win_if_end(self.prob_with_new_knowledge((best_feature, 1.0))),
                      self.prob_win_if_end(self.prob_with_new_knowledge((best_feature, 0.5))),
                      self.prob_win_if_end(self.prob_with_new_knowledge((best_feature, 0.0))),
                      self.prob_win_if_end(self.prob_with_new_knowledge((best_feature, -0.5))),
                      self.prob_win_if_end(self.prob_with_new_knowledge((best_feature, -1.0)))]
        
       
        expected_gains = list(np.array(prob_responses)*np.array(info_gains))
        

        helper_str = "eig = " + str(self.expected_gain(best_feature)) + ':\n'\
                        '   y = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[0], info_gains[0], expected_gains[0]) + '\n' \
                        '  py = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[1], info_gains[1], expected_gains[1]) + '\n' \
                        '   u = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[2], info_gains[2], expected_gains[2]) + '\n' \
                        '  pn = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[3], info_gains[3], expected_gains[3]) + '\n' \
                        '   n = {:.4f} * {:.4f} = {:.4f}'.format(prob_responses[4], info_gains[4], expected_gains[4]) + '\n'                                                                                                                                                                                         

        return helper_str
              

    def __str__(self):
        ordered = sorted([(items[self.items_left[i]], prob) for i, prob in zip(range(10000), self.probabilities)], key=lambda x: -x[1])
        to_print_probs = repr([(item, "{:.3}%".format(prob*100)) for item, prob in ordered][:10])
        if print_for_test:
            return ""
            return "\n\nQuestions asked: " + str(self.question_num)
        else:
            return "\nProbabilities: " + to_print_probs + "\n" + \
                    "Entropy: " + str(self.entropy) + '\n' +\
                    "Prob win: " + str(self.prob_win_if_end(self.probabilities)) + '\n' + \
                    "Questions asked: " + str(self.question_num)


def expected_gain(param):
    feature, player = param
    eig = 0
    for i in range(5):
        eig += player.prob_response(feature, index_to_prob(i)) * player.prob_win_if_end(player.prob_with_new_knowledge((feature, index_to_prob(i))))
        #eig += self.prob_response(feature, index_to_prob(i)) * np.max(self.prob_with_new_knowledge((feature, index_to_prob(i))))
    #print 'EIG for', features[feature], '=', eig
    return  eig