'''
Created on Aug 3, 2015

@author: alxcoh
'''

from runner_clust import *
from runner_numpy import *
from runner_randchoice import *

def make_logfile(class1_copy1, class1_copy2, class2_copy1, class2_copy2, name1, name2):
    model1_model2 = []
    model2_model1 = []
        
    model1 = class1_copy1
    model2 = class2_copy1
        
    for i in range(20):
        choice, ordered_model1 = model1.computer_iterate()
        ordered_model2 = model2.get_normed_gains()
        model2.features_left.remove(choice)
        model2.update_all()
        model1_model2.append([ordered_model1, ordered_model2])
        
    with open('../data/' + name1 + '-' + name2 + '.pickle', 'w') as m1m2:
        pickle.dump(model1_model2, m1m2)
        
    model1 = class1_copy2
    model2 = class2_copy2
        
    for i in range(20):
        choice, ordered_model2 = model2.computer_iterate()
        ordered_model1 = model1.get_ordered_feats()
        model1.features_left.remove(choice)
        model1.update_all()
        model2_model1.append([ordered_model2, ordered_model1])
        
  
        
    with open('../data/' + name2 + '-' + name1 + '.pickle', 'w') as m2m1:
        pickle.dump(model2_model1, m2m1)
'''      
for i in range(10):
    make_logfile(OptimalPlayer(), OptimalPlayer(),
                 ClustPlayer(i), ClustPlayer(i),
                 "OptimalPlayer", ("ClustPlayer" + str(2**(i+1) if i < 9 else 999)))
'''
make_logfile(OptimalPlayer(), OptimalPlayer(),
             RandchoicePlayer(9), RandchoicePlayer(9),
             "OptimalPlayer", "RandchoicePlayer")