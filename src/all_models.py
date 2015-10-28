'''
Created on Aug 3, 2015

@author: alxcoh
'''
from runner_numpy import *
from runner_clust import *
from runner_randchoice import *
from runner_maxprob import *
import scipy.spatial.distance as ssd
import scipy.stats as stats

with open("../data/ClustPlayer2-OptimalPlayer.pickle") as mf:
    c = pickle.load(mf)
    
for i in range(20):
    print ssd.euclidean(c[i][0], c[i][1])

def make_logfile(class1_copy1, class1_copy2, class2_copy1, class2_copy2, name1, name2):
    global item
    model1_model2_specific = []
    model2_model1_specific = []
    
    model1_model2 = []
    model2_model1 = []
    
    model1_model2_probs = []
    model2_model1_probs = []
    
    model1_model2_unnormalized = []
    model2_model1_unnormalized = []
    
    model1 = class1_copy1
    model2 = class2_copy1
    print '\n\n******************\n', name1, name2, '\n\n'
    for i in range(20):
        
        model1_model2_probs.append([model1.probabilities[:].copy(), model2.probabilities[:].copy()])
        
        choice, ordered_model1 = model1.computer_iterate(item)
        ordered_model2 = model2.get_unnormed_gains()
        
        model1_model2_unnormalized.append([ordered_model1[:].copy(), ordered_model2[:].copy()])
        print np.sum(ordered_model1), np.sum(ordered_model2)
        ordered_model1 /= np.sum(ordered_model1)
        ordered_model2 /= np.sum(ordered_model2)
        
        #print ordered_model1, np.sum(ordered_model1), '\n', ordered_model2, np.sum(ordered_model2)
        #print features[choice], features[model1.features_left[np.argmax(ordered_model1)]]
        #print features[model1.features_left[np.argmax(ordered_model2)]]
        #print features[choice], features[model2.features_left[np.argmax(ordered_model2)]]
        model2.features_left.remove(choice)
        model2.knowledge = model1.knowledge
        #print model2.probabilities
        model2.update_all()
        
        
        
        #print model2.probabilities
        #print ordered_model1[0], ordered_model2[0]
        
        m1choices = (np.argsort(ordered_model1)[::-1])[2::len(ordered_model1)/8]
        m1ratings = ordered_model1[m1choices]
        m2ratings = ordered_model2[m1choices]
        

        m1ratings_normalized = list(m1ratings / np.sum(m1ratings))
        m2ratings_normalized = list(m2ratings / np.sum(m2ratings))

        m1choices = list(m1choices)
        m1ratings = list(m1ratings)
        m2ratings = list(m2ratings)
        
        #print len(ordered_model1), len(model1.features_left)
        print name1 + " --> " + name2
        print str(model1.question_num-1) + '). ' + features[choice], data_matrix[items.index(item), choice], '\n'
        for c, m1, m2, m1n, m2n in zip(m1choices, m1ratings, m2ratings, m1ratings_normalized, m2ratings_normalized):
            print features[model1.features_left[c]], ' '*(50-len(features[model1.features_left[c]])), 'Raw: {:.5f}, {:.5f}, Normalized: {:.5f}, {:.5f}'.format(m1, m2, m1n, m2n)
            
        print "Spearman:", stats.spearmanr(ordered_model1, ordered_model2)[0]
            
        model1.features_left.remove(choice)
            
        model1_model2.append([ordered_model1, ordered_model2])
        model1_model2_specific.append([m1choices, m1ratings, m2ratings])
        
        
        
    with open('../data/' + item + '/' + name1 + '-' + name2 + '_specific.pickle', 'w') as m1m2:
        pickle.dump(model1_model2_specific, m1m2)
        
    with open('../data/' + item + '/' + name1 + '-' + name2 + '.pickle', 'w') as m1m2:
        pickle.dump(model1_model2, m1m2)
        
    with open('../data/' + item + '/' + name1 + '-' + name2 + '_probs.pickle', 'w') as m1m2:
        pickle.dump(model1_model2_probs, m1m2)
        
    with open('../data/' + item + '/'+ name1 + '-' + name2 + '_unnormalized.pickle', 'w') as m1m2:
        pickle.dump(model1_model2_unnormalized, m1m2)
        
    model1 = class1_copy2
    model2 = class2_copy2
    print '\n\n******************\n', name2, name1, '\n\n'
    for i in range(20):
        
        model2_model1_probs.append([model2.probabilities[:].copy(), model1.probabilities[:].copy()])
        
        choice, ordered_model2 = model2.computer_iterate(item)
        ordered_model1 = model1.get_unnormed_gains()

        model2_model1_unnormalized.append([ordered_model2[:].copy(), ordered_model1[:].copy()])
        
        ordered_model1 /= np.sum(ordered_model1)
        ordered_model2 /= np.sum(ordered_model2)

        model1.features_left.remove(choice)
        model1.knowledge = model2.knowledge
        model1.update_all()
        
        
        
        m2choices = (np.argsort(ordered_model2)[::-1])[2::len(ordered_model2)/8]
        m2ratings = ordered_model2[m2choices]
        m1ratings = ordered_model1[m2choices]
        
        m1ratings_normalized = list(m1ratings / np.sum(m1ratings))
        m2ratings_normalized = list(m2ratings / np.sum(m2ratings))
        
        m2choices = list(m2choices)
        m2ratings = list(m2ratings)
        m1ratings = list(m1ratings)
        
        print name2 + " --> " + name1
        print str(model2.question_num-1) + '). ' + features[choice], data_matrix[items.index(item), choice], '\n'
        #print m2choices, len(ordered_model2), len(model2.features_left), len(model1.features_left)
        for c, m2, m1, m2n, m1n in zip(m2choices, m2ratings, m1ratings, m2ratings_normalized, m1ratings_normalized):
            print features[model2.features_left[c]], ' '*(50-len(features[model2.features_left[c]])), 'Raw: {:.5f}, {:.5f}, Normalized: {:.5f}, {:.5f}'.format(m2, m1, m2n, m1n)
        
        
        print "Spearman:", stats.spearmanr(ordered_model1, ordered_model2)[0]
        
        
        model2.features_left.remove(choice)
        
        model2_model1.append([ordered_model2, ordered_model1])
        model2_model1_specific.append([m2choices, m2ratings, m1ratings])
  
        
    with open('../data/' + item + '/' + name2 + '-' + name1 + '_specific.pickle', 'w') as m2m1:
        pickle.dump(model2_model1_specific, m2m1)
        
    with open('../data/' + item + '/' + name2 + '-' + name1 + '.pickle', 'w') as m2m1:
        pickle.dump(model2_model1, m2m1)
        
    with open('../data/' + item + '/' + name2 + '-' + name1 + '_probs.pickle', 'w') as m2m1:
        pickle.dump(model2_model1_probs, m2m1)
        
    with open('../data/' + item + '/' + name2 + '-' + name1 + '_unnormalized.pickle', 'w') as m2m1:
        pickle.dump(model2_model1_unnormalized, m2m1)

item = 'desk'
print "\n\n=================\n=================\nDESK:\n\n\n"

'''
for i in range(0, 10):
    make_logfile(ClustPlayer(9), ClustPlayer(9),
                 ClustPlayer(i), ClustPlayer(i),
                 "FullBayesian", ("ClustPlayer" + str(2**(i+1) if i < 9 else 1000)))

make_logfile(ClustPlayer(9), ClustPlayer(9),
             RandchoicePlayer(9), RandchoicePlayer(9),
             "FullBayesian", "BonusTask")
'''
make_logfile(ClustPlayer(9), ClustPlayer(9),
             MaxprobPlayer(9), MaxprobPlayer(9),
             "FullBayesian", "Maxprob")


item = 'dog'
print "\n\n=================\n=================\nDOG:\n\n\n"
'''
for i in range(0, 10):
    make_logfile(ClustPlayer(9), ClustPlayer(9),
                 ClustPlayer(i), ClustPlayer(i),
                 "FullBayesian", ("ClustPlayer" + str(2**(i+1) if i < 9 else 1000)))

make_logfile(ClustPlayer(9), ClustPlayer(9),
             RandchoicePlayer(9), RandchoicePlayer(9),
             "FullBayesian", "BonusTask")
'''
make_logfile(ClustPlayer(9), ClustPlayer(9),
             MaxprobPlayer(9), MaxprobPlayer(9),
             "FullBayesian", "Maxprob")