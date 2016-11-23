'''
Created on Feb 1, 2016

@author: alxcoh
'''
from runner_clust import *
from runner_randomN import *
from runner_nonbayesian import *
from runner_randchoice import *
import random
import scipy.stats as scistats

games = []
ngames = 100
maxdepth = 10
'''
itms = random.sample(range(1000), ngames)
for i in range(20):
    print "New Game"
    itm = itms[i]
    new_game = {}
    new_game['item'] = itm
    player = ClustPlayer(9)
    player.play_game_computer(items[itm], maxdepth)
    new_game['knowledge'] = player.knowledge
    games.append(new_game)
    

with open(base_path+'data/allgames.pickle', 'w') as allgames:
    pickle.dump(games, allgames)
'''
with open(base_path+'data/allgames.pickle', 'r') as allgames:
    games = pickle.load(allgames)

cor_random20 = [0 for d in range(maxdepth+2)]
cor_nonbayesian = [0 for d in range(maxdepth+2)]
cor_utility = [0 for d in range(maxdepth+2)]
cor_contextinsensitive = [0 for d in range(maxdepth+2)]
n_each = [0 for d in range(maxdepth+2)]
for g in games[:15]:
    print "New Game"
    optimal_bayesian = ClustPlayer(9)
    context_insensitive = ClustPlayer(9)
    context_insensitive.update_all()
    non_bayesian = NonBayesianPlayer()
    expected_utility = RandchoicePlayer(9, 20)
    random20 = RandomN(9, 20)
    for k, i in zip([None] + g['knowledge'], range(1000)):
        if k != None:
            optimal_bayesian.knowledge.append(k)
            non_bayesian.knowledge.append(k)
            expected_utility.knowledge.append(k)
            random20.knowledge.append(k)
        
        print "Knowledge:", optimal_bayesian.knowledge
        
        optimal_bayesian.update_all()
        non_bayesian.update_all()
        expected_utility.update_all()
        random20.update_all()
        
        
        normal_eig = optimal_bayesian.expected_gains()
        
        cor_random20[i] += scistats.pearsonr(normal_eig, random20.expected_gains())[0]
        cor_nonbayesian[i] += scistats.pearsonr(normal_eig, non_bayesian.expected_gains())[0]
        cor_utility[i] += scistats.pearsonr(normal_eig, expected_utility.expected_gains())[0]
        cor_contextinsensitive[i] += scistats.pearsonr(normal_eig, context_insensitive.expected_gains())[0]
        
        n_each[i] += 1
        
n_each = np.array(n_each)
print 'PEARSON VERSION'
print n_each
print cor_random20
print cor_nonbayesian
print cor_contextinsensitive
print cor_utility

print "\n\nRandom 20:\n"
print np.array(cor_random20) / n_each

print "\n\nExpected Utility:\n"
print np.array(cor_utility) / n_each

print "\n\nNon Bayesian:\n"
print np.array(cor_nonbayesian) / n_each

print "\n\Context Insensitive:\n"
print np.array(cor_contextinsensitive) / n_each


