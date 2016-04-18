'''
Created on Oct 22, 2015

@author: alxcoh
'''

import numpy as np
import json
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import cPickle as pickle
from runner_clust import *
from get_serverdata import *
import scipy.stats as scistats
from get_serverdata import *


fsize = 70
                  
font = {'family' : 'normal',
                'size'   : fsize}
    
matplotlib.rc('font', **font)

print "here"
people = None 
#752, 674
with open("../experiment_data/people_clust_bayesian.pickle", 'r') as peoplefile:
    people = pickle.load(peoplefile)
    
with open("../experiment_data/times_seen.pickle", 'r') as times_seen_file:
    times_seen = pickle.load(times_seen_file)    
    
print "there"
trials = [[] for i in range(10)]
for p in people:
    for t, i in zip(p.analyzed_oneshots.trials, range(100)):
        trials[i].append(t)

def nameof(index):
    if index == 0:
        return "Optimal Bayesian"
    if index == 7:
        return "Context Insensitive".ljust(23)
    elif index < 8:
        return ("Variational " + str(min(2**(index+1), 1000)) + ":").ljust(23)
    else:
        return ("Expected Utility " +  str(min(2**(index-8), 1000)) + ":").ljust(23)

def nameofNojust(index):
    if index == 0:
        return "Optimal Bayesian"
    if index == 7:
        return "Context Insensitive".ljust(23)
    elif index < 8:
        return "Variational " + str(min(2**(index+1), 1000)) + ":"
    else:
        return "Expected Utility " +  str(min(2**(index-8), 1000)) + ":"
    
def nameofNojustShort(index):
    if index == 0:
        return "Optimal Bayesian"
    if index == 7:
        return "Context Insensitive".ljust(23)
    elif index < 8:
        return "Variational " + str(min(2**(index+1), 1000)) + ":"
    else:
        return "Utility " +  str(min(2**(index-8), 1000)) + ":"

trials = sorted(trials, key = lambda x: x[0].depth)
sumtot= 0
numtot = 0

sumtotChance = 0
numtotChance = 0
expectedandbonus = []
testing_times_seen = False
if not testing_times_seen:
    for trial, trial_index in zip(trials, range(100)):
        
        itm = filter(lambda x: x[0] == 'oneshot_itm_chosen', trial[0].data)[0][1][2]
        
        
        matrx = np.array([t.rankorder for t in trial])
        if trial_index == 0: print matrx
        #matrx[matrx >= 2] = 2
        person = [5-x for x in np.average(matrx, axis=0)]
        standard_error = scistats.sem(matrx, axis=0)
        print "Standard error", standard_error
        #print "bonus example: ", [t.bonus for t in trial]
        #bonus = np.sum([np.sum([float(b[1]) for b in t.bonus]) for t in trial])
        #print "Bonus:", bonus
        
        
        print "\n-----------------\nTrial " + str(trial_index) + ":\nItem: " + itm
        print "\nSetup questions:"
        for q in trial[0].qa:
            pass
            print features[int(float(q[0]))], q[1]
        print "\nQuestion options (in same order as in vectors):"
        for q in trial[0].questions:
            pass
            print features[int(float(q))]
            
        bestModel = 0
        biggestCor = 0
        bestOrdered = 0
        bestpearson = 0
        bestChance = 0
        bestForChance = 0
        #f, axarr = plt.subplots(5, 4, sharex = False, sharey = False)
        f, axarr = plt.subplots(1, 1)
        
        #f.text(0.5, 0.04, 'Expected Information Gain', ha='center', va='center')
        #f.text(0.03, 0.5, 'Average Human Rank', ha='center', va='center', rotation='vertical')
        f.set_size_inches(20, 20, dpi=5000, forward=True)
        f.tight_layout(pad = 5)
       
        
        print len(trial[0].expected)
        #for indx in range(8):
        for indx in range(1):
            
            
       
            #curplot = axarr[x][y]
            curplot = axarr
            if indx == 7 or indx == 8: print ''
            #i = indx if indx != 6 else 16
            i = indx
            #print i
            compare = np.array(trial[0].expected[i])
            ordered = [sorted(compare, key = lambda x: x)[::-1].index(elem) for elem in compare]
            thresh = 10
            ordered = [thresh if elem >= thresh else elem for elem in ordered]
            #print np.argmax(trial[0].expected[i]), np.argmin(matrx[0]), matrx[0]
            chanceGuess = np.average([1 if np.argmin(vec) == np.argmax(trial[0].expected[i]) else 0 for vec in matrx])
            isBest = np.argmax(person) == np.argmax(trial[0].expected[i])
            
            normed = True
            
            expected = np.array(trial[0].expected[i])
            reallynormed = expected - np.min(expected)
            reallynormed /= np.max(reallynormed)
            
            cor = scistats.pearsonr(person, reallynormed if normed else expected)[0]
            cor_spearman = scistats.spearmanr(person, reallynormed if normed else expected)[0]
            #normed_expected = np.array(trial[0].expected[i]) / np.sum(trial[0].expected[i])
            
            #if i == 0: expectedandbonus.append([bonus, expected[np.argmax(person)]])
            
           
            
            print nameof(i), [format(elem, '.2f') for elem in person], \
                    [format(elem, '.2f') for elem in reallynormed], format(cor, '.3f'), ("YES" if isBest else "NO")
                    
            y = indx % 5
            x = indx / 5
            #print x, y
            '''
            plt.plot(normed_expected, person, 'ro')
            plt.title("Trial=" + str(trial_index) + " x=" + str(x) + " y=" + str(y))
            plt.show()
            plt.close()
            '''
            expandX = 0.1*(max(reallynormed) - min(reallynormed))
            newX = [min(reallynormed) - expandX, max(reallynormed) + expandX]
            
            expandY = 0.1*(max(person) - min(person))
            newY = [min(person) - expandY, max(person) + expandY]
            
            #curplot.axis(newX[0], newX[1], newY[0], newY[1])
            curplot.set_xlim(newX)
            curplot.set_ylim(newY)
            
            isBest = np.argmax(person) == np.argmax(trial[0].expected[i])
            
            display_text = True
            if display_text:
                for k in range(len(reallynormed)):
                    curplot.text(reallynormed[k], person[k], chr(k + ord('A')), horizontalalignment = 'center', verticalalignment = 'center', fontsize=fsize)
                
            size = 1 if display_text else 600
            curplot.scatter(list(reallynormed), list(person),  s=size, color="blue", zorder=2)
            curplot.scatter(newX, newY,  s=1, color="blue", zorder=2)
            curplot.errorbar(reallynormed, person, fmt=',', yerr = standard_error, elinewidth=2, capsize=10, color="red", zorder=1)
        
            #curplot.set_title(nameofNojustShort(i) + "\nPearson: " + format(cor, '0.2f') + ", Spearman: " + format(cor_spearman, '0.2f') + ", " + ("YES" if isBest else "NO"), fontsize=fsize)
            curplot.set_title("$r=$" + format(cor, '0.2f') + ", $\\rho=$" + format(cor_spearman, '0.2f'), fontsize=fsize, y = 1.1)
            curplot.set_ylim([0, 5])
            
            start, end = curplot.get_xlim()
            curplot.xaxis.set_ticks([0, 1])
            curplot.tick_params('both', length=5, width=2, which='major', pad = 15)
            
            
            #labels[0] =  "{:.4}".format(labels[0])
            #labels[1] =  "{:.4}".format(labels[1])
            
            #curplot.set_xticklabels(labels)
            
            #curplot.set_xlim([0, max(normed_expected)])
            #curplot.xaxis.set_ticks([a*max(normed_expected)/2.0 for a in range(2)])
            curplot.axes.get_xaxis().set_visible(True)
            curplot.axes.get_yaxis().set_visible(True)
            plt.sca(curplot)
            #plt.xticks([  float(format(a*max(normed_expected)/4.0, '0.2f')) for a in range(4)])
            #print nameofNojust(i)
            #curplot.title("Trial, " +  str(trial_index) + " " + nameofNojust(i) + " " + format(biggestCor, '.3f'))
            #curplot.xlabel("Expected Information Gain")
            #curplot.ylabel("Average Human Rank")
            #curplot.set_title(nameofNojust(i))
            if cor > biggestCor:
                bestModel = i
                bestOrdered = ordered
                biggestCor = cor
                bestpearson = biggestCor
                
            if chanceGuess > bestChance:
                bestChance = chanceGuess
                bestForChance = i
                
        model = np.array(trial[bestModel].expected[i])
        normed_model = model / np.sum(model)
        #plt.plot(model, person, 'ro')
        #plt.title("Trial, " +  str(trial_index) + " " + nameofNojust(bestModel) + " " + format(biggestCor, '.3f'))
        #plt.show()
        #print bestOrdered
        sumtot += biggestCor
        numtot += 1
        
        '''
        sumtotChance += bestChance
        numtotChance += 1
        '''
        '''
        sumtotChance += 1.0 if isBest else 0.0
        numtotChance += 1.0
        '''
        #f.set_title("Trial " + str(trial_index))
        print "saving:", trial_index
        f.savefig("new_plots/Trial_" + str(trial_index) + "_ticks_lettered.pdf")
        #axarr.set_title("Trial " + str(trial_index))
        #f.show()
        #f.close()
        print '\nBest model for this trial pearson:\n', nameofNojust(bestModel), format(biggestCor, '.3f')
        print '\nBest model for this trial chance guess:\n', nameofNojust(bestForChance), format(bestChance, '.3f')
    #plt.show()
    
    print "\n\nAverage pearson with best for each trial:"
    print sumtot / float(numtot)
'''
print "\nAverage Chance guess with best for each trial:"
print sumtotChance / float(numtotChance)
'''

def calcfor(clustindx):
    sumtot = 0
    numtot = 0
    sumtot_spearman = 0
    numtotChance = 0
    sumtotChance = 0
    cors = []
    corspears = []
    if not testing_times_seen:
        for indx, trial in zip(range(1000), trials):
            matrx = np.array([t.rankorder for t in trial])
            
            #matrx[matrx >= 2] = 2
            person = [5-x for x in np.average(matrx, axis=0)]
           
            #print "\nNew Trial:"
            for q in trial[0].qa:
                pass
                #print features[int(float(q[0]))], q[1]
            
            for q in trial[0].questions:
                pass
                #print features[int(float(q))]
                
    
            i = clustindx
            compare = np.array(trial[0].expected[i])
            ordered = [sorted(compare, key = lambda x: x)[::-1].index(elem) for elem in compare]
            thresh = 10
            ordered = [thresh if elem >= thresh else elem for elem in ordered]
            #print np.argmax(trial[0].expected[i]), np.argmin(matrx[0]), matrx[0]
            chanceGuess = np.average([1 if np.argmin(vec) == np.argmax(trial[0].expected[i]) else 0 for vec in matrx])
            isBest = np.argmax(person) == np.argmax(trial[0].expected[i])
            cor = scistats.pearsonr(person, trial[0].expected[i])[0]
            cor_spearman = scistats.spearmanr(person, trial[0].expected[i])[0]
            cors.append(cor)
            corspears.append(cor_spearman)
            #print cor
            #print ["{:.4}".format(elem) for elem in person], ordered, \
            #        ["{:.3}".format(elem) for elem in trial[0].expected[i]], cor, cor_spearman,\
            #        "YES" if np.argmax(person) == np.argmax(trial[0].expected[i]) else "NO"
            #variables_matrix = [times_seen[i]]
            print cor, cor_spearman
        
            model = np.array(trial[0].expected[i])
            #plt.plot(model, person, 'ro')
            #plt.show()
            #print bestOrdered
            sumtot_spearman  += cor_spearman
            sumtot += cor
            numtot += 1
            #print cor
            
            sumtotChance += 1.0 if isBest else 0.0
            numtotChance += 1.0
            #print bestModel, biggestCor
    
    else:
        tot = 0
        num = 0
        totspear = 0
        numspear = 0
        cors = []
        spears = []
        for i in range(len(times_seen)):
            for trial, indx in zip(trials, range(1000)):
                matrx = np.array([t.rankorder for t in trial])
                they_saw = [times_seen[i][int(q)] for q in trial[0].questions]
                they_saw = trial[0].expected[0]
                #matrx[matrx >= 2] = 2
                person = [5-x for x in np.average(matrx, axis=0)]
                person = [5-x for x in matrx[i]]
                they_saw[-1] += 0.00001
                cor = scistats.pearsonr(person, they_saw)[0]
                cors.append(cor)
                corspear = scistats.spearmanr(person, they_saw)[0]
                spears.append(corspear)
                #print person
                #print they_saw
                #print cor
                #print ""
                if not (cor > 0 or cor < 0 or cor == 0): 
                    print person
                    print they_saw
                    print scistats.pearsonr(person, they_saw)
                    print '\n'
                    cor = 0
                
                tot += cor
                num += 1
                totspear += corspear
        print tot, totspear, num
        print "Pearson:", float(tot)/float(num)
        print "Spearman:", float(totspear)/float(num)
    
    if testing_times_seen: return tot / float(num)
    else: 
        print "average depth 2:", np.average(cors[1:4]), np.average(corspears[1:4])
        print "average depth 4:", np.average(cors[4:7]), np.average(corspears[4:7])
        print "average depth 6:", np.average(cors[7:10]), np.average(corspears[7:10])
        print "average all:", np.average(cors), np.average(corspears)
        return sumtot / float(numtot), float(sumtot_spearman) / float(numtot), sumtotChance / float(numtotChance)
    
print expectedandbonus
print '\n\n\n***********\n\n'

for indx in range(1):    
    #if indx == 6: print ''
    #i = indx if i != 6 else 16
    i = indx
    
    print '\n*************\n\n'
    #calcfor(i)

    calced = calcfor(i)
    if not testing_times_seen:
        print '\nName:', nameof(i)
        print 'Average pearson: ' + format(calced[0], '.3f')
        print 'Average spearman: ' + format(calced[1], '.3f')
        print 'Average chance guess:' + format(calced[2], '.3f')
        print '\n*************\n'

    
'''
for s, i in zip(avgrank, range(100)):
    print "\n***************\n", (2**i)*4, " Clusts:"
    for working, j in zip(s, range(100)):
        print "\nDepth =", 2*j
        for elem, i in zip(working, range(100)):
            print "{:.4}".format(float(elem[0]) / float(elem[1]))
        
'''

        