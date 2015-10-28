'''
Created on Oct 22, 2015

@author: alxcoh
'''
from sqlalchemy import create_engine, MetaData, Table
import numpy as np
import json
import pandas as pd
from matplotlib import pyplot as plt
import cPickle as pickle
from runner_clust import *
from matplotlib import pyplot as plt
import matplotlib
import scipy.stats as scistats
from get_serverdata import Oneshot, Trial

with open("oneshots3.pickle", 'r') as oneshotfile:
    oneshotlist = pickle.load(oneshotfile)
print "im here!"
trials = [[] for i in range(10)]
for o in oneshotlist:
    for t, i in zip(o.trials, range(100)):
        trials[i].append(t)

def nameof(index):
    if index < 10:
        return ("Bayesian " + str(min(2**(index+1), 1000)) + ":").ljust(23)
    else:
        return ("Expected Utility " +  str(min(2**(index-9), 1000)) + ":").ljust(23)

def nameofNojust(index):
    if index < 10:
        return "Bayesian " + str(min(2**(index+1), 1000)) + ":"
    else:
        return "Expected Utility " +  str(min(2**(index-9), 1000)) + ":"
    
def nameofNojustShort(index):
    if index < 10:
        return "Bayesian " + str(min(2**(index+1), 1000)) + ":"
    else:
        return "Utility " +  str(min(2**(index-9), 1000)) + ":"

trials = sorted(trials, key = lambda x: x[0].depth)
sumtot= 0
numtot = 0

sumtotChance = 0
numtotChance = 0
for trial, trial_index in zip(trials, range(100)):
    
    itm = filter(lambda x: x[0] == 'oneshot_itm_chosen', trial[0].data)[0][1][2]
    
    
    matrx = np.array([t.rankorder for t in trial])
    
    #matrx[matrx >= 2] = 2
    person = [5-x for x in np.average(matrx, axis=0)]
    
    
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
    f, axarr = plt.subplots(4, 5, sharex = False, sharey = False)
    
    f.text(0.5, 0.04, 'Expected Information Gain', ha='center', va='center')
    f.text(0.03, 0.5, 'Average Human Rank', ha='center', va='center', rotation='vertical')
    f.set_size_inches(20, 15, dpi=5000, forward=True)
    f.tight_layout(pad = 5)
  
    for i in range(0, 20):
        if i == 10: print ''
        
        compare = np.array(trial[0].expected[i])
        ordered = [sorted(compare, key = lambda x: x)[::-1].index(elem) for elem in compare]
        thresh = 10
        ordered = [thresh if elem >= thresh else elem for elem in ordered]
        #print np.argmax(trial[0].expected[i]), np.argmin(matrx[0]), matrx[0]
        chanceGuess = np.average([1 if np.argmin(vec) == np.argmax(trial[0].expected[i]) else 0 for vec in matrx])
        isBest = np.argmax(person) == np.argmax(trial[0].expected[i])
        cor = scistats.pearsonr(person, trial[0].expected[i])[0]
        #normed_expected = np.array(trial[0].expected[i]) / np.sum(trial[0].expected[i])
        normed_expected = np.array(trial[0].expected[i])
        reallynormed = normed_expected / np.sum(normed_expected)
        
        print nameof(i), [format(elem, '.2f') for elem in person], \
                [format(elem, '.2f') for elem in reallynormed], format(cor, '.3f'), ("YES" if isBest else "NO")
                
        y = i % 5
        x = i / 5
        #print x, y
        '''
        plt.plot(normed_expected, person, 'ro')
        plt.title("Trial=" + str(trial_index) + " x=" + str(x) + " y=" + str(y))
        plt.show()
        plt.close()
        '''
        expandX = 0.1*(max(normed_expected) - min(normed_expected))
        newX = [min(normed_expected) - expandX, max(normed_expected) + expandX]
        
        expandY = 0.1*(max(person) - min(person))
        newY = [min(person) - expandY, max(person) + expandY]
        
        #axarr[x][y].axis(newX[0], newX[1], newY[0], newY[1])
        axarr[x][y].set_xlim(newX)
        axarr[x][y].set_ylim(newY)
        
        isBest = np.argmax(person) == np.argmax(trial[0].expected[i])
        fsize = 16
        #axarr[x][y].plot(list(normed_expected) + newX, list(person) + newY, ',')
        for k in range(len(normed_expected)):
            axarr[x][y].text(normed_expected[k], person[k], chr(k + ord('A')), horizontalalignment = 'center', verticalalignment = 'center', fontsize=fsize)
    
            
        font = {'family' : 'normal',
                'size'   : fsize}

        matplotlib.rc('font', **font)
    
        axarr[x][y].set_title(nameofNojustShort(i) + "\n" + format(cor, '0.2f') + ", " +  ("YES" if isBest else "NO"), fontsize=fsize)
        axarr[x][y].set_ylim([0, 5])
        
        start, end = axarr[x][y].get_xlim()
        axarr[x][y].xaxis.set_ticks([start, end])
        axarr[x][y].tick_params('both', length=5, width=2, which='major')
        #labels[0] =  "{:.4}".format(labels[0])
        #labels[1] =  "{:.4}".format(labels[1])
        
        #axarr[x][y].set_xticklabels(labels)
        
        #axarr[x][y].set_xlim([0, max(normed_expected)])
        #axarr[x][y].xaxis.set_ticks([a*max(normed_expected)/2.0 for a in range(2)])
        axarr[x][y].axes.get_xaxis().set_visible(True)
        axarr[x][y].axes.get_yaxis().set_visible(True)
        #plt.sca(axarr[x][y])
        #plt.xticks([  float(format(a*max(normed_expected)/4.0, '0.2f')) for a in range(4)])
        #axarr[x][y].title("Trial, " +  str(trial_index) + " " + nameofNojust(i) + " " + format(biggestCor, '.3f'))
        #axarr[x][y].xlabel("Expected Information Gain")
        #axarr[x][y].ylabel("Average Human Rank")
        #axarr[x][y].set_title(nameofNojust(i))
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
    #f.savefig("new_plots/Trial_" + str(trial_index))
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
    numtotChance = 0
    sumtotChance = 0
    for trial in trials:
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
        cor = scistats.pearsonr(person, trial[0].expected[i])[0]
        
        isBest = np.argmax(person) == np.argmax(trial[0].expected[i])
        
        chanceGuess = np.average([1 if np.argmin(vec) == np.argmax(trial[0].expected[i]) else 0 for vec in matrx])
        
        #print ["{:.4}".format(elem) for elem in person], ordered, ["{:.3}".format(elem) for elem in trial[0].expected[i]], cor, biggestCor


                
        model = np.array(trial[0].expected[i])
        #plt.plot(model, person, 'ro')
        #plt.show()
        #print bestOrdered
        sumtot += cor
        numtot += 1
        #print cor
        
        sumtotChance += 1.0 if isBest else 0.0
        numtotChance += 1.0
        #print bestModel, biggestCor
    return sumtot / float(numtot), sumtotChance / float(numtotChance)
    
print '\n\n\n***********\n\n'
for i in range(20):
    if i == 10: print ''
    calced = calcfor(i)
    print '\n', nameof(i) + '\nAverage pearson: ' + format(calced[0], '.3f') + '\nAverage chance guess:' + format(calced[1], '.3f')

'''
for s, i in zip(avgrank, range(100)):
    print "\n***************\n", (2**i)*4, " Clusts:"
    for working, j in zip(s, range(100)):
        print "\nDepth =", 2*j
        for elem, i in zip(working, range(100)):
            print "{:.4}".format(float(elem[0]) / float(elem[1]))
        
'''

        