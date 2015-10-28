'''
Created on Aug 31, 2015

@author: alxcoh
'''

import scipy
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import math
import random
from runner_clust import *
from multiprocessing import Pool
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
'''
model = TSNE(n_components = 2, init='pca', verbose=1)
print model.fit_transform(data_matrix)
'''
'''
pca = PCA()
pca.fit(data_matrix)
mtrx = pca.transform(data_matrix)
print pca.explained_variance_ratio_, '\n'
print np.cumsum(pca.explained_variance_ratio_), '\n'
print pca.components_, '\n'
print '\n'.join([repr(x) for x in zip(list(pca.transform(data_matrix[items.index('dog')])[0]), list(pca.transform(data_matrix[items.index('sofa')])[0]))]), '\n'



print [items[i] for i in sorted(range(1000), key = lambda x: sum([ (10**n) * mtrx[x][n] for n in range(2) ]))]
#print pca.transform(data_matrix[items.index('cat')]), '\n'
'''
print "starting file"

itemlist = range(len(items))
featurelist = range(len(features))
numitms = float(len(items))

clusts_index = all_clust_matrix[4][0]
clusts = [np.where(np.array(clusts_index) == val)[0] for val in list(set(clusts_index))]

starting_clusts = clusts
tot_num = float(sum([len(c) for c in starting_clusts]))
num_clusts = len(clusts)

print "computing initial"

clust_prior = np.array([float(len(c))/tot_num for c in starting_clusts]) #prior prob of clust
probfr_clust = np.array([ [ [float(sum([data_probs_temp[r][i][f] for i in c])) / float(len(c)) for r in range(5)] for f in featurelist]  for c in starting_clusts]) #clust x feature x response

clusts = np.array(range(len(starting_clusts))) #index for the clusters

knowledge = [] #make this non empty for putting in middle of game

print "making player"
myplayer = ClustPlayer(9)
myplayer.knowledge = knowledge
myplayer.update_all()
print "player completed"

def val_to_indx(val):
    return int(2*(val+1))

def index_to_val(index):
    return (float(index) / 2.0) - 1


def E(probitemdata):
    q = []
    for z in clusts:
        probclust_fromitem = []
        prior = clust_prior[z]
        for i in itemlist:
            prob = prior
        
            prob *= probitemdata[i][z]
            probclust_fromitem.append(prob)
            
        probclust_fromitem = np.array(probclust_fromitem)
        
        q.append(probclust_fromitem)
    
    q = np.array(q)
    q = q / np.sum(q, axis=0)
    return q

def item_posterior(tup):
    clust = tup[0]
    probitemdata = tup[1]
    posterior = []
    for itm in itemlist:
        '''
        tempplayer = ClustPlayer(9)
        tempplayer.knowledge = [(f, data_matrix[itm][f]) for f in featurelist]
        tempplayer.update_all()
        posterior.append(tempplayer.probabilities[itm] * probitemdata[itm][clust] ) #templayer.probabilities[itm] approx = 1
        '''
        posterior.append(probitemdata[itm][clust])
    
    posterior = np.array(posterior)
    posterior /= np.sum(posterior)
    #print "computed posterior for", clust
    return posterior


def fvforclust(post):
    returner = [[np.sum([ data_probs_temp[r][i][f] * post[i] for i in itemlist]) for r in range(5)] for f in featurelist]
    #print "Clust finished"
    return returner

def M(q, probitemdata):
    global probfr_clust
    global clust_prior
    for z in clusts:
        qlist = q[z]
        #print '\n', np.sum(qlist)
        clust_prior[z] = np.sum( [qlist[i] * 0.001 for i in itemlist] )
    
    
    
    print "Priors computed"
    p = Pool()
    posteriors = p.map(item_posterior, [(z, probitemdata) for z in clusts]) 
    p.terminate()
    print "Posteriors computed"
    p = Pool()
    probfr_clust = p.map(fvforclust, posteriors)
    p.terminate()
        
    print "M-step finished"
    
def update():
    probitemdata = []
    for i in itemlist:
        foritem = []
        row = data_matrix[i]
        indexrow = [val_to_indx(row[f]) for f in featurelist]
        for z in clusts: 
            foritem.append(np.product( [probfr_clust[z][f][v]  for f, v in zip(featurelist, indexrow)] ) )
        #print items[i], np.sum( [elem for elem in foritem] )
        probitemdata.append(foritem)
        
    probitemdata = np.array(probitemdata)
    '''
    probitems = probitemdata / np.sum(probitemdata, axis=0)
    #print np.sum(probitems[0]), np.sum(probitems.T[0])
    probitems_total = np.array([np.sum([ probitems[i][z]*clust_prior[z] for z in clusts]) for i in itemlist])
    print clust_prior
    for c in starting_clusts:
        print [items[elem] for elem in c]
    print '\n'.join([repr((items[i], probitems_total[i])) for i in itemlist])
    '''
    '''    
    probitemdata = np.array(probitemdata) 
    best = np.argmax([np.sum( [probitemdata[i][z] * clust_prior[z]  for z in clusts])  for i in itemlist])
    print sorted([np.sum( [probitemdata[i][z] * clust_prior[z]  for z in clusts])  for i in itemlist])
    print probitemdata[items.index('essay')]
    '''
    #print probitemdata[0]
    print "\n\n*************\nLikelihood:", np.sum([ math.log( np.sum([probitemdata[i][z] * clust_prior[z]  for z in clusts]) ) for i in itemlist])
    q = E(probitemdata)

    probitems_frmclust = [ [ for z in clusts] for i in itemlist]
    print np.sum(q, axis=1)
    print np.sum(clust_prior), sorted(clust_prior)
    print np.sum(probitems), sorted(probitems)

    print "E-step finished"
    M(q, probitemdata)

while True:
    update()