'''
Created on Aug 12, 2015

@author: alxcoh
'''

from import_all_files import *
from runner_randomN import *
import random
from multiprocessing import Pool
from sklearn.cluster import KMeans
import scipy.spatial.distance as scidist



class GoodN(RandomN):
    def __init__(self, clusts = 9, n=20, strict_only_consider = True, seed = 1):
        #perform clustering
        if seed == 'random': KMeans(n_clusters = n, n_init = 1)
        else: clustering = KMeans(n_clusters = n, n_init = 1, random_state = seed)
        clustering.fit(data_matrix)

        randomN = []

        #get representative objects
        for c in xrange(n):
            elems = np.arange(1000)[clustering.labels_ == c] #elements of cluster c
            distances = scidist.pdist(data_matrix[elems], 'sqeuclidean') #pairware distance between all elements of cluster
            sum_dist = np.sum(distances, axis = 0)
           
            rep = elems[np.argmin(sum_dist)] #get most central object
            randomN.append(rep)
            print [items[o] for o in elems], items[rep]




        self.strict_only_consider = strict_only_consider
        super(GoodN, self).__init__(clusts, randomN)



class GoodN_averaged():
    def __init__(self, clusts, n_objects, n_simulated_people, strict_only_consider = True, seed = 1):
        random.seed(seed)
        seeds = [random.randint(1, 100000) for x in xrange(n_simulated_people)] #make random seeds for each person

        self.knowledge = []
        self.simulated_people = \
            [GoodN(clusts, n_objects, seed = s) for s in seeds]

    def update_all(self):
        #pool = Pool()

        for p in self.simulated_people:
            p.knowledge = self.knowledge

        map(updater, self.simulated_people)
        #pool.close()

    def expected_gain(self, question):
        return np.average(  [p.expected_gain(question) for p in self.simulated_people] )

    def expected_gains(self):
        return np.average( np.array( [p.expected_gains() for p in self.simulated_people ] ), 0 )

    def get_simulated_people(self):
        return [list(p.get_randomN()) for p in self.simulated_people]

