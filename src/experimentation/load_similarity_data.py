from import_all_files import *


import sklearn.metrics
import sklearn
from sklearn.metrics import pairwise_distances
import numpy as np

def get_data():

	data_matrix = np.loadtxt(base_path+"data/intel_dat.txt")


	with open(base_path+"data/intel_feat.txt", "r") as feats:
	    features = [l[:-2] for l in feats]

	with open(base_path+"data/intel_items.txt", "r") as items:
	    items = [l[:-2] for l in items]


	sim_matrix_items = pairwise_distances(data_matrix, metric='euclidean')
	sim_matrix_features = pairwise_distances(data_matrix.T, metric='correlation') 

	return data_matrix, items, features, sim_matrix_items, sim_matrix_features