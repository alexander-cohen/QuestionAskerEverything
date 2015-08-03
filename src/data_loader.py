'''
Created on Jun 22, 2015

@author: alxcoh
'''

import numpy as np
import scipy.spatial.distance as ssd




def get_data():
    data_matrix_np = np.loadtxt("../data/intel_dat.txt")
    features = [l[:-2] for l in open("../data/intel_feat.txt", "r")]
    objects = [l[:-2] for l in open("../data/intel_items.txt", "r")]
    
    data_dict = {}

    for i in range(len(data_matrix_np)):
        to_add = {}
        r = data_matrix_np[i]
        for j in range(len(r)):
            val = r[j]
            to_add[features[j]] = val
        #print objects[i]
        data_dict[objects[i]] = to_add
          
    return data_matrix_np, data_dict, features, objects