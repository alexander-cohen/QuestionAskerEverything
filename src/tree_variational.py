'''
Created on Dec 5, 2015

@author: alxcoh
'''
import numpy as np
from scipy.stats import entropy
import cPickle as pickle
import math
import data_probs_generator

def which_cluster(clusters, item):
    for clust_index, c in zip(range(len(clusters)), clusters):
        if item in c: return clust_index
    #if item == 0: print "Could not find:", items[item], item, item in clusters[2], '\n', '\n'.join([repr(list(elem)) for elem in clusters])
    

class ClustTree():
    def __init__(self, val):
        self.left = None
        self.right = None
        self.val = val
        self.leaf = True
        self.depth = 0
        self.lindex = 0 #going from left to right, the number
    
    def setleft(self, l):
        self.left = l
        self.leaf = False
    
    def setright(self, r):
        self.right = r
        self.leaf = False
        
    def getleaves(self, all = []):
        if self.leaf: 
            #print self.left
            #print self.right
            return list(self.val)
        else:
            return self.left.getleaves(all) + self.right.getleaves(all)
    
    def mendleaves(self):
        if self.leaf:
            if len(self.val) > 1:
                self.setleft(ClustTree([self.val[0]]))
                self.setright(ClustTree([self.val[1]]))
        else:
            self.left.mendleaves()
            self.right.mendleaves()
            
    def numleaves(self, n=0):
        if self.leaf:
            return n+1
        else:
            return self.left.numleaves() + self.right.numleaves()
        
        
    def get_all_deep(self, desired_depth, depth=0):
        if depth == desired_depth or self.leaf:
            print self.lindex
            return [self.val]
        else:
            return list(self.left.get_all_deep(desired_depth, depth+1)) + list(self.right.get_all_deep(desired_depth, depth+1))
    
    def set_node_coords(self):
        openset = [self.left, self.right]
        depth = 1
        self.depth = 0
        self.lindex = 0
        while True:
            lindex = 0
            new_openset = []
            
            for n in openset:
                n.depth = depth
                n.lindex = lindex
                if n.left != None: new_openset.append(n.left)
                if n.right != None: new_openset.append(n.right)
                lindex += 1
            
            openset = new_openset
            if openset == []: break
            depth += 1
            
    def printrow(self, row):
        openset = [self.left, self.right]
        depth = 1
        self.depth = 0
        self.lindex = 0
        while True:
            print depth, row
            new_openset = []
            
            for n in openset:
                if depth == row: print list(n.val)
                if n.left != None: new_openset.append(n.left)
                if n.right != None: new_openset.append(n.right)
                
            if depth > row: break
            
            openset = new_openset
            if openset == []: break
            depth += 1
            
    
    def build_from_list(self, treelist, top = True):
        if treelist == None: return
        self.val = treelist[0]
    
        
        self.depth

        left = None
        right = None
        if treelist[1][0] == None: left = None
        else:
            left = ClustTree(None)
            left.build_from_list(treelist[1][0], False)
            
        if treelist[1][1] == None: right = None
        else:
            right = ClustTree(None)
            right.build_from_list(treelist[1][1], False)
            
        self.setleft(left)
        self.setright(right)
        if self.left == None and self.right == None: self.leaf = True
        
    
    def tolist(self):
        llist = None if self.left == None else self.left.tolist()
        rlist = None if self.right == None else self.right.tolist()
        return [self.val, [llist, rlist]]
    
    def update_depth(self):
        new = ClustTree()
        new.build_from_list(self.tolist())
        self = new
        
 
    
    def __str__(self):
        return str(self.tolist())
    
def set_prob_resp_tree(tree):
    tree.prob_resp = data_probs_generator.prob_resps_oneclust(tree.val)
    print np.shape(tree.val), np.shape(tree.prob_resp)
    #print np.shape(tree.prob_resp)
    if tree.leaf: return
    else:
        set_prob_resp_tree(tree.left)
        set_prob_resp_tree(tree.right)

'''

with open("../data/clusttree_list.pickle") as clust_tree_file:
    clusttree_list = pickle.load(clust_tree_file)
    
clusttree = ClustTree(np.array(range(1000)))
clusttree.build_from_list(clusttree_list)
clusttree.set_node_coords()
set_prob_resp_tree(clusttree)

with open('../data/clusttree.pickle', 'w') as clusttree_pickle_file:
    pickle.dump(clusttree, clusttree_pickle_file)

'''
    


#print '\n'.join([str(list(elem)) for elem in clusttree.get_all_deep(5)])
#print '\n\n*************\n\n'
#clusttree.printrow(5)