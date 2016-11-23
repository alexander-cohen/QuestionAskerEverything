from import_all_files import *

from runner_numpy import *

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import scipy
import time
import random

import sys
sys.setrecursionlimit(2000)

def split_kmeans(elements):
	clustering = KMeans(n_clusters = 2, n_init = 200)
	clustering.fit(data_matrix[elements])
	return elements[clustering.labels_ == 0], elements[clustering.labels_ == 1]



n_completed = 0

def create_cluster_tree_kmeans(elements, path = ''):
	global n_completed

	if len(list(elements)) == 1: 
		if n_completed % 10 == 0: print '{}%'.format(n_completed / 10)
		n_completed += 1

		return [elements[0]]

	else:
		left, right = split_kmeans(elements)
		return [create_cluster_tree(left, method = method, path = path + 'l'), create_cluster_tree(right, method = method, path = path + 'r')]

def create_cluster_linkage_kmeans(elements, min_name):
	global n_completed

	if len(list(elements)) == 1:
		if n_completed % 10 == 0: print '{}%'.format(n_completed / 10)
		n_completed += 1

		return elements[0], []

	else:
		left, right = split_kmeans(elements)		
		node_name_left, linkage_left = create_cluster_linkage_kmeans(left, min_name)
		node_name_right, linkage_right = create_cluster_linkage_kmeans(right, max(node_name_left, min_name))

		new_name = max(min_name, node_name_left, node_name_right) + 1
		return new_name, linkage_left + linkage_right + [[node_name_left, node_name_right, new_name]]

def get_tree_rules_from_linkage(linkage):
	rules = {}
	for l in linkage:
		rules[l[2]] = [l[0], l[1]]
	return rules

def save_cluster_tree(method = 'kmeans'):
	global n_completed
	n_completed = 0

	root, linkage = create_cluster_linkage_kmeans(np.arange(1000), 999)
	rules = get_tree_rules_from_linkage(linkage)

	with open("/pickled_data/cluster_{}_linkage.pickle".format(method), 'w') as pfile:
		pickle.dump(linkage, pfile)

	with open(base_path+"pickled_data/cluster_{}_rules.pickle".format(method), 'w') as pfile:
		pickle.dump(rules, pfile)

	with open(base_path+"pickled_data/cluster_{}_linkage.txt".format(method), 'w') as f:
		f.write('\n'.join([', '.join([str(e) for e in l]) for l in linkage]))


def retrieve_cluster_tree(method = 'kmeans'):
	with open(base_path+"pickled_data/cluster_{}_linkage.pickle".format(method), 'r') as pfile:
		linkage = pickle.load(pfile)

	with open(base_path+"pickled_data/cluster_{}_rules.pickle".format(method), 'r') as pfile:
		rules = pickle.load(pfile) 

	root = linkage[-1][-1]
	
	return linkage, rules, root


linkage, rules, root = retrieve_cluster_tree()

elements_below = {}
nodes_below = {}

def create_elems_below(node):
	if node < 1000: 
		elements_below[node] = [node]
		return [node]
	else:
		left, right = rules[node]
		all_below = create_elems_below(left) + create_elems_below(right)
		elements_below[node] = all_below
		return all_below

def create_nodes_below(node):
	if node < 1000:
		nodes_below[node] = []
		return []
	else:
		left, right = rules[node]
		all_below = [left, right] + create_nodes_below(left) + create_nodes_below(right)
		nodes_below[node] = all_below
		return all_below

create_elems_below(root)
create_nodes_below(root)
total_num_nodes = 1999



class VariationalPlayer2(OptimalPlayer):
	def __init__(self, nclusts, method = 'greedy', verbose = False, value_func = None):
		self.verbose = verbose
		self.nclusts = nclusts
		self.priors = np.zeros(1999)
		self.nbelow = np.zeros(1999)

		self.posteriors = np.zeros(1999)
		self.logprob_all_below = np.zeros(1999)

		self.temp_posteriors = np.zeros(1999)
		self.values = np.zeros(1999)

		self.partition = None
		self.calculate_priors(root)
		self.clusterset = [root]

		self.optimize_method = method

		if value_func == None: self.value_func = VariationalPlayer2.kl_p_from_q
		else: self.value_func = value_func


		if self.value_func.__name__ == self.kl_p_from_q.__name__: self.value_func_all = VariationalPlayer2.kl_p_from_q_all
		elif self.value_func.__name__ == self.kl_q_from_p.__name__: self.value_func_all = VariationalPlayer2.kl_q_from_p_all

		elif self.value_func.__name__ == self.entropy_label_space.__name__: self.value_func_all = VariationalPlayer2.entropy_label_space_all
		elif self.value_func.__name__ == self.entropy_object_space.__name__: self.value_func_all = VariationalPlayer2.entropy_object_space_all
		elif self.value_func.__name__ == self.combined_cost.__name__: self.value_func_all = VariationalPlayer2.combined_cost_all
		
		else: self.value_func_all = None 


		super(VariationalPlayer2, self).__init__()

	def is_leaf(self, node):
		return node < 1000

	def calculate_partition(self):
		self.partition = np.sum(self.prob_knowledge_from_items)

	def calculate_priors(self, node):
		if self.is_leaf(node):
			self.priors[node] = 0.001
			self.nbelow[node] = 1
			return 0.001

		else:
			left, right = rules[node]
			prior = self.calculate_priors(left) + self.calculate_priors(right)
			self.priors[node] = prior
			self.nbelow[node] = len(elements_below[node])

			return prior

	def calculate_logprob_below(self, node):
		if self.is_leaf(node):
			self.logprob_all_below[node] = math.log(self.posteriors[node])
			return self.logprob_all_below[node]
		
		else:
			left, right = rules[node]
			logprob_sum = self.calculate_logprob_below(left) + self.calculate_logprob_below(right)
			self.logprob_all_below[node] = logprob_sum

			return logprob_sum

	def calculate_posterior(self, node, initial_probs, posterior_list):
		if self.is_leaf(node):
			posterior_list[node] = initial_probs[node]
			return posterior_list[node]
		
		else:
			left, right = rules[node]
			posterior = self.calculate_posterior(left, initial_probs, posterior_list) + self.calculate_posterior(right, initial_probs, posterior_list)
			posterior_list[node] = posterior

			return posterior

	'''
	#D_KL(P||Q) =  H(P, Q) - H(P), equivalent to H(P, Q)
	#the expected information lost when approximating P with Q
	#P is the true posterior
	#Q is the approximated cluster posterior
	def kl_p_from_q(self, node):
		nb = float(self.nbelow[node])
		prob_each = self.posteriors[node] / nb
		return entropy(self.posteriors[elements_below[node]], np.array([prob_each] * int(nb)))

	def kl_p_from_q_all(self, true_posterior, approx_posterior, clusterset):
		return entropy(true_posterior + 0.00001, approx_posterior + 0.00001)

	#D_KL(Q||P) = H(Q, P) - H(Q)
	#the expected information lost when approximating Q with P
	#P is the true posterior
	#Q is the approximated cluster posterior
	def kl_q_from_p(self, node):
		nb = float(self.nbelow[node])
		prob_each = self.posteriors[node] / nb

		return entropy([prob_each] * int(nb), self.posteriors[elements_below[node]])


	def kl_q_from_p_all(self, true_posterior, approx_posterior, clusterset):
		return entropy(approx_posterior, true_posterior)

	#H(Q) computer over C rather then over O
	#the information content of the posterior in label space
	#P is the true posterior
	#Q is the approximated cluster posterior
	#no negative sign because we want to maximize the entropy, not minimize
	def entropy_label_space(self, node):
		posterior = self.posteriors[node]
		nbelow = self.nbelow[node]

		return posterior * math.log(posterior)

	def entropy_label_space_all(self, true_posterior, approx_posterior, clusterset):
		return entropy(self.posteriors[clusterset])


	#H(Q) computed over O
	#the information content of the posterior in label space
	#P is the true posterior
	#Q is the approximated cluster posterior
	#exact opposite of KL(P||Q)
	#no negative because we want to maximize the entropy
	def entropy_object_space(self, node):
		nb = float(self.nbelow[node])
		prob_each = self.posteriors[node] / nb

		return entropy([prob_each] * int(nb))

	def entropy_object_space_all(self, true_posterior, approx_posterior, clusterset):
		return entropy(approx_posterior)

	'''

	#D_KL(P||Q) =  H(P, Q) - H(P), equivalent to H(P, Q)
	#the expected information lost when approximating P with Q
	#P is the true posterior
	#Q is the approximated cluster posterior
	def kl_p_from_q(self, node):
		posterior = self.posteriors[node]
		nbelow = self.nbelow[node]

		return -posterior * math.log(posterior / float(nbelow))

	def kl_p_from_q_all(self, true_posterior, approx_posterior, clusterset):
		return entropy(true_posterior + 0.00001, approx_posterior + 0.00001) + entropy(true_posterior + 0.00001)

	#D_KL(Q||P) = H(Q, P) - H(Q)
	#the expected information lost when approximating Q with P
	#P is the true posterior
	#Q is the approximated cluster posterior
	def kl_q_from_p(self, node):
		posterior = self.posteriors[node]
		nbelow = self.nbelow[node]

		logprob = self.logprob_all_below[node]
		
		cross_entropy = -(posterior / float(nbelow)) * logprob #NOTE: not dividing by nbelow here works way better, unclear why
		label_entropy = -posterior * math.log(posterior / float(nbelow))

		return cross_entropy - label_entropy

	def kl_q_from_p_all(self, true_posterior, approx_posterior, clusterset):
		return entropy(approx_posterior, true_posterior)

	#H(Q) computed over C rather then over O
	#the information content of the posterior in label space
	#P is the true posterior
	#Q is the approximated cluster posterior
	#no negative sign because we want to maximize the entropy, not minimize
	def entropy_label_space(self, node):
		posterior = self.posteriors[node]
		nbelow = self.nbelow[node]

		return posterior * math.log(posterior)

	def entropy_label_space_all(self, true_posterior, approx_posterior, clusterset):
		return entropy(self.posteriors[clusterset])

 
	#H(Q) computed over O
	#the information content of the posterior in label space
	#P is the true posterior
	#Q is the approximated cluster posterior
	#exact opposite of KL(P||Q)
	#no negative because we want to maximize the entropy
	def entropy_object_space(self, node):
		posterior = self.posteriors[node]
		nbelow = self.nbelow[node]

		return posterior * math.log(posterior / float(nbelow))

	def entropy_object_space_all(self, true_posterior, approx_posterior, clusterset):
		return entropy(approx_posterior)


	#this is not particularly justified
	def combined_cost(self, node):
		return self.kl_p_from_q(node) + self.entropy_label_space(node)

	def combined_cost_all(self, true_posterior, approx_posterior, clusterset):
		return self.kl_p_from_q_all(true_posterior, approx_posterior, clusterset) + \
				self.entropy_label_space_all(true_posterior, approx_posterior, clusterset)


	def calculate_values(self, node):
		self.values[node] = self.value_func(self, node)

		if self.is_leaf(node): return
		else:
			left, right = rules[node]
			self.calculate_values(left)
			self.calculate_values(right)


	def greedy_optimize(self, current_set):
		#print "in_greedy"
		if len(current_set) == self.nclusts: return current_set

		available = [n for n in current_set if not self.is_leaf(n)]
		values = [sum([self.values[c] for c in rules[node]]) - self.values[node] for node in available] #minimize total value
		
		best_node = available[np.argmin(values)]
		best_index = current_set.index(best_node) #index in the current set of the best node to split on
		
		return sorted(self.greedy_optimize( current_set[:best_index] + current_set[best_index+1:] + rules[best_node] ))

	def get_min_clust_val(self, node, nclusts):
		if self.is_leaf(node) and nclusts > 1: return 10 ** 10, []

		if nclusts == 1: return self.values[node], [node]

		else:
			s = '{},{}'.format(node, nclusts)
			if s in self.min_clustcost: return self.min_clustcost[s]
			left, right = rules[node]
			possibilities = [ [self.get_min_clust_val(left, i), self.get_min_clust_val(right, nclusts - i)] \
								for i in range(1, nclusts) ]


			choice = min(possibilities, key = lambda x: x[0][0] + x[1][0])
			returnval = choice[0][0] + choice[1][0], choice[0][1] + choice[1][1]

			self.min_clustcost[s] = returnval

			#print node, nclusts, returnval

			return returnval


	def fully_optimize(self):
		self.min_clustcost = {}
		return sorted(self.get_min_clust_val(root, self.nclusts)[1])

	def random_optimize(self, current_set):
		#print "in_random"
		if len(current_set) == self.nclusts: return current_set

		available = [n for n in current_set if not self.is_leaf(n)]

		choice = random.randint(0, len(available)-1)
		best_node = available[choice]

		best_index = current_set.index(best_node) #index in the current set of the best node to split on
		
		return sorted(self.random_optimize( current_set[:best_index] + current_set[best_index+1:] + rules[best_node] ))


	def find_clustering(self):
		if self.optimize_method == 'greedy': return self.greedy_optimize([root])
		elif self.optimize_method == 'optimal': return self.fully_optimize()
		else: return self.random_optimize([root])


	def posterior_from_clustering(self, posteriors):
		probs = np.zeros(1000)

		for c in self.clusterset:
			ebelow = elements_below[c]
			probs[ebelow] = posteriors[c] / self.nbelow[c]

		return probs

	def perform_recursive_value_calcs(self):
		self.calculate_partition()
		self.calculate_posterior(root, self.probabilities, self.posteriors)
		self.calculate_logprob_below(root)
		self.calculate_values(root)

	def update_all(self):
		super(VariationalPlayer2, self).update_all()

		self.perform_recursive_value_calcs()
		self.clusterset = self.find_clustering()
		
		if self.verbose:
			print [[features[f], a] for f, a in self.knowledge]
			for c in self.clusterset:
				ebelow = elements_below[c]
				print len(ebelow), self.posteriors[c], [items[e] for e in ebelow]

			print '\n\n'


		self.full_posterior = self.probabilities
		self.probabilities = self.posterior_from_clustering(self.posteriors)	

		self.clustering_cost = self.value_func_all(self, self.full_posterior, self.probabilities, self.clusterset)

		self.entropy = entropy(self.probabilities)
		#print self.clustering_cost, np.sum(self.values[self.clusterset])

	def entropy_with_new_knowledge(self, new_knowledge):
		new_probs = self.prob_with_new_knowledge(new_knowledge)

		self.calculate_posterior(root, new_probs, self.temp_posteriors)

		probs = self.posterior_from_clustering(self.temp_posteriors)


		ent = entropy( probs ) - 100

		#print probs[:5], ent
		return (10 ** 10) if math.isinf(ent) else ent






# player = EpsilonAndNPlayer(2, method = 'greedy', verbose = True)
# player.play_game()

if __name__ == "__main__":
	player = VariationalPlayer2(10, method = 'greedy', verbose = True, value_func = VariationalPlayer2.kl_q_from_p)
	player.play_game()
