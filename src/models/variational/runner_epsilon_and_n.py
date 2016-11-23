
from runner_variational2 import *
from sklearn.cluster import KMeans
import random
#from multiprocessing import Pool
import pathos.multiprocessing as mp


viable_pairs = []

for node1 in range(1900, total_num_nodes):
	for node2 in range(node1+1, total_num_nodes):
		if node1 not in nodes_below[node2] and \
		   	node2 not in nodes_below[node1]:

		   viable_pairs.append([node1, node2])



class EpsilonAndNPlayer(VariationalPlayer2):
	def __init__(self, nclusts, epsilon = 0.0001, method = 'greedy', value_func = None, verbose = False):
		self.epsilon = epsilon

		if value_func == None or value_func.__name__ == self.kl_p_from_q.__name__: self.specific_value_func = EpsilonAndNPlayer.kl_p_from_q_specific
		elif value_func.__name__ == self.kl_q_from_p.__name__: self.specific_value_func = EpsilonAndNPlayer.kl_q_from_p_specific

		elif value_func.__name__ == self.entropy_label_space.__name__: self.specific_value_func = EpsilonAndNPlayer.entropy_label_space_specific
		elif value_func.__name__ == self.entropy_object_space.__name__: self.specific_value_func = EpsilonAndNPlayer.entropy_object_space_specific
		elif value_func.__name__ == self.combined_cost.__name__: self.specific_value_func = EpsilonAndNPlayer.combined_cost_specific
		
		else: self.specific_value_func = self.specific_value_func = EpsilonAndNPlayer.kl_p_from_q_specific 

		super(EpsilonAndNPlayer, self).__init__(nclusts, method = method, verbose = verbose, value_func = value_func)


	def kl_p_from_q_specific(self, real_prob, adjusted_prob, sum_log_prob, prob_for_each):
		cross_entropy = - real_prob * math.log(prob_for_each)
		return cross_entropy

	def kl_q_from_p_specific(self, real_prob, adjusted_prob, sum_log_prob, prob_for_each):
		cross_entropy = - prob_for_each * sum_log_prob
		label_entropy = - adjusted_prob * math.log(prob_for_each)

		return cross_entropy - label_entropy

	def entropy_label_space_specific(self, real_prob, adjusted_prob, sum_log_prob, prob_for_each):
		return adjusted_prob * math.log(adjusted_prob)

	def entropy_object_space_specific(self, real_prob, adjusted_prob, sum_log_prob, prob_for_each):
		return adjusted_prob * math.log(prob_each)

	def combined_cost_specific(self, real_prob, adjusted_prob, sum_log_prob, prob_for_each):
		return self.kl_p_from_q_specific(real_prob, adjusted_prob, sum_log_prob, prob_for_each) + \
			   self.entropy_label_space_specific(real_prob, adjusted_prob, sum_log_prob, prob_for_each)

	def cost(self, clusterset):
		all_in_clusters = reduce(lambda x, y: x+y, [elements_below[c] for c in clusterset])
		epsilon_set = [o for o in range(1000) if o not in all_in_clusters]

		prob_clusts = sum([self.posteriors[c] for c in clusterset])
		prob_epsilon = 1 - prob_clusts

		n_epsilon = len(epsilon_set)


		if n_epsilon > 0:
			logsum_epsilon = sum(self.logprob_all_below[epsilon_set])
			prob_each_epsilon = self.epsilon / float(n_epsilon)

			multiplier = (1.0 - self.epsilon) / (1 - prob_epsilon)
			epsilon_cost = self.specific_value_func(self, prob_epsilon, self.epsilon, logsum_epsilon, prob_each_epsilon)

		else:
			epsilon_cost = 0
			multiplier = 1.0

		#we want to minimize the cost
		total_cost = 0
		total_cost += epsilon_cost

		for c in clusterset:
			real_prob = self.posteriors[c]
			adjusted_prob = real_prob * multiplier
			sum_log_prob = self.logprob_all_below[c]
			prob_for_each = adjusted_prob / float(self.nbelow[c])

			cost = self.specific_value_func(self, real_prob, adjusted_prob, sum_log_prob, prob_for_each)
			total_cost += cost


		return total_cost
	
	def greedy_refine(self, current_set):
		
		potential_sets = []

		all_in_clusters = reduce(lambda x, y: x+y, [elements_below[c] for c in current_set])
		epsilon_set = [o for o in range(1000) if o not in all_in_clusters]

		logsum_epsilon = sum(self.logprob_all_below[epsilon_set])

		prob_clusts = sum([self.posteriors[c] for c in current_set])
		prob_epsilon = 1 - prob_clusts

		n_epsilon = len(epsilon_set)

		#cost associated with putting each cluster in the trash bag
		cost_dispose = []
		for c in current_set:
			
			prob_epsilon_new = prob_epsilon + self.posteriors[c]
			logsum_new = logsum_epsilon + self.logprob_all_below[c]
			prob_each_epsilon = self.epsilon / (n_epsilon + self.nbelow[c])

			multiplier = (1.0 - self.epsilon) / (1 - prob_epsilon_new)

			epsilon_cost = self.specific_value_func(self, prob_epsilon_new, self.epsilon, logsum_new, prob_each_epsilon)
			#NOTE: DOES NOT TAKE INTO ACCOUNT HOW THE CHANGING MULTIPLIER EFFECTS STUFF
			cost_dispose.append([epsilon_cost, c, multiplier])

		cost_dispose = sorted(cost_dispose, key = lambda x: x[0])

		for c in current_set:
			if self.is_leaf(c): continue

			left, right = rules[c]

			#take the left child instead
			s = current_set[:]
			s.remove(c)
			s.append(left)
			potential_sets.append(sorted(s))

			#take the right child instead
			s = current_set[:]
			s.remove(c)
			s.append(right)
			potential_sets.append(sorted(s))

			s = current_set[:]
			s.remove(c)
			s.append(left)
			s.append(right)
			s.remove(cost_dispose[0][1] if cost_dispose[0][1] != c else cost_dispose[1][1])
			potential_sets.append(sorted(s))

			'''
			#take both the left and right child, get rid of something else
			for c2 in current_set:
				if c2 == c: continue
				s = current_set[:]
				s.remove(c)
				s.append(left)
				s.append(right)
				s.remove(c2)
				potential_sets.append(sorted(s))
			'''

		potential_sets = [[[], 10 ** 10]] + [[s, self.cost(s)] for s in potential_sets]	


		min_set, min_cost = min(potential_sets, key = lambda x: x[1])
		if self.verbose: print min_set, min_cost, self.cost(current_set)
		
		if min_cost > self.cost(current_set): return current_set
		else: return self.greedy_refine(min_set)

	def greedy_optimize(self, current_set):
		temp_player = VariationalPlayer2(self.nclusts, method = 'greedy', value_func = self.value_func, verbose = False)
		temp_player.knowledge = self.knowledge
		temp_player.update_all()
		return self.greedy_refine(sorted(temp_player.clusterset))


	def posterior_from_clustering(self, posteriors):
		probs = np.zeros(1000)

		for c in self.clusterset:
			ebelow = elements_below[c]
			probs[ebelow] = \
					self.multiplier * posteriors[c] / self.nbelow[c]

		if len(self.remaining_elements) > 0: probs[self.remaining_elements] = self.epsilon / len(self.remaining_elements) #for the extra cluster
		return probs



	def update_all(self):
		super(VariationalPlayer2, self).update_all()

		self.perform_recursive_value_calcs()

		self.clusterset = self.find_clustering()

		all_clust_elements = reduce(lambda x, y: x+y, [elements_below[c] for c in self.clusterset])
		self.remaining_elements = np.array([o for o in range(1000) if o not in all_clust_elements])
		self.empty_remaining = (len(self.remaining_elements) == 0)
		self.tot_sum = sum([self.posteriors[c] for c in self.clusterset])
		self.multiplier = 1.0 if self.empty_remaining else (1.0 - self.epsilon) / self.tot_sum

		if self.verbose:
			print [[features[f], a] for f, a in self.knowledge]
			for c in self.clusterset:
				ebelow = elements_below[c]
				print len(ebelow), self.multiplier * self.posteriors[c], self.posteriors[c], [items[e] for e in ebelow]

			if not self.empty_remaining: print len(self.remaining_elements), self.epsilon, 1 - self.tot_sum, [items[e] for e in self.remaining_elements]
			print '\n\n'

		self.full_posterior = self.probabilities
		self.probabilities = self.posterior_from_clustering(self.posteriors)	

		self.clustering_cost = self.value_func_all(self, self.full_posterior, self.probabilities, self.clusterset)

		self.entropy = entropy(self.probabilities)

class EpsilonAnd2Player(EpsilonAndNPlayer):
	def __init__(self, epsilon = 0.0001, method = 'greedy', value_func = None, verbose = False):
		super(EpsilonAnd2Player, self).__init__(2, method = method, epsilon = epsilon, verbose = verbose, value_func = value_func)

	def greedy_optimize(self, current_set):
		best_choice = None
		best_cost = 10 ** 10

		p = mp.ProcessingPool()
		costs = p.map(self.cost, viable_pairs)
		p.close()

		best_choice = viable_pairs[np.argmin(costs)]
		return best_choice
		
if __name__ == '__main__':
	#player = EpsilonAnd2Player(value_func = VariationalPlayer2.combined_cost, verbose = True)
	#player.play_game()

	player = EpsilonAndNPlayer(2, value_func = VariationalPlayer2.entropy_label_space, verbose = True)
	player.play_game()

