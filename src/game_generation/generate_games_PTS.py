from runner_pts import *
from runner_variational2 import *
import scipy.stats
import random
import cPickle as pickleg

cur_used = []
cur_used_options = []
items_used = []

def get_options_without_used(argsorted, knowledge):
	features_used = [f for f, a in knowledge]
	argsorted_parsed = [e for e in argsorted if e not in cur_used and e not in features_used]
	r = 0
	indeces = [0 + random.randint(0, r)]
	indeces += [(len(argsorted_parsed)/6) * x + random.randint(-r, r) for x in range(1, 5)]
	indeces += [len(argsorted_parsed) + random.randint(-r, 0) - 1]
	indeces = np.array(indeces, np.uint8)

	return np.array(argsorted_parsed)[indeces]

def add_final(game):
	global cur_used
	global cur_used_options
	global items_used
	cur_used += [k[0] for k in game["knowledge"]]
	cur_used += list(game["current_options"])
	cur_used_options += list(game["current_options"])

	items_used.append(game["item"])

def parse_bag(bag):
	new_bag = []
	for b in bag:
		if b["item"] in items_used:
			continue
		#if np.any([f in cur_used_options for f, a in b["knowledge"]]):
		#	continue


		copy_dict = b.copy()
		copy_dict["current_options"] = get_options_without_used(copy_dict["bayes_argsort_mostfirst"], copy_dict["knowledge"]) \
											if copy_dict["bayes_generate"] else \
											get_options_without_used(copy_dict["pts_argsort_mostfirst"], copy_dict["knowledge"])

		indeces_bayes = [list(copy_dict["bayes_argsort_mostfirst"]).index(e) for e in copy_dict["current_options"]]
		indeces_pts =   [list(copy_dict["pts_argsort_mostfirst"]).index(e)   for e in copy_dict["current_options"]]

		indeces_bayes = np.array(indeces_bayes)
		indeces_pts = np.array(indeces_pts)

		indeces_bayes[np.argsort(indeces_bayes)] = np.arange(6)
		indeces_pts[np.argsort(indeces_pts)] = np.arange(6)

		copy_dict["correlation"] = scipy.stats.spearmanr ( indeces_bayes, indeces_pts)[0]

		new_bag.append(copy_dict)

	return new_bag
'''
def make_bag(n, depth, use_bayesian):
	global num_bags
	bag = []
	for i in range(n):
		item = random.randint(0, 999)
		game_dict = {"bayes_generate": use_bayesian, "item": item, "knowledge":[], "current_options":[], "bayes_argsort_mostfirst": [], "pts_argsort_mostfirst":[]}		
		

		for j in range(depth):
			player_bayes = ClustPlayer(9)
			player_pts = PositiveBiasModel(9)

			player_bayes.knowledge = game_dict["knowledge"]
			player_pts.knowledge = game_dict["knowledge"]

			if use_bayesian: player_bayes.update_all()
			else: player_pts.update_all()

			bayes_argsorted = np.argsort( (player_bayes if use_bayesian else player_pts).expected_gains())[::-1]
			index = random.randint(0, 5)
			q = bayes_argsorted[index]
			a = data_matrix[item, q]
			k = [(q, a)]
			game_dict["knowledge"] += k




		player_bayes = ClustPlayer(9)
		player_pts = PositiveBiasModel(9)

		player_bayes.knowledge = game_dict["knowledge"]
		player_pts.knowledge = game_dict["knowledge"]

		player_bayes.update_all()
		player_pts.update_all()

		game_dict["bayes_argsort_mostfirst"] = np.argsort(player_bayes.expected_gains())[::-1]
		game_dict["pts_argsort_mostfirst"] = np.argsort(player_pts.expected_gains())[::-1]
		
		bag.append(game_dict)
		num_bags += 1
		print "Bag {} completed".format(num_bags)

	return parse_bag(bag)
'''

def get_player_bayes():
	return OptimalPlayer()

def get_player_alternative():
	return VariationalPlayer2(10)

def gen_game_carefully(tup):
	global num_bags
	depth = tup[0]
	use_bayesian = tup[1]
	item = random.randint(0, 999)
	game_dict = {"bayes_generate": use_bayesian, "item": item, "knowledge":[], "current_options":[], "bayes_argsort_mostfirst": [], "pts_argsort_mostfirst":[]}		
	
	for j in range(depth):
		'''
		player_bayes = ClustPlayer(9)
		player_pts = PositiveBiasModel(9)
		'''
		player_bayes = get_player_bayes()
		player_pts = get_player_alternative()

		player_bayes.knowledge = game_dict["knowledge"]
		player_pts.knowledge = game_dict["knowledge"]

		if use_bayesian: player_bayes.update_all()
		else: player_pts.update_all()

		bayes_argsorted = np.argsort( (player_bayes if use_bayesian else player_pts).expected_gains())[::-1]
		index = random.randint(0, 0)
		available = [q for q in bayes_argsorted if not q in cur_used_options]
		q = available[index]
		a = data_matrix[item, q]
		k = [(q, a)]
		game_dict["knowledge"] += k


	player_bayes = get_player_bayes()
	player_pts = get_player_alternative()

	player_bayes.knowledge = game_dict["knowledge"]
	player_pts.knowledge = game_dict["knowledge"]

	player_bayes.update_all()
	player_pts.update_all()

	game_dict["bayes_argsort_mostfirst"] = np.argsort(player_bayes.expected_gains())[::-1]
	game_dict["pts_argsort_mostfirst"] = np.argsort(player_pts.expected_gains())[::-1]
	
	num_bags += 1
	print "Bag {} completed".format(num_bags)

	return game_dict
	


def make_bag_carefully(n, depth, use_bayesian):
	p = Pool()
	bag = p.map(gen_game_carefully, [(depth, use_bayesian)] * n)
	p.close()
	return parse_bag(bag)

def print_game(g):
	print "Bayes Generated:", g["bayes_generate"]
	print "Item:", items[g["item"]]
	print "Knowledge:", [(features[f], a) for f, a in g["knowledge"]]

	sorted_bayesian = sorted(g["current_options"], key = lambda x: list(g["bayes_argsort_mostfirst"]).index(x))
	sorted_pts = sorted(g["current_options"], key = lambda x: list(g["pts_argsort_mostfirst"]).index(x))

	print "Question options ranked bayesian:", [features[f] for f in sorted_bayesian]
	print "Question options ranked pts", [features[f] for f in sorted_pts]
	print "Correlation:", g["correlation"]


def make_bag_multiarg(tup):
	return make_bag(*tup)

'''
depths = [2, 4, 6]
num_bags = 0
bag = []
for i in depths:
 	bag += make_bag(50, i, True)
 	bag += make_bag(50, i, False)
print len(bag)
final_bag = []

trials_to_pick = [(d, is_bayes) for is_bayes in [True, False] for i in range(3) for d in depths]
random.shuffle(trials_to_pick)
print trials_to_pick


for t in trials_to_pick:
	bag = sorted(bag, key = lambda x: x["correlation"] - \
							(1000 if len(x["knowledge"]) == t[0] else 0) - \
							(1000 if x["bayes_generate"] == t[1] else 0) )
	final_bag.append(bag[0].copy())
	add_final(bag[0])
	bag = parse_bag(bag)
	print "Completed trial {}".format(str(t))

for g in final_bag:
	print "\n\n****************\n\n"
	print_game(g)

with open(base_path+"data/experiment2_trials.pickle", 'w') as trial_file:
	pickle.dump(final_bag, trial_file)

'''


num_bags = 0
depths = [2, 4, 6]
final_bag = []

trials_to_pick = [(0, True), (0, False)] + [(d, is_bayes) for is_bayes in [True, False] for i in range(2) for d in depths]
random.shuffle(trials_to_pick)



for d, is_bayes in trials_to_pick:
	options = make_bag_carefully(100, d, is_bayes)
	bag = sorted(options, key = lambda x: x["correlation"] - \
							(1000 if len(x["knowledge"]) == d else 0) - \
							(1000 if x["bayes_generate"] == is_bayes else 0) )
	final_bag.append(bag[0].copy())
	add_final(bag[0])
	
	print "\nCompleted trial {}, {}".format(d, is_bayes)
	print_game(bag[0])
	print "\n*******************\n"

	bag = parse_bag(bag)


for g in final_bag:
	print "\n\n****************\n\n"
	print_game(g)

with open(base_path+"data/experiment2_trials.pickle", 'w') as trial_file:
	pickle.dump(final_bag, trial_file)
