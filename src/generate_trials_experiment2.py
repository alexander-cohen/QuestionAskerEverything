from data_analyzer import *
import random


with open("../data/centralities.pickle", 'r') as centralities_file:
	centralities = pickle.load(centralities_file)

with open("../data/ordered_clusters.pickle", 'r') as centralities_file:
	ordered_clusters = pickle.load(centralities_file)

fullgame_items = [items.index(f[0]) for f in fullgames]
sorted_items = sorted(fullgame_items, key = lambda x: centralities[x])

n_per_square = len(ordered_clusters)

central_fullgames_items = []
distant_fullgames_items = []

for c in ordered_clusters:
	in_clust = [f for f in fullgame_items if f in c]
	in_clust = sorted(in_clust, key = lambda x: c.index(x))
	print [items[e] for e in in_clust]
	central_fullgames_items.append(in_clust[0])
	distant_fullgames_items.append(in_clust[-1])


central_fullgames = [f for f in fullgames if f[1] in central_fullgames_items]
distant_fullgames = [f for f in fullgames if f[1] in distant_fullgames_items]

central_fullgames_trials = []
distant_fullgames_trials = []

all_used = []


def get_trial_from_fullgame(fullgame):
	item = fullgame[1]
	questions_and_answers = [(int(q['Choice_int']), int(q['Resp'])) for q in fullgame[2]]
	return [item, questions_and_answers]

def get_trial_from_model(item, length):
	itm = items[item]
	model = ClustPlayer(9)
	k = []
	for i in range(length):
		feat, _ = model.computer_iterate(itm)
		k.append((feat, data_matrix[item, feat]))

	
	return [item, k]

def print_trial(trial):
	print "Type:", trial[0]
	print "Item:", items[trial[1]]
	print "\nKnowledge:"

	for q, i in zip(trial[2], range(1000)):
		print "{}).".format(i), features[q[0]], q[1]

	print "\nOptions:"
	for q, i in zip(trial[3], range(1000)):
		print "{}).".format(i), features[q]

central_fullgames_trials = [get_trial_from_fullgame(f) for f in central_fullgames]
distant_fullgames_trials = [get_trial_from_fullgame(f) for f in distant_fullgames]

central_fullgames_trials = sorted(central_fullgames_trials, key = lambda x: len(x[1]))
distant_fullgames_trials = sorted(distant_fullgames_trials, key = lambda x: len(x[1]))

for l, i in zip([2, 2, 4, 4, 6, 6], range(1000)):
	central_fullgames_trials[i] = [central_fullgames_trials[i][0], central_fullgames_trials[i][1][:l]]
	distant_fullgames_trials[i] = [distant_fullgames_trials[i][0], distant_fullgames_trials[i][1][:l]]

#print [len(e[1]) for e in central_fullgames_trials]
#print [len(e[1]) for e in distant_fullgames_trials]



central_computer_trials = [get_trial_from_model(t[0], len(t[1])) for t in central_fullgames_trials]
distant_computer_trials = [get_trial_from_model(t[0], len(t[1])) for t in distant_fullgames_trials]


all_trials = []
all_trials += [['central_fullgame'] + t for t in central_fullgames_trials]
all_trials += [['distant_fullgame'] + t for t in distant_fullgames_trials]
all_trials += [['central_computer'] + t for t in central_computer_trials]
all_trials += [['distant_computer'] + t for t in distant_computer_trials]

needed_questions = [(t[2], n, i) for n in range(6) for t, i in zip(all_trials, range(1000))]
found_questions = [0 for i in range(len(needed_questions))]

shuffled = range(len(all_trials))


set1 = shuffled[:len(shuffled)/2]
set2 = shuffled[len(shuffled)/2:]

needed_questions_set1 = [q for q in needed_questions if q[2] in set1]
needed_questions_set2 = [q for q in needed_questions if q[2] in set2]

set1_trials = [all_trials[i] for i in set1]
set2_trials = [all_trials[i] for i in set2]

used_questions_overall = []
for t in all_trials:
	#print t
	used_questions_overall += [e[0] for e in t[2]]

#print '\n', used_questions_overall, '\n\n\n'

used_questions_set1 = []
used_questions_set2 = []

while len(needed_questions_set1) > 0:
	choice = random.choice(needed_questions_set1)

	trial = choice[0]
	num_needed = choice[1]

	index = needed_questions.index(choice)

	needed_questions_set1.remove(choice)

	model = ClustPlayer(9)
	model.knowledge = trial
	model.update_all()
	eigs = model.expected_gains()
	argsorted = np.argsort(eigs)[::-1]


	#pick the open spot nearest to our target spot
	target = (len(features) / 6) * num_needed
	open_options = np.array([argsorted[f] for f in range(len(features)) if f not in used_questions_overall and f not in used_questions_set1])
	dif = np.abs ( open_options - target )
	best = np.argmin(dif)
	feat = list(argsorted).index(best)

	found_questions[index] = feat
	used_questions_set1.append(feat)



while len(needed_questions_set2) > 0:
	choice = random.choice(needed_questions_set2)
	trial = choice[0]
	num_needed = choice[1]
	index = needed_questions.index(choice)

	needed_questions_set2.remove(choice)
	model = ClustPlayer(9)
	model.knowledge = trial
	model.update_all()
	eigs = model.expected_gains()
	argsorted = np.argsort(eigs)[::-1]


	#pick the open spot nearest to our target spot
	target = (len(features) / 6) * num_needed
	open_options = np.array([argsorted[f] for f in range(len(features)) if f not in used_questions_overall and f not in used_questions_set2])
	dif = np.abs ( open_options - target )
	best = np.argmin(dif)
	feat = list(argsorted).index(best)

	found_questions[index] = feat
	used_questions_set2.append(feat)
	


'''
print set1
print set2
print used_questions_overall
print used_questions_set1
print used_questions_set2
print found_questions
print all_trials
'''
seen_so_far = 0
for t in all_trials:
	t += [[ found_questions[seen_so_far], \
			found_questions[seen_so_far+1], \
			found_questions[seen_so_far+2], \
			found_questions[seen_so_far+3], \
			found_questions[seen_so_far+4], \
			found_questions[seen_so_far+5] ] ]

	seen_so_far += 6

#print seen_so_far

for t in all_trials:
	print "***********"
	print_trial(t)

print "Central items:", [items[e] for e in central_fullgames_items]
print "Distant items:", [items[e] for e in distant_fullgames_items]