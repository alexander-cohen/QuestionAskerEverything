from runner_pts import *
import scipy.stats as scistats
from matplotlib import pyplot as plt
import cPickle as pickle
from matplotlib import rc
import random
import copy

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
fsize = 8

with open(base_path+"data/experiment2/oneshots_by_people.pickle", 'r') as f:
	oneshots_by_people = pickle.load(f)

with open(base_path+"data/experiment2/oneshots_by_trial.pickle", 'r') as f:
	oneshots_by_trial = pickle.load(f)

with open(base_path+"data/experiment2_trials.pickle") as f:
	all_trials_fulldata = pickle.load(f)[1:]


for oneshots in oneshots_by_trial:
	rc = sorted(oneshots[0]['ranked_choices'])
	trial = None
	for t in all_trials_fulldata:
		if sorted(t['current_options']) == rc:
			trial = t
			break

	for o in oneshots:
		o['eig_generated'] = trial['bayes_generate']

for oneshots in oneshots_by_people:
	for o in oneshots:
		rc = sorted(o['ranked_choices'])
		trial = None
		for t in all_trials_fulldata:
			if sorted(t['current_options']) == rc:
				trial = t
				break

		o['eig_generated'] = trial['bayes_generate']

def norm_sum(vec):
	npvec = np.array(vec)
	return npvec / np.sum(npvec)

def norm_01(vec):
	min0 = np.array(vec)
	min0 = min0 - np.min(min0)
	return min0 / np.max(min0)


def get_model_rankings(oneshot):
	qa_pairs = oneshot['question_answer_pairs']
	question_choices = oneshot['ranked_choices']

	eig_model = ClustPlayer(9)
	eig_model.knowledge = qa_pairs
	eig_model.update_all()
	
	pts_model = PositiveBiasModel(9)
	pts_model.knowledge = qa_pairs
	pts_model.update_all()

	eig_rankings = [eig_model.expected_gain(f) for f in question_choices]
	pts_rankings = [pts_model.expected_gain(f) for f in question_choices]

	return eig_rankings, pts_rankings

def get_human_rankings(oneshots):
	human_rankings = np.array( [o['rankings'] for o in oneshots] )
	human_rankings = np.array(human_rankings)

	human_avg_rankings = 5 - np.average(human_rankings, axis = 0)

	return human_avg_rankings

def get_rankings(oneshots):
	human_avg_rankings = get_human_rankings(oneshots)
	eig_rankings, pts_rankings = get_model_rankings(oneshots[0])

	return human_avg_rankings, eig_rankings, pts_rankings

def get_correlations(human_avg_rankings, eig_rankings, pts_rankings):
	eig_cor_pearson = scistats.pearsonr(human_avg_rankings, eig_rankings)[0]
	eig_cor_spearman = scistats.spearmanr(human_avg_rankings, eig_rankings)[0]

	pts_cor_pearson = scistats.pearsonr(human_avg_rankings, pts_rankings)[0]
	pts_cor_spearman = scistats.spearmanr(human_avg_rankings, pts_rankings)[0]

	return eig_cor_pearson, eig_cor_spearman, pts_cor_pearson, pts_cor_spearman


def analyze_and_print_all_trials(oneshots_by_trial):
	#outer level: generation
	#inner level: model

	all_correlations = {'eig': {}, 'pts': {}}

	for k in all_correlations:
		all_correlations[k]['eig'] = {}
		all_correlations[k]['pts'] = {}


	for k in all_correlations:
		for k_inner in all_correlations[k]:
			all_correlations[k][k_inner] = {'pearson': [], 'spearman': []}


	for oneshots, i in zip(oneshots_by_trial, range(1000)):
		human_avg_rankings, eig_rankings, pts_rankings = get_rankings(oneshots)
		eig_cor_pearson, eig_cor_spearman, pts_cor_pearson, pts_cor_spearman = \
			get_correlations(human_avg_rankings, eig_rankings, pts_rankings)
		
		qa_pairs = oneshots[0]['question_answer_pairs']
		question_choices = oneshots[0]['ranked_choices']

		human_avg_rankings = norm_sum(human_avg_rankings)
		eig_rankings = norm_sum(eig_rankings)
		pts_rankings = norm_sum(pts_rankings)


		eig_generated = oneshots[0]['eig_generated']
		relevent_dict = all_correlations['eig' if eig_generated else 'pts']
		
		relevent_dict['eig']['pearson'].append(eig_cor_pearson)
		relevent_dict['eig']['spearman'].append(eig_cor_spearman)
		relevent_dict['pts']['pearson'].append(pts_cor_pearson)
		relevent_dict['pts']['spearman'].append(pts_cor_spearman)

		'''
		all_pearson_eig.append(eig_cor_pearson)
		all_spearman_eig.append(eig_cor_spearman)
		all_pearson_pts.append(pts_cor_pearson)
		all_spearman_pts.append(pts_cor_spearman)
		'''
		
		item = oneshots[0]['item']

		print "Trial #{}".format(i+1)
		print "\nItem:", items[item]
		print "\nSetup:"	
		for q, a in qa_pairs:
			print features[q], a


		top = "Question:" + (" "*40) + "Human:" + (" "*6) + "EIG:" + (" "*6) + "PTS:"

		l1 = top.index('Human:') - top.index("Question:")
		l2 = top.index('EIG:') - top.index("Human:")
		l3 = top.index('PTS:') - top.index("EIG:")

		print "\nResults:"
		print top
		for q, hrank, erank, prank in zip(question_choices, human_avg_rankings, eig_rankings, pts_rankings):
			question_part = features[q].ljust(l1)
			human_rank = "{:0.2f}".format(hrank).ljust(l2)
			eig_rank = "{:0.2f}".format(erank).ljust(l3)
			pts_rank = "{:0.2f}".format(prank)
			print question_part + human_rank + eig_rank + pts_rank

		'''
		print ['{:0.2f}'.format(n) for n in list(human_avg_rankings)]
		print ['{:0.2f}'.format(n) for n in list(eig_rankings)]
		print ['{:0.2f}'.format(n) for n in list(pts_rankings)]
		'''
		print "\nPearson  correlation with eig: {:0.2f}, with PTS: {:0.2f}".format(eig_cor_pearson, pts_cor_pearson)
		print   "Spearman correlation with eig: {:0.2f}, with PTS: {:0.2f}".format(eig_cor_spearman, pts_cor_spearman)

		print '\n*****************\n'

	average_correlations = copy.deepcopy(all_correlations)
	for gen_type in average_correlations:
		for cor_model in average_correlations[gen_type]:
			print ''
			for cor_type in average_correlations[gen_type][cor_model]:
				cors = average_correlations[gen_type][cor_model][cor_type]
				average_correlations[gen_type][cor_model][cor_type] = sum(cors) / float(len(cors))
				print "Average correlation for gen type: {:<3}, model type: {:<3}, cor type: {:<8} = {:0.3f}".\
					format(gen_type, cor_model, cor_type, average_correlations[gen_type][cor_model][cor_type])
	
	#print average_correlations
	return all_correlations, average_correlations

		#print "Average pearson  eig: {:0.2f}, with PTS: {:0.2f}".format(avg_pearson_eig, avg_pearson_pts)
		#print "Average spearman eig: {:0.2f}, with PTS: {:0.2f}".format(avg_spearman_eig, avg_spearman_pts)



def make_graph(oneshots, n = "", lettered = False):
	human_avg_rankings, eig_rankings, pts_rankings = get_rankings(oneshots)
	eig_cor_pearson, eig_cor_spearman, pts_cor_pearson, pts_cor_spearman = \
		get_correlations(human_avg_rankings, eig_rankings, pts_rankings)

	f, axarr = plt.subplots(1, 1)
	f.set_size_inches(5, 5, dpi=5000, forward=True)
	f.tight_layout(pad = 5)

	axarr.set_xlim([-0.2, 1.2])
	axarr.set_ylim([-0.2, 5.2])

	eig_rankings_01 = norm_01(eig_rankings)
	pts_rankings_01 = norm_01(pts_rankings)

	print human_avg_rankings
	print eig_rankings, eig_rankings_01
	print pts_rankings, pts_rankings_01

	order_eig = np.argsort(eig_rankings_01)[::-1]
	order_pts = np.argsort(pts_rankings_01)[::-1]
	order_human = np.argsort(human_avg_rankings)[::-1]

	eig_line = axarr.plot(eig_rankings_01[order_eig], human_avg_rankings[order_eig], color = 'r', label = "EIG")
	pts_line = axarr.plot(pts_rankings_01[order_pts], human_avg_rankings[order_pts], color = 'b', label = "PTS")


	if lettered:
		for h, e, p, letter in zip(human_avg_rankings, eig_rankings_01, pts_rankings_01, ['A', 'B', 'C', 'D', 'E', 'F']):
			axarr.text(e, h, letter, horizontalalignment = 'center', verticalalignment = 'center', fontsize = fsize)
			axarr.text(p, h, letter, horizontalalignment = 'center', verticalalignment = 'center', fontsize = fsize)


	else:
		eig_scatter = axarr.scatter(eig_rankings_01[order_eig], human_avg_rankings[order_eig], color = 'r')
		pts_scatter = axarr.scatter(pts_rankings_01[order_pts], human_avg_rankings[order_pts], color = 'b')

	axarr.legend(["EIG", "PTS"], fontsize = fsize)

	title = "EIG: $r={:0.2f}$, $\\rho={:0.2f}$".format(eig_cor_pearson, eig_cor_spearman)
	title += "\nPTS: $r={:0.2f}$, $\\rho={:0.2f}$".format(pts_cor_pearson, pts_cor_spearman)

	axarr.set_title(title, fontsize = fsize)
	
	f.savefig("new_plots_exp2/{}/plot_{}_{}.pdf".format(n, "lettered" if lettered else "dotted", "lettered" if lettered else "dotted"))


def softmax(vec, choice, tao, norm = True):
	vec2 = vec / np.sum(vec) if norm else vec
	
	mult = vec2 * tao
	exp = np.exp(mult)
	return exp[choice] / np.sum(exp)

#returns gradient and value
def softmax_iter(vec, choice, tao, norm = True):
	vec2 = vec / np.sum(vec) if norm else vec
	
	mult = vec2 * tao
	exp = np.exp(mult)

	f = exp[choice]
	g = np.sum(exp)

	softmax_val = f / g

	derivative_f = vec2[choice] * exp[choice]
	derivative_g = np.sum( vec2 * exp )
	derivative = (derivative_f * g - f * derivative_g) / (g ** 2) #w.r.t tao

	return softmax_val, derivative


#returns the gradient of theta, tao_eig, tao_pts
def mixture_model_iter(person, theta, tao_eig, tao_pts, tao_human, norm_cost = True):
	theta_grad = 0
	tao_eig_grad = 0
	tao_pts_grad = 0
	logprob = 0

	for oneshot in person:
		eig_rankings, pts_rankings = get_model_rankings(oneshot)
		human_rank = oneshot['rankings']


		eig_softmax, tao_eig_grad_part = softmax_iter(eig_rankings, )

'''
* four parameters: theta, tao_eig, tao_pts, tao_human
* theta     : probability to use full eig model
* tao_eig   : heat parameter for eig softmax
* tao_pts   : heat parameter for pts softmax
* tao_human : heat parameter for humans
* norm_cost : norms cost of question options before computing probs with softmax
'''
def mixture_model_analysis(person, norm_cost = True):
	pass
'''
for i in range(14):
	make_graph(oneshots_by_trial[i], i, lettered = True)
'''
analyze_and_print_all_trials(oneshots_by_trial)