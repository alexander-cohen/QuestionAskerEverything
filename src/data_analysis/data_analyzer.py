from import_all_files import *

import cPickle as pickle
from runner_randomN import *
#from runner_goodn import GoodN
from runner_expected_utility import ExpectedUtilityPlayer
import scipy.stats as scistats
from multiprocessing import Pool
import os
import time

from runner_pts import *
from runner_maxprob import *
from runner_gini import *

from runner_variational2 import *
from runner_epsilon_and_n import *

from matplotlib import pyplot as plt
from matplotlib import rc
import itertools

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)


with open(base_path+"src/analysis_files/datalogs/fullgamedata.pickle", 'r') as fullgamedata:
	fullgames = pickle.load(fullgamedata)

with open(base_path+"src/analysis_files/datalogs/oneshotdata.pickle", 'r') as oneshotdata:
	oneshots = pickle.load(oneshotdata)

with open(base_path+"src/analysis_files/datalogs/peopledata.pickle", 'r') as peopledata:
	people = pickle.load(peopledata)

with open(base_path+"src/analysis_files/datalogs/fullgamedata_correctorder.pickle", 'r') as fullgamedata:
	fullgames_correctorder = pickle.load(fullgamedata)

with open(base_path+"src/analysis_files/datalogs/oneshotdata_correctorder.pickle", 'r') as oneshotdata:
	oneshots_correctorder = pickle.load(oneshotdata)

with open(base_path+"src/analysis_files/datalogs/peopledata_correctorder.pickle", 'r') as peopledata:
	people_correctorder = pickle.load(peopledata)

should_strict_only_consider = True
questions_in_full_game = 9

def makedir(pathname):
	if not os.path.exists(os.path.dirname(pathname)):
	    try:
	        os.makedirs(os.path.dirname(pathname))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	            raise


def normalize(a):
	row_sums = a.sum(axis=1)
	return a / row_sums[:, np.newaxis]

def getNumSeenBeforePerTrial(persondata):
	their_oneshots = persondata["oneshots"]
	their_fullgames = persondata["fullgames"]

	seen_before = np.zeros(len(features))
	seen_before_pertrial = []

	for f in their_fullgames:
		for q in f[2]:
			seen = q["QuestionOptions_int"]
			seen_before[seen] += 1

	for o in their_oneshots:
		seen_prev = [k[0] for k in o["Knowledge"]]
		seen_before[seen_prev] += 1
		seen_before_pertrial.append(np.copy(seen_before))
		seen_after = o["Questions_int"]
		seen_before[seen_after] += 1

	#print [l[78] for l in seen_before_pertrial], '\n\n'
	return seen_before_pertrial

def correlationOfPeopleWithEIG_times_seen_method(persondata):
	numSeenPerTrial = getNumSeenBeforePerTrial(persondata)
	their_oneshots = persondata["oneshots"]
	allpearson = []
	allspearman = []
	for t, numSeenInTrial in zip(their_oneshots, numSeenPerTrial):
		model = ClustPlayer(9)
		weight_matrix = np.zeros( (len(t["Questions_int"])) )

		model.knowledge = t["Knowledge"]
		model.update_all()

		for j in range(len(t["Questions_int"])):
			weight_matrix[j] = model.expected_gain(t["Questions_int"][j])

		rankorder = t["Rankorder"]
		rankorder = [5-n for n in rankorder]

		try:
			pearson = scistats.pearsonr(weight_matrix, rankorder)[0]
			spearman = scistats.spearmanr(weight_matrix, rankorder)[0]

		except:
			pearson = 0
			spearman = 0

		if math.isnan(pearson): pearson = 0
		if math.isnan(spearman): spearman = 0

		allpearson.append(pearson)
		allspearman.append(spearman)

	return allpearson, allspearman

def correlationOfSeenBeforeWithEIG(persondata):
	numSeenPerTrial = getNumSeenBeforePerTrial(persondata)
	their_oneshots = persondata["oneshots"]
	allpearson = []
	allspearman = []
	for t, numSeenInTrial in zip(their_oneshots, numSeenPerTrial):
		model = ClustPlayer(9)
		weight_matrix = np.zeros( (len(t["Questions_int"])) )

		model.knowledge = t["Knowledge"]
		model.update_all()

		for j in range(len(t["Questions_int"])):
			weight_matrix[j] = model.expected_gain(t["Questions_int"][j])


		seen_ranks = [numSeenInTrial[indx] for indx in t["Questions_int"]]
		
		try:
			pearson = scistats.pearsonr(weight_matrix, seen_ranks)[0]
			spearman = scistats.spearmanr(weight_matrix, seen_ranks)[0]

		except:
			pearson = 0
			spearman = 0

		if math.isnan(pearson): pearson = 0
		if math.isnan(spearman): spearman = 0

		allpearson.append(pearson)
		allspearman.append(spearman)

	return allpearson, allspearman

def correlationOfChoicesWithSeenBefore(persondata):
	numSeenPerTrial = getNumSeenBeforePerTrial(persondata)
	their_oneshots = persondata["oneshots"]
	allpearson = []
	allspearman = []
	for t, numSeenInTrial, index in zip(their_oneshots, numSeenPerTrial, range(1000)):
		rankorder = t["Rankorder"]

		rankorder = [5-n for n in rankorder]
		seen_ranks = [numSeenInTrial[indx] for indx in t["Questions_int"]]
		


		if max(rankorder) != min(rankorder) and max(seen_ranks) != min(seen_ranks):
			pearson = scistats.pearsonr(rankorder, seen_ranks)[0]
			spearman = scistats.spearmanr(rankorder, seen_ranks)[0]


		else:
			pearson = 0
			spearman = 0

		if pearson == float("nan"): pearson = 0
		if spearman == float("nan"): spearman = 0

		if t['Depth'] == 0: 
			print rankorder, seen_ranks, pearson, spearman


		allpearson.append(pearson)
		allspearman.append(spearman)

	return allpearson, allspearman

def analyze_times_seen():
	allcors = []
	for p in people_correctorder:
		#allcors.append(correlationOfChoicesWithSeenBefore(p))
		allcors.append(correlationOfPeopleWithEIG_times_seen_method(p))

	overalltotpear = 0
	overalltotspear = 0
	overallnumtot = 0

	totpear_bydepth = np.zeros(4)
	totspear_bydepth = np.zeros(4)
	numtot_bydepth = np.zeros(4)

	for p, person_index in zip(allcors, range(1000)):
		print "\nPerson " + str(person_index) + ":"
		totpear = 0
		totspear = 0
		numtot = 0

		for i in range(len(p[0])):
			totpear += p[0][i]
			totspear += p[1][i]
			
			depth = len(people_correctorder[person_index]['oneshots'][i]['Knowledge'])
			print "Trial " + str(i) + " (pearson/spearman):", p[0][i], p[1][i], depth
			
			totpear_bydepth[depth/2] += p[0][i]
			totspear_bydepth[depth/2] += p[1][i]
			numtot_bydepth[depth/2] += 1
			
			numtot += 1.0



		overalltotpear += totpear
		overalltotspear += totspear
		overallnumtot += numtot

		
		

		print "Average pearson for person " + str(i) + ":", str(totpear/numtot)
		print "Average spearman for person " + str(i) + ":", str(totspear/numtot)

	print "\n\n"
	print "Pearson by depth", totpear_bydepth / numtot_bydepth
	print "Spearman by depth", totspear_bydepth / numtot_bydepth
	print "Overall Pearson / Num Pearson:", overalltotpear, overallnumtot
	print "Overall Spearman / Num Pearsom:", overalltotspear, overallnumtot
	print "Average pearson overall:", overalltotpear/overallnumtot
	print "Average spearman overall:", overalltotspear/overallnumtot

def get_person_scoring(trialnum):
	nresps = len(people[0]["oneshots"][0]["Questions_int"])
	person_matrx = np.zeros( (len(people), nresps)  )
	for p, i in zip(people, range(1000)):
		for j in range(nresps):
			person_matrx[i, j] = 5 - p["oneshots"][trialnum]["Rankorder"].index(j)

	return np.average(person_matrx, 0)

def perform_random_subset(numSimPeople = 25, numObjects = 20, good = False):
	#use person 0 for now, as it does not matter
	#good refers to whether or not the chosen objects are central

	person = people[2]
	ntrials = len(person["oneshots"])
	if not good: objectsChosen = [[random.sample(range(1000), numObjects) for i in range(numSimPeople)] for j in range(ntrials)]
	else: objectsChosen = []

	cors = np.zeros((ntrials, numSimPeople))
	pearsons = []
	spearmans = []
	standarderrors = []
	for t, i in zip(person["oneshots"], range(1000)):
		#print t["Depth"]
		if not good: simulatedPeople = [RandomN(9, numObjects, objectsChosen[i][x], strict_only_consider = should_strict_only_consider) for x in range(numSimPeople)]
		else: 
			simulatedPeople = [GoodN(9, numObjects, strict_only_consider = should_strict_only_consider) for x in range(numSimPeople)]
			objectsChosen.append([p.randomN for p in simulatedPeople])

		weight_matrix = np.zeros( (numSimPeople, len(t["Questions_int"])) )

		for s, personIndx in zip(simulatedPeople, range(1000)):
			s.knowledge = t["Knowledge"]
			#print s.knowledge
			s.update_all()
			for j in range(len(t["Questions_int"])):
				weight_matrix[personIndx, j] = s.expected_gain(t["Questions_int"][j])

		avgScore = np.average(weight_matrix, 0)
		humanScore = get_person_scoring(i)
		pearsons.append( scistats.pearsonr(avgScore, humanScore)[0] )
		spearmans.append( scistats.spearmanr(avgScore, humanScore)[0] )
		standarderrors.append( np.average(np.std(weight_matrix, 0)) )



	return objectsChosen, pearsons, spearmans, standarderrors

def perform_random_subset_tup(tup):
	val = perform_random_subset(*tup)
	print "iteration performed"
	return val

def perform_n_random_subsets(numiters, numSimPeople = 25, numObjects = 20, numTrials = 10, good = False):
	pearsonMat = np.zeros( (numiters, numTrials) )
	spearmanMat = np.zeros( (numiters, numTrials) )
	standarderrorsMat = np.zeros( (numiters, numTrials) )
	allObjectsChosen = []

	p = Pool()
	vals = p.map(perform_random_subset_tup, [(numSimPeople, numObjects, good)]*numiters)
	p.close()

	for elem, i in zip(vals, range(numiters)):
		oc, p, s, se = elem
		allObjectsChosen.append(oc)
		pearsonMat[i] = np.array(p)
		spearmanMat[i] = np.array(s)
		standarderrorsMat[i] = np.array(se)

	'''
	for i in range(numiters):
		print "iteration performed"
		oc, p, s, se = perform_random_subset(numSimPeople, numObjects)
		allObjectsChosen.append(oc)
		pearsonMat[i] = np.array(p)
		spearmanMat[i] = np.array(s)
		standarderrorsMat[i] = np.array(se)
	'''

	return allObjectsChosen, pearsonMat, spearmanMat, standarderrorsMat

def analyze_randomsubsets(numIters, numSimPeople, numObjects, log = True, printout = True, verbosePrint = False, verboseLog = True, numappend = 0, good = False):
	ac, pm, sm, sem = perform_n_random_subsets(numIters, numSimPeople, numObjects, good = good)

	datstr_concise = ""
	datstr_concise += str(numIters) + " trials ran. " + str(numSimPeople) + " simulated people per trial. " + str(numObjects) + " objects per simulated person.\n\n"
	datstr_concise += "Average pearson (by: trial number):\n" + str(np.average(pm, 0)) + "\n\n"
	datstr_concise += "Average spearman (by: trial number):\n" + str(np.average(sm, 0)) + "\n\n"
	datstr_concise += "Average standard error (by: trial number):\n" + str(np.average(sem, 0)) + "\n\n"

	datstr = datstr_concise
	datstr += "All objects (by: run iteration, oneshot trial, simulated person):\n" + str(ac) + "\n\n"
	datstr += "All pearson (by: run iteration, trial number):\n" + str(pm) + "\n\n"
	datstr += "All spearman (by: run iteration, trial number):\n" + str(sm) + "\n\n"
	datstr += "All standard error (by: run iteration, trial number):\n" + str(sem) + "\n\n"

	if log:
		filepath = "strict_only_consider" if should_strict_only_consider else "loose_only_consider"
		filepath += "-" + time.strftime("%m:%d:%Y-%H:%M:%S")
		fullpath = dir_for_this_run + filepath + "/"
		makedir(fullpath)

		with open(dir_for_this_run + filepath + "/random_subset" + str(numappend) + ".txt", 'w') as randomsubsetfile:
			randomsubsetfile.write(datstr if verboseLog else datstr_concise)

	if printout:
		print datstr if verbosePrint else datstr_concise

	return ac, pm, sm, sem, datstr, datstr_concise




def random_subsets_over_n(startSize = 5, endSize = 100, stepSize = 5, numItersPer = 50, numPeoplePer = 25, log = True, good = False):
	filepath = "strict_only_consider" if should_strict_only_consider else "loose_only_consider"
	filepath += "-" + time.strftime("%m:%d:%Y-%H:%M:%S")
	fullpath = dir_for_this_run + filepath + "/"
	if not os.path.exists(os.path.dirname(fullpath)):
	    try:
	        os.makedirs(os.path.dirname(fullpath))
	    except OSError as exc: # Guard against race condition
	        if exc.errno != errno.EEXIST:
	            raise

	allobjects = []
	pearsons = []
	spearmans = []
	standarderrors = []
	allstr = []
	'''
	paramlist = [(numItersPer, numPeoplePer, n, True, True, False, True, n) for n in range(startSize, endSize, stepSize)]
	p = Pool()
	allentities = p.map(analyze_randomsubsets_tup, paramlist)
	'''

	for n in range(startSize, endSize, stepSize):
		ac, pm, sm, sem, datstr, datstr_concise = analyze_randomsubsets(numItersPer, numPeoplePer, n, log = True, printout = True, verbosePrint = False, verboseLog = True, numappend = n, good = good)
		allstr.append(datstr)
		allobjects.append(ac)
		pearsons.append(np.average(pm))
		spearmans.append(np.average(sm))
		standarderrors.append(np.average(sem))



	if log:
		with open(dir_for_this_run + filepath + "random_subset_overn_strictconsider.txt", 'w') as thefile:
			thefile.write("All num objects in range from " + str(startSize) + " to " + str(endSize) + " with step size " + str(stepSize) + "\n\n")
			
			for dastr, i in zip(allstr, range(1000)):
				thefile.write("\n\n--------------\n")
				thefile.write("Data string #" + str(range(startSize, endSize, stepSize)[i]) + "\n")
				thefile.write(datstr)

			thefile.write("\n\n----------\n\n")

			thefile.write("All objects:\n" + str(allobjects) + "\n\n")
			thefile.write("All pearsons:\n" + str(pearsons) + "\n\n")
			thefile.write("All spearmans:\n" + str(spearmans) + "\n\n")
			thefile.write("All standard errors:\n" + str(standarderrors) + "\n\n")

def goodN_analysis(k):
	#use person 0 for now, as it does not matter
	person = people[0]
	ntrials = len(person["oneshots"])
	
	pearsons = []
	spearmans = []
	standarderrors = []
	for t, i in zip(person["oneshots"], range(1000)):
		#print t["Depth"]
		model = GoodN(k)
		weight_matrix = np.zeros( (len(t["Questions_int"])) )

		model.knowledge = t["Knowledge"]
		#print t, '\n\n'
		model.update_all()


		for j in range(len(t["Questions_int"])):
			weight_matrix[j] = model.expected_gain(t["Questions_int"][j])

		avgScore = weight_matrix
		humanScore = get_person_scoring(i)
		pearsons.append( scistats.pearsonr(avgScore, humanScore)[0] )
		spearmans.append( scistats.spearmanr(avgScore, humanScore)[0] )
		standarderrors.append( np.average(np.std(weight_matrix, 0)) )

	return pearsons, spearmans, standarderrors


def full_oneshot_data_analyses(model_creator = lambda: OptimalPlayer(), record_ce = False):
	#use person 0 for now, as it does not matter
	person = people[0]
	ntrials = len(person["oneshots"])
	
	pearsons = []
	spearmans = []
	standarderrors = []
	if record_ce: all_cross_entropy = []

	all_trials = zip(person["oneshots"], range(1000))
	all_trials = sorted(all_trials, key = lambda x: len(x[0]["Knowledge"]))

	for t, i in all_trials:
		#print t["Depth"]
		weight_matrix = np.zeros( (len(t["Questions_int"])) )

		model = model_creator()
		model.knowledge = [(a, int(b)) for a, b in t["Knowledge"]]
		print len(model.knowledge)
		#model.knowledge = []

		#print t, '\n\n'
		model.update_all()
		if record_ce: all_cross_entropy.append(model.clustering_cost)

		for j in range(len(t["Questions_int"])):
			weight_matrix[j] = model.expected_gain(t["Questions_int"][j])

		avgScore = weight_matrix
		humanScore = get_person_scoring(i)
		pearsons.append( scistats.pearsonr(avgScore, humanScore)[0] )
		spearmans.append( scistats.spearmanr(avgScore, humanScore)[0] )
		standarderrors.append( np.average(np.std(weight_matrix, 0)) )


	if record_ce: return pearsons, spearmans, standarderrors, all_cross_entropy
	else: return pearsons, spearmans, standarderrors
	

def optimize_fullgame(tup):
	f = tup[0]
	models = tup[1]
	context_insensitive = tup[2]	
	index = tup[3]

	eig_vec = []
	item_name = f[0]
	item_indx = f[1]
	knowledge = []
	for q in f[2]:
		for m in models:
			m.knowledge = knowledge

		for i in context_insensitive:
			models[i].knowledge = []

		for m in models:
			m.update_all()


		question_options_int = q["QuestionOptions_int"]
		choice_str = q["Choice_str"]
		choice_index = int(q["choice_index"])

		eig = np.array([[m.expected_gain(f) for f in question_options_int] for m in models])
		knowledge_str = str(knowledge)
		question_options_int_str = str(question_options_int)
		eig_str = str(eig)

		#print '\n'
		#print knowledge_str
		#print question_options_int_str
		#print eig_str
		
		#eig = normalize(eig)
		
		#eig -= np.min(eig)
		#eig /= np.max(eig)

		eig_vec.append(eig)
		#print eig
		#print eig_vec
		#print ''
		resp = q["Resp"]
		k = (question_options_int[choice_index], resp)
		knowledge.append(k)
	#print "finished fullgame #" + str(item_indx)
	#print "final eig vec:", eig_vec
	return eig_vec

def full_game_multiple_heat_analysis(values_to_test = np.arange(-10, 10, 0.1), verbose = False):
	alpha = 0.1
	iters = 0

	#second is optimal, first is context insensitive
	#models = [RandomN_averaged(9, 5, 1), RandomN_averaged(9, 30, 1), RandomN_averaged(9, 100, 1)]
	models = [ClustPlayer(9)]
	#models = [RandchoicePlayer(9, 20)]
	context_insensitive = []
	randoms = []
	curtemp = np.ones(len(models))

	filepath = "random_objects_chosen"
	filepath += "-" + time.strftime("%m:%d:%Y-%H:%M:%S")

	with open(dir_for_this_run + "fullgame_analysis/"+filepath, 'w') as logfile:
		for r in randoms:
			logfile.write(repr((models[r].get_simulated_people())) + "\n\n")

	def softmax(raw_arr, heat):
		exp = np.array( [ math.exp(heat * v) for v in raw_arr] )
		return exp / np.sum(exp)

	def gradient(raw_arr, heat, choice):
		grad = 0
		grad += raw_arr[choice]
		grad -= sum([v * math.exp(heat * v) for v in raw_arr])/ \
				sum([math.exp(heat * v) for v in raw_arr])

		return grad

	pool = Pool()
	eig_vecs = pool.map(optimize_fullgame, [(f, models, context_insensitive, i) for f, i in zip(fullgames, range(1000))])
	pool.close()

	
	all_logprobs = []
	for i in values_to_test:
		curtemp = [i for j in models]
		logprob = np.zeros((questions_in_full_game, len(models)))
		numsamples = np.zeros(np.shape(logprob))
	
		for f, eig_vec in zip(fullgames, eig_vecs):
			knowledge = []
			for q, eig in zip(f[2], eig_vec):

				question_options_int = q["QuestionOptions_int"]
				choice_str = q["Choice_str"]
				choice_index = int(q["choice_index"])

				smax = np.array([softmax(expected_gains, heat) for expected_gains, heat in zip(eig, curtemp)])
				
				logprob[len(knowledge)] += np.array([math.log(smax[m][choice_index]) for m in range(len(models))])
				numsamples[len(knowledge)] += np.ones(len(models))
				
				if verbose:
					print '\n', [(features[feat] + " " + str(resp)) for feat, resp in knowledge]
					print [features[feat] for feat in question_options_int]
					print choice_index, item_name
					print eig
					print smax
					print np.array([math.log(smax[m][choice_index]) for m in range(len(models))])
					print logprob

				resp = q["Resp"]
				k = (question_options_int[choice_index], resp)
				knowledge.append(k)

		all_logprobs.append((np.sum(logprob, axis=0) / np.sum(numsamples, axis=0))[0])
	return all_logprobs

def full_game_analysis(verbose = False):
	alpha = 0.01
	iters = 0

	#second is optimal, first is context insensitive
	#models = [RandomN_averaged(9, 5, 1), RandomN_averaged(9, 30, 1), RandomN_averaged(9, 100, 1)]
	models = [ClustPlayer(9)]
	#models = [RandchoicePlayer(9, 20)]
	context_insensitive = []
	randoms = []
	curtemp = np.ones((23, len(models)))

	filepath = "random_objects_chosen"
	filepath += "-" + time.strftime("%m:%d:%Y-%H:%M:%S")

	with open(dir_for_this_run + "fullgame_analysis/"+filepath, 'w') as logfile:
		for r in randoms:
			logfile.write(repr((models[r].get_simulated_people())) + "\n\n")

	def softmax(raw_arr, heat):
		exp = np.array( [ math.exp(heat * v) for v in raw_arr] )
		return exp / np.sum(exp)

	def gradient(raw_arr, heat, choice):
		grad = 0
		grad += raw_arr[choice]
		grad -= sum([v * math.exp(heat * v) for v in raw_arr])/ \
				sum([math.exp(heat * v) for v in raw_arr])

		return grad

	pool = Pool()
	eig_vecs = pool.map(optimize_fullgame, [(f, models, context_insensitive, i) for f, i in zip(fullgames, range(1000))])
	pool.close()


	logprob = np.zeros((questions_in_full_game, len(models)))
	numsamples = np.zeros(np.shape(logprob))
	grad = np.ones((23, len(models)))

	while np.max(grad) > 0.00001:
	#for i in range(1):
		logprob = np.zeros((questions_in_full_game, len(models)))
		numsamples = np.zeros(np.shape(logprob))
		grad = np.zeros((23, len(models)))
		total_num_questions = 0

		for f, eig_vec, fullgame_index in zip(fullgames, eig_vecs, range(1000)):
			#print total_num_questions
			knowledge = []

			for q, eig in zip(f[2], eig_vec):
				person_index = fullgame_index / 4
				total_num_questions += 1
				question_options_int = q["QuestionOptions_int"]
				choice_str = q["Choice_str"]
				choice_index = int(q["choice_index"])

				smax = np.array([softmax(expected_gains, heat) for expected_gains, heat in zip(eig, curtemp[person_index])])
				if len(knowledge) == 0:
					pass
					#print str(smax) + " " + str(smax[0][choice_index])
				#print smax
				grad[person_index] += np.array([gradient(expected_gains, heat, choice_index) for expected_gains, heat in zip(eig, curtemp[person_index])])
				#grad += np.array([gradient(expected_gains, heat, choice_index) for expected_gains, heat in zip(eig, curtemp)])
				
				logprob[len(knowledge)] += np.array([math.log(smax[m][choice_index]) for m in range(len(models))])
				numsamples[len(knowledge)] += np.ones(len(models))
				
				if verbose:
					print '\n', [(features[feat] + " " + str(resp)) for feat, resp in knowledge]
					print [features[feat] for feat in question_options_int]
					print choice_index, item_name
					print eig
					print smax
					print np.array([math.log(smax[m][choice_index]) for m in range(len(models))])
					print logprob

				resp = q["Resp"]
				k = (question_options_int[choice_index], resp)
				knowledge.append(k)

			#print logprob
		old_temp = curtemp[:]
		curtemp += alpha * grad


		print "\n*********\nIterations:", iters
		print "Log probability by depth: \n", logprob /  numsamples
		print "Log probability overall:", np.sum(logprob, axis=0) / np.sum(numsamples, axis=0)
		#print "Num by depth:\n", numsamples
		print "\nGradient vector: \n", grad
		#print "Old temperature: ", old_temp
		#print "New temperature: ", curtemp

		

		iters += 1

	with open(dir_for_this_run + "fullgame_analysis/randomN_fullgame.txt", 'w') as f:
		f.write("Temperature:\n" + str(list(curtemp)) + "\n\n")
		f.write("Log probability by depth:\n" + str(list(logprob /  numsamples)) + "\n\n")
		f.write("Log probability overall:\n" + str(list(np.sum(logprob, axis=0) / np.sum(numsamples, axis=0))) + "\n\n")
		f.write("Num by depth:\n" + str(list(numsamples)))

	return logprob /  numsamples

def fullgame_percentage_analysis():
	
	context_insensitive = []
	models = [RandchoicePlayer(9, 20)]
	sumtot = [[0 for i in range(len(models))] for j in range(9)]
	numtot = [[0 for i in range(len(models))] for j in range(9)]
	randoms = []

	for i in range(1):
		
		pool = Pool()
		eig_vecs = pool.map(optimize_fullgame, [(f, models, context_insensitive, fullgame_index) for f, fullgame_index in zip(fullgames, range(1000))])
		pool.close()


		filepath = "random_objects_chosen"
		filepath += "-" + time.strftime("%m:%d:%Y-%H:%M:%S")

		with open(dir_for_this_run + "fullgame_analysis/"+filepath, 'w') as logfile:
			for r in randoms:
				logfile.write(repr((models[r].get_simulated_people())) + "\n\n")
			
		for f, eig_vec in zip(fullgames, eig_vecs):
					
			knowledge = []

			for q, eig in zip(f[2], eig_vec):


				question_options_int = q["QuestionOptions_int"]
				choice_str = q["Choice_str"]
				choice_index = int(q["choice_index"])

				for m_index in range(len(models)):
					max_index = np.argmax(eig[m_index])

					eig_normalized = eig[m_index][:]
					eig_normalized -= np.min(eig_normalized)
					eig_normalized /= np.max(eig_normalized)
					#print eig_normalized, np.argsort(eig_normalized), eig_normalized[choice_index], list(np.argsort(eig_normalized)).index(choice_index)
					#sumtot[m_index] += list(np.argsort(eig_normalized)).index(choice_index)
					sumtot[len(knowledge)][m_index] += 1 if np.argmax(eig_normalized) == choice_index else 0
					numtot[len(knowledge)][m_index] += 1

				
				resp = q["Resp"]
				k = (question_options_int[choice_index], resp)
				knowledge.append(k)


		sumtot_np = np.array(sumtot, dtype=np.float32)
		numtot_np = np.array(numtot, dtype=np.float32)
		print sumtot_np / numtot_np, np.sum(sumtot_np, axis=0) / np.sum(numtot_np, axis=0)
	
	sumtot_np = np.array(sumtot, dtype=np.float32)
	numtot_np = np.array(numtot, dtype=np.float32)
	return sumtot_np / numtot_np, np.sum(sumtot_np, axis=0) / np.sum(numtot_np, axis=0)


def analyze_variational(methods = ['greedy', 'random', 'optimal'], \
						value_calcs = ['kl_p_q', 'kl_q_p', 'entropy_label_space'], \
						clusts = range(5, 55, 5), \
						epsilon_and_n = False):

	def print_list_and_avgs(l, name, pad_len = 14):
		avg_len = float(len(l))
		print ("{:<"+str(pad_len)+"}: [{}], average: {: 0.3f}").format(name, ', '.join(list(['{: 0.3f}'.format(e) for e in l])), sum(l)/avg_len)


	method_value_combo = list(itertools.product(methods, value_calcs))



	averages_p = {}
	averages_s = {}
	averages_ce = {}

	for m in methods:
		averages_p[m] = {}
		averages_s[m] = {}
		averages_ce[m] = {}


	for method, vfunc in method_value_combo:
		ps, ss, cea = [], [], []

		all_ps = []
		all_ss = []
		all_ce = []


		print "\n+++++++++++++++++\nMethod:", method.upper()
		print "Value:", vfunc.upper()
		print "Cluster Amounts:", clusts, '\n'


		for c in clusts:
			verbose = False
			if vfunc == 'combined_cost': func = VariationalPlayer2.combined_cost
			elif vfunc == 'kl_p_q': func = VariationalPlayer2.kl_p_from_q
			elif vfunc == 'kl_q_p': func = VariationalPlayer2.kl_q_from_p
			elif vfunc == 'entropy_label_space': func = VariationalPlayer2.entropy_label_space
			elif vfunc == 'entropy_object_space': func = VariationalPlayer2.entropy_object_space
			else: func = None



			if epsilon_and_n: player_func = lambda: EpsilonAndNPlayer(c, method = method, value_func = func, verbose = verbose)
			else: player_func = lambda: VariationalPlayer2(c, method = method, value_func = func, verbose = verbose)

			p, s, _ , all_cross_entropy = full_oneshot_data_analyses(player_func, record_ce = True)
			
			n_trials = float(len(p))

			all_ps.append(p)
			all_ss.append(s)
			all_ce.append(all_cross_entropy)

			ps.append(np.sum(p) / n_trials)
			ss.append(np.sum(s) / n_trials)
			cea.append(np.sum(all_cross_entropy) / n_trials)



			print "For all depths at clust {}:".format(c)

			print_list_and_avgs(p, "Pearsons")
			print_list_and_avgs(s, "Spearmans")
			print_list_and_avgs(all_cross_entropy, "Cross Entropy")
			print '\n\n'




		averages_p[method][vfunc]  = ps
		averages_s[method][vfunc]  = ss
		averages_ce[method][vfunc] = cea


		print "\n---------\nAverages for every clust amount for method: {}, value func: {}".format(method.upper(), vfunc.upper())
		print_list_and_avgs(ps, "Pearsons")
		print_list_and_avgs(ss, "Spearmans")
		print_list_and_avgs(cea, "Cross Entropy")
		print '\n\n'

	print "\n\n********************\nAll methods:"
	for m in methods:
		print "\n\nMethod: {}\n".format(m.upper())
		for v in value_calcs:
			print "\nValue: {}".format(v.upper())

			print_list_and_avgs(averages_p[m][v], "Pearsons")
			print_list_and_avgs(averages_s[m][v], "Spearmans")
			print_list_and_avgs(averages_ce[m][v], "Cross Entropy")


	all_p = reduce(lambda x, y: x+y, averages_p)
	all_s = reduce(lambda x, y: x+y, averages_s)
	all_ce = reduce(lambda x, y: x+y, averages_ce)

	#f.savefig("new_plots_confirmation_bias/Trial_" + str(trial_index) + "_ticks_lettered.pdf")

	def make_scatter(x_axis, y_axis, x_name, y_name):
		f, ax = plt.subplots()

		pearson =  scistats.pearsonr(x_axis, y_axis)[0]
		spearman = scistats.spearmanr(x_axis, y_axis)[0]

		ax.scatter(x_axis, y_axis)
		ax.set_title("{}-{}\n $r = {}$, $\\rho = {}$".format(x_name, y_name, pearson, spearman))

		ax.set_xlabel(x_name)
		ax.set_ylabel(y_name)

		plt.savefig(base_path + "src/analysis_files/variational_analysis/plots/{}-{}.pdf".format(x_name, y_name))


	def make_line(x_values, y_values, names, x_name, y_name):
		f, ax = plt.subplots()

		handles = []
		for x, y, n in zip(x_values, y_values, names):
			handles.append(ax.plot(x, y, label = n))

		#plt.legend(handles, names)

		ax.set_title("{}-{}".format(x_name, y_name))
		ax.set_xlabel(x_name)
		ax.set_ylabel(y_name)

		plt.savefig(base_path + "src/analysis_files/variational_analysis/plots/{}-{}.pdf".format(x_name, y_name))

	def create_graphs():
		make_scatter(all_ce, all_p, "CE", "Pearson")
		make_scatter(all_ce, all_s, "CE", "Spearman")

		make_line([range(5, 55, 5)]*len(methods), averages_p, methods, "Clusters", "Pearson")
		make_line([range(5, 55, 5)]*len(methods), averages_s, methods, "Clusters", "Spearman")
		make_line([range(5, 55, 5)]*len(methods), averages_ce, methods, "Clusters", "CE")

	#create_graphs()


'''
print "Full Variational:\n\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n\n"
analyze_variational(methods = ['optimal', 'greedy', 'random'], \
					value_calcs = ['kl_p_q', 'kl_q_p', 'entropy_label_space', 'combined_cost'], \
					clusts = range(5, 30, 5), \
					epsilon_and_n = False)
'''
'''
print  "Epsilon and N:  \n\n\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\nXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\n\n"
analyze_variational(methods = ['greedy'], \
					value_calcs = ['entropy_label_space', 'combined_cost'], \
					clusts = range(2, 10, 2), \
					epsilon_and_n = True)
'''
'''
makedir(dir_for_this_run)
makedir(dir_for_this_run + "/fullgame_analysis/")
#full_game_analysis()
#analyze_times_seen()
fullgame_percentage_analysis()
'''
#print full_game_multiple_heat_analysis([])
#print fullgame_percentage_analysis()
#print [np.average(l) for l in full_oneshot_data_analyses()]
#full_game_analysis()
#print full_game_multiple_heat_analysis()
# dir_for_this_run = base_path+"analyzed_data/dir_for_time-" + time.strftime("%m:%d:%Y-%H:%M:%S") + "/"
# print "Good Subset "
# random_subsets_over_n(good = True)

'''

#analyze_times_seen()
n = 0
cursum = np.zeros((questions_in_full_game, 3), dtype = np.float32)
for i in range(100):
	to_add = full_game_analysis()
	if np.any(np.isnan(to_add)): 
		print "passed because NaN\n"
		print to_add
		print np.isnan(to_add)

		continue
	else: cursum += to_add
	n += 1
	print "\n\n************\nIterations:", n, "\n", cursum / n
#p, s, se = full_oneshot_data_analyses()
#print se, np.average(se)
'''
#print goodN_analysis(20)


p, s, _ = full_oneshot_data_analyses(model_creator = lambda: PositiveBiasModel())
print p, s
print np.sum(p) / 10.0, np.sum(s) / 10.0

#dir_for_this_run = base_path+"analyzed_data/dir_for_time-" + time.strftime("%m:%d:%Y-%H:%M:%S") + "/"
#random_subsets_over_n()

