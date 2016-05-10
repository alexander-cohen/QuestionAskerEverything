import cPickle as pickle
from runner_randomN import *
from runner_randchoice import RandchoicePlayer
import scipy.stats as scistats
from multiprocessing import Pool
import os
import time

with open("datalogs/fullgamedata.pickle", 'r') as fullgamedata:
	fullgames = pickle.load(fullgamedata)

with open("datalogs/oneshotdata.pickle", 'r') as oneshotdata:
	oneshots = pickle.load(oneshotdata)

with open("datalogs/peopledata.pickle", 'r') as peopledata:
	people = pickle.load(peopledata)

with open("datalogs/fullgamedata_correctorder.pickle", 'r') as fullgamedata:
	fullgames_correctorder = pickle.load(fullgamedata)

with open("datalogs/oneshotdata_correctorder.pickle", 'r') as oneshotdata:
	oneshots_correctorder = pickle.load(oneshotdata)

with open("datalogs/peopledata_correctorder.pickle", 'r') as peopledata:
	people_correctorder = pickle.load(peopledata)

should_strict_only_consider = True

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

	return seen_before_pertrial

def correlationOfChoicesWithSeenBefore(persondata):
	numSeenPerTrial = getNumSeenBeforePerTrial(persondata)
	their_oneshots = persondata["oneshots"]
	allpearson = []
	allspearman = []
	for t, numSeenInTrial in zip(their_oneshots, numSeenPerTrial):
		rankorder = t["Rankorder"]
		rankorder = [5-n for n in rankorder]
		seen_ranks = [numSeenInTrial[indx] for indx in t["Questions_int"]]
		
		if max(rankorder) != min(rankorder) and max(seen_ranks) != min(seen_ranks):
			pearson = scistats.pearsonr(rankorder, seen_ranks)[0]
			spearman = scistats.spearmanr(rankorder, seen_ranks)[0]

		else:
			pearson = 0
			spearman = 0

		allpearson.append(pearson)
		allspearman.append(spearman)

	return allpearson, allspearman

def analyze_times_seen():
	allcors = []
	for p in people_correctorder:
		allcors.append(correlationOfChoicesWithSeenBefore(p))

	overalltotpear = 0
	overalltotspear = 0
	overallnumtot = 0

	for p, i in zip(allcors, range(1000)):
		print "\nPerson " + str(i) + ":"
		totpear = 0
		totspear = 0
		numtot = 0

		for i in range(len(p[0])):
			print "Trial " + str(i) + " (pearson/spearman):", p[0][i], p[1][i]
			totpear += p[0][i]
			totspear += p[1][i]
			numtot += 1.0

		overalltotpear += totpear
		overalltotspear += totspear
		overallnumtot += numtot

		print "Average pearson for person " + str(i) + ":", str(totpear/numtot)
		print "Average spearman for person " + str(i) + ":", str(totspear/numtot)

	print "\n\n"
	print "Average pearson overall:", overalltotpear/overallnumtot
	print "Average spearman overall:", overalltotspear/overallnumtot

def get_person_scoring(trialnum):
	nresps = len(people[0]["oneshots"][0]["Questions_int"])
	person_matrx = np.zeros( (len(people), nresps)  )
	for p, i in zip(people, range(1000)):
		for j in range(nresps):
			person_matrx[i, j] = 5 - p["oneshots"][trialnum]["Rankorder"][j]

	return np.average(person_matrx, 0)

def perform_random_subset(numSimPeople = 25, numObjects = 20):
	#use person 0 for now, as it does not matter
	person = people[2]
	ntrials = len(person["oneshots"])
	objectsChosen = [[random.sample(range(1000), numObjects) for i in range(numSimPeople)] for j in range(ntrials)]
	
	cors = np.zeros((ntrials, numSimPeople))
	pearsons = []
	spearmans = []
	standarderrors = []
	for t, i in zip(person["oneshots"], range(1000)):
		#print t["Depth"]
		simulatedPeople = [RandomN(9, numObjects, objectsChosen[i][x], strict_only_consider = should_strict_only_consider) for x in range(numSimPeople)]
		weight_matrix = np.zeros( (numSimPeople, len(t["Questions_int"])) )

		for s, personIndx in zip(simulatedPeople, range(1000)):
			s.knowledge = t["Knowledge"]
			print s.knowledge
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

def perform_n_random_subsets(numiters, numSimPeople = 25, numObjects = 20, numTrials = 10):
	pearsonMat = np.zeros( (numiters, numTrials) )
	spearmanMat = np.zeros( (numiters, numTrials) )
	standarderrorsMat = np.zeros( (numiters, numTrials) )
	allObjectsChosen = []

	p = Pool()
	vals = p.map(perform_random_subset_tup, [(numSimPeople, numObjects)]*numiters)
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

def analyze_randomsubsets(numIters, numSimPeople, numObjects, log = True, printout = True, verbosePrint = False, verboseLog = True, numappend = 0):
	ac, pm, sm, sem = perform_n_random_subsets(numIters, numSimPeople, numObjects)

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
		fullpath = "../analyzed_data/" + filepath + "/"
		makedir(fullpath)

		with open("../analyzed_data/" + filepath + "/random_subset" + str(numappend) + ".txt", 'w') as randomsubsetfile:
			randomsubsetfile.write(datstr if verboseLog else datstr_concise)

	if printout:
		print datstr if verbosePrint else datstr_concise

	return ac, pm, sm, sem, datstr, datstr_concise

def analyze_randomsubsets_tup(tup):
	return analyze_randomsubsets(*tup)

def random_subsets_over_n(startSize = 5, endSize = 500, stepSize = 5, numItersPer = 100, numPeoplePer = 25, log = True):
	filepath = "strict_only_consider" if should_strict_only_consider else "loose_only_consider"
	filepath += "-" + time.strftime("%m:%d:%Y-%H:%M:%S")
	fullpath = "../analyzed_data/" + filepath + "/"
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
		ac, pm, sm, sem, datstr, datstr_concise = analyze_randomsubsets(numItersPer, numPeoplePer, n, log = True, printout = True, verbosePrint = False, verboseLog = True, numappend = n)
		allstr.append(datstr)
		allobjects.append(ac)
		pearsons.append(np.average(pm))
		spearmans.append(np.average(sm))
		standarderrors.append(np.average(sem))



	if log:
		with open("../analyzed_data/" + filepath + "random_subset_overn_strictconsider.txt", 'w') as thefile:
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

def full_oneshot_data_analyses():
	#use person 0 for now, as it does not matter
	person = people[0]
	ntrials = len(person["oneshots"])
	
	pearsons = []
	spearmans = []
	for t, i in zip(person["oneshots"], range(1000)):
		#print t["Depth"]
		model = ClustPlayer(9)
		weight_matrix = np.zeros( (len(t["Questions_int"])) )

		model.knowledge = t["Knowledge"]
		model.update_all()

		for j in range(len(t["Questions_int"])):
			weight_matrix[j] = model.expected_gain(t["Questions_int"][j])

		avgScore = weight_matrix
		humanScore = get_person_scoring(i)
		pearsons.append( scistats.pearsonr(avgScore, humanScore)[0] )
		spearmans.append( scistats.spearmanr(avgScore, humanScore)[0] )

	return pearsons, spearmans

def full_game_analysis(verbose = False):
	alpha = 0.1
	iters = 0

	#second is optimal, first is context insensitive
	models = [RandomN_averaged(9, 5, 100), RandomN_averaged(9, 30, 100), RandomN_averaged(9, 100, 100)]
	randoms = [0, 1, 2]
	curtemp = np.ones(len(models))

	filepath = "random_objects_chosen"
	filepath += "-" + time.strftime("%m:%d:%Y-%H:%M:%S")

	with open("../analyzed_data/fullgame_analysis/"+filepath, 'w') as logfile:
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

	eig_vecs = []
	for f in fullgames:
		eig_vec = []
		item_name = f[0]
		item_indx = f[1]
		knowledge = []
		for q in f[2]:
			for m in models[1:]:
				m.knowledge = knowledge
			for m in models:
				m.update_all()


			question_options_int = q["QuestionOptions_int"]
			choice_str = q["Choice_str"]
			choice_index = int(q["choice_index"])

			eig = np.array([[m.expected_gain(f) for f in question_options_int] for m in models])
			eig = normalize(eig)
			eig_vec.append(eig)
			resp = q["Resp"]
			k = (question_options_int[choice_index], resp)
			knowledge.append(k)

		eig_vecs.append(eig_vec)

	while True:
		logprob = np.zeros((10, len(models)))
		numsamples = np.zeros(np.shape(logprob))
		grad = np.zeros(len(models))
		for f, eig_vec in zip(fullgames, eig_vecs):
			
			knowledge = []

			for q, eig in zip(f[2], eig_vec):

				question_options_int = q["QuestionOptions_int"]
				choice_str = q["Choice_str"]
				choice_index = int(q["choice_index"])

				smax = np.array([softmax(expected_gains, heat) for expected_gains, heat in zip(eig, curtemp)])
				#print smax
				grad += np.array([gradient(expected_gains, heat, choice_index) for expected_gains, heat in zip(eig, curtemp)])
				
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

		print "\n*********\nIterations:", iters
		print "Log probability by depth: \n", logprob /  numsamples
		print "Log probability overall:", np.sum(logprob, axis=0) / np.sum(numsamples, axis=0)
		print "Num by depth:\n", numsamples
		print "\nGradient vector: ", grad
		print "Old temperature: ", curtemp

		curtemp += alpha * grad
		print "New temperature: ", curtemp

		iters += 1



#random_subsets_over_n(log = True, numItersPer = 1, numPeoplePer = 1)
#analyze_times_seen()
full_game_analysis()