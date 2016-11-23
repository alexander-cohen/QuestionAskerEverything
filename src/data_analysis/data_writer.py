#from runner_randomN import *
from import_all_files import *

import ast
import re
import cPickle as pickle

with open(base_path + "src/analysis_files/datalogs/fulldata_correctorder.txt", 'r') as fulldatafile:
	data = fulldatafile.read()

data = data.replace('\n', '|')

def splitIntoParts(delimeters, s):
	regex = '|'.join(delimeters)
	return re.split(regex, s)

def parseQuestion(question):
	lines = question.split('|')
	qdict = {}
	for l in lines:
		if ':' in l:
			key = l.split(':')[0]
			val = l.split(':')[1]
			try:
				val = ast.literal_eval(val)
			except:
				pass

			qdict[key] = val
	choice_index = qdict["QuestionOptions_int"].index(qdict["Choice_int"])
	qdict["choice_index"] = choice_index
	return qdict

def parseGame(game):
	parts = splitIntoParts(['--------------', '0000', '1111', '2222', '3333'], game)

	for p in parts:
		if p[:9] == '|Item_str':
			lines = p.split('|')
			item_str = lines[1].split(':')[1]
			item_int = lines[2].split(':')[1]


	questions = [q for q in parts if q[:11] == '|Choice_str']

	return str(item_str), int(item_int), [parseQuestion(q) for q in questions]

def getFullGames(s):
	dat = splitIntoParts(['--------------', '1111', '2222', '3333'], s)
	dat = [d for d in dat if d[:9] == "|Item_str"]
	
	return [parseGame(g) for g in dat]

def getFullOneshots(s):
	dat = splitIntoParts(['--------------', '0000', '1111', '2222', '3333'], s)
	oneshots = []
	order = range(10)
	for d in dat:
		if d[:15] == "|OneshotOrders:": 
			lines = d.split('|')
			order = ast.literal_eval(lines[1].split(":")[1])
			print order
		if d[:9] == "|TrialNum":
			#print d, '\n\n\n'
			lines = d.split('|')
			oneshotdict = {}
			for l in lines:
				if ':' in l:
					key = l.split(':')[0]
					val = l.split(':')[1]
					try:
						val = ast.literal_eval(val)
					except:
						pass

					oneshotdict[key] = val
			oneshots.append(oneshotdict)
	

	realoneshots = [oneshots[i] for i in range(10)]
	for oneshot in realoneshots:
		print oneshot["Depth"]

	return realoneshots

def getAllOneshots(peoplestrs):
	trials = [[] for i in range(len(peoplestrs))]
	for p, i in zip(peoplestrs, range(1000)):
		for oneshot, j in zip(getFullOneshots(p), range(1000)):
			trials[j].append(oneshot)


def getDataAsDicts(data):
	peoplestrs = data.split("--------------")[1:]
	people = []
	for p in peoplestrs:
		people.append( {"fullgames":getFullGames(p), "oneshots":getFullOneshots(p)} )

	return people, getFullGames(data), getAllOneshots(peoplestrs)

people, fgames, oneshots = getDataAsDicts(data)

with open(base_path+"src/analysis_files/datalogs/peopledata_correctorder.pickle", 'w') as peopledatapicklefile:
	pickle.dump(people, peopledatapicklefile)

with open(base_path+"src/analysis_files/datalogs/peopledata_correctorder.txt", 'w') as peopledatatextfile:
	pickle.dump(str(people), peopledatatextfile)

with open(base_path+"src/analysis_files/datalogs/fullgamedata_correctorder.pickle", 'w') as fullgamedatapicklefile:
	pickle.dump(fgames, fullgamedatapicklefile)

with open(base_path+"src/analysis_files/datalogs/fullgamedata_correctorder.txt", 'w') as fullgamedatatextfile:
	pickle.dump(str(fgames), fullgamedatatextfile)

with open(base_path+"src/analysis_files/datalogs/oneshotdata_correctorder.pickle", 'w') as oneshotdatapicklefile:
	pickle.dump(oneshots, oneshotdatapicklefile)

with open(base_path+"src/analysis_files/datalogs/oneshotdata_correctorder.txt", 'w') as oneshotdatatextfile:
	pickle.dump(str(oneshots), oneshotdatatextfile)