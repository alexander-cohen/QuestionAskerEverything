analyze_randomchoice = False
analyzes = False
analyze_fullgame = False
analyze_fullgame_probs = True
import random
from runner_randchoice import RandchoicePlayer
from runner_randomN import *
    
from sqlalchemy import create_engine, MetaData, Table
import numpy as np
import json
#from matplotlib import pyplot as plt
import cPickle as pickle
#from runner_nonbayesian import NonBayesianPlayer

#from matplotlib import pyplot as plt
import scipy.stats as scistats
from multiprocessing import Pool
import data_loader
import copy

import cPickle as pickle
data_matrix, data_dict, features, items = data_loader.get_data()


db_url = "mysql://lab:fishneversink@gureckislab.org/mt_experiments"
table_name = '20q_model_tester_exp2'
data_column_name = 'datastring'
# boilerplace sqlalchemy setup
engine = create_engine(db_url)
metadata = MetaData()
metadata.bind = engine
table = Table(table_name, metadata, autoload=True)
# make a query and loop through
s = table.select()
rows = s.execute()


data = []
#status codes of subjects who completed experiment
#statuses = [3,4,5,7]
statuses = [7]
# if you have workers you wish to exclude, add them here
exclude = ['A150GMV1YQWWB3:3JAOYWH7VJVPDAV8W7VF35ZLO279L6',\
		   'A2WGW5Y3ZFBDEC:32ZKVD547GERLZKGOL5BYP0UCJSB3J',\
		   'AZ9VAMDM5PEOR:3EJJQNKU9SWTVE84Z3G32Y68D74HRU',\
		   'A11S8IAAVDXCUS:3Z3ZLGNNSJLEP5M57321TEVTYHB3Q1']
for row in rows:
    # only use subjects who completed experiment and aren't excluded
    i = 0
    if row['status'] in statuses and \
    	row['uniqueid'] not in exclude and \
    	row['beginhit'].year == 2016:
        try:
            dstr = row[data_column_name]
            dstr = dstr.replace("DO YOU USE IT?", "DO YOU USE IT DAILY?")
            data.append(json.loads(dstr))
            #print i, row['status']
            i +=1
        except:
            continue

def format_oneshot_dict(trial_data):
	oneshot_data = trial_data[1]
				
	oneshot_dict = {}
	for e in oneshot_data:
		key = e[0]
		value = e[1]
		
		oneshot_dict[key] = value

	reformatted_dict = {}


	item = items.index(oneshot_dict['oneshot_itm_chosen'][2])
	#for k in oneshot_dict:
	#	print k

	reformatted_dict['quest_freeform'] = oneshot_dict['quest_freeform'][2] #just the question part
	reformatted_dict['question_answer_pairs'] = oneshot_dict['question_answer_pairs'][2] #just the important part
	reformatted_dict['question_answer_pairs'] = [[int(q), float(a)] for q, a in reformatted_dict['question_answer_pairs']] #convert to numbers from strings
	reformatted_dict['rankings'] = [int(c) for c in oneshot_dict['ranked_choices'][2]]
	reformatted_dict['ranked_choices'] = [int(q) for q, a in oneshot_dict['questions_to_rank'][2]]
	reformatted_dict['item'] = item

	return reformatted_dict
				

#get data into nicer format
def format_dstring(dstring):
	unformated_data = dstring['questiondata']
	main_data = dstring['data']

	order = unformated_data['order']

	oneshots = []

	for d in main_data:
		trial_data = d['trialdata'] #trial data list

		try: #will work if trial data is a list, otherwise skip
			if trial_data[0] == 'oneshot_data': #is oneshot
				oneshot_dict = format_oneshot_dict(trial_data)
				oneshots.append(oneshot_dict)

		except:
			continue
	
	#print len(oneshots), [len(o['question_answer_pairs']) for o in oneshots], sorted([len(o['question_answer_pairs']) for o in oneshots])
	#print "{}:{}".format(dstring['workerId'], dstring['assignmentId']), order, '\n\n'
	
	try:
		oneshots = [oneshots[order.index(i)] for i in range(len(order))] #we can find the i'th oneshot at the index of that element in the order list
		return oneshots
	except:
		return

	for o in oneshots:
		for k in o:
			pass
			#print "{}: {}".format(k, o[k])
		#print '\n******************\n'



oneshots_by_people = [format_dstring(d) for d in data] #list of list of oneshots, outer is people inner is oneshot
oneshots_by_trial = [[p[i] for p in oneshots_by_people] for i in range(len(oneshots_by_people[0]))] #list of list of oneshots, outer is trial inner is all play throughs by all people


with open(base_path+"data/experiment2/oneshots_by_people.pickle", 'w') as f:
	pickle.dump(oneshots_by_people, f)

with open(base_path+"data/experiment2/oneshots_by_trial.pickle", 'w') as f:
	pickle.dump(oneshots_by_trial, f)
