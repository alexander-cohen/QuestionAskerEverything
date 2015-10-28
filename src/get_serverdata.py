'''
Created on Sep 5, 2015

@author: alxcoh
'''
from runner_randchoice import RandchoicePlayer
'''
Created on Jul 1, 2015

@author: alxcoh
'''
from sqlalchemy import create_engine, MetaData, Table
import numpy as np
import json
import pandas as pd
from matplotlib import pyplot as plt
import cPickle as pickle
from runner_clust import *
from matplotlib import pyplot as plt
import scipy.stats as scistats
db_url = "mysql://lab:fishneversink@gureckislab.org/mt_experiments"
table_name = '20q_model_tester'
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
exclude = []
for row in rows:
    # only use subjects who completed experiment and aren't excluded
    i = 0
    if row['status'] in statuses and row['uniqueid'] not in exclude:
        try:
            dstr = row[data_column_name]
            dstr = dstr.replace("DO YOU USE IT?", "DO YOU USE IT DAILY?")
            data.append(json.loads(dstr))
            #print i, row['status']
            i +=1
        except:
            continue
        

        
def get_answers(data_json):
    lines = data_json['data'];
    return [l['trialdata'] for l in lines[2:]]
    

datum = []
class Fullgame:
    def __init__(self, game):
        self.item = game[1][0][1][0]
        print self.item
        self.questions = []
        self.ranksum = 0.0
        
        self.models = []
        #self.models.append( ["fullbayesian", ClustPlayer(9), np.zeros(10, dtype=np.float16)] )
        '''
        for i in range(9, -1, -1):
            self.models.append( ["clust" + str(i), ClustPlayer(i), np.zeros(10, dtype = np.float16), np.zeros(10, dtype = np.float16)] )
        '''
        self.models = []
        depth = 0
        for arr in game[1]:
            if arr[0] == '20q_choice':
                choice = arr[1][3]
        
                resp = float(arr[1][4])
                #print arr[1][6]
                questionarr = sorted(arr[1][6], key = lambda x: -float(x[1]))
                questionarr = [e[0] for e in questionarr]
                
                rank = questionarr.index(choice)
                indx = questionarr.index(choice)
                self.questions.append( (questionarr, indx, resp) )
                self.ranksum += rank
                print rank
                k = (features.index(choice), resp)
                #print '*****'
                for m in self.models:
                    infogains = m[1].get_unnormed_gains()
                    order = (np.argsort( infogains[ np.array([features.index(x) for x in questionarr]) ] ) )[::-1]
                    orderedm = sorted(questionarr, key = lambda x: -infogains[features.index(x)])
                    
                    rank = orderedm.index(choice)
                    
                    #print m[0], rank
                    m[2][depth] += rank
                    m[3][depth] += 1.0
                    
                    '''
                    orderedm = sorted(questionarr, key = lambda x: -infogains[features.index(x)])
                    m[2][depth] += orderedm.index(choice)
                    #print m[0], m[2][depth]
                    m[1].knowledge.append(k)
                    m[1].update_all()
                    '''
                    '''
                    print '\n************\n', depth, rank, m[0]
                    for o in orderedm:
                        if choice == o: print '-' + str(o)
                        else: print '', o
                    '''
                depth += 1
                #print self.questions[-1]
            #print depth
        
        '''
        for m in self.models:
            print "***********"
            print m[0]
            print m[2]
            print m[3]
        '''  
        global all_avg
        all_avg.append(self.ranksum/float(len(self.questions)))
        print self.ranksum, len(self.questions), self.ranksum/float(len(self.questions))
        print "\n====================\n"

class Trial:
    def __init__(self, t):
        self.data = t[1]
        #print self.data
        self.trial_num = int(self.data[1][1][1])
        #print self.data[5]
        print self.data
    
        self.questions = [elem[0] for elem in filter(lambda x: x[0] == 'questions_to_rank', self.data)[0][1][2]]
        self.ranked = filter(lambda x: x[0] == 'ranked_choices', self.data)[0][1][3]
        self.rankorder = [self.ranked.index(elem) for elem in self.questions]
        print '\n\n', self.rankorder
        #self.ranked_order = np.argsort([int(elem) for elem in filter(lambda x: x[0] == 'ranked_choices', self.data)[0][1][2]])
        #print '\n\n', self.ranked_order
        self.freeform = filter(lambda x: x[0] == 'quest_freeform', self.data)[0][1]
        self.qa = self.data[1][1][2]
        self.depth = len(self.qa)
        
        #print self.freeform
        #print self.ranked
        #print self.qa
        #print ""
        players = [ClustPlayer(n) for n in range(10)] + [RandchoicePlayer(n) for n in range(10)]
        for k in self.qa:
            #print k, k[0], k[1], float(k[0]), int(float(k[1]))
            for p in players:
                p.knowledge.append((float(k[0]), int(float(k[1]))))
                p.update_all()
        
        self.expected = []
        self.ordered = []
        self.pearson = []
        for p in players:
            self.expected.append( [p.expected_gain(r) for r in self.questions] )
            ordered = sorted(self.questions, key = lambda x: -self.expected[-1][self.questions.index(x)])
            self.ordered.append( [ordered.index(elem) for elem in self.questions] )
            print self.ordered[-1]
            self.pearson.append( scistats.pearsonr([0, 1, 2, 3, 4, 5], self.ordered[-1])[0] )
            
        

        

class Oneshot:
    def __init__(self, trial, order):
        #print order
        #print len(trial)

            #print t[1]
        #print '\n\n'
        self.trials = [Trial(trial[order.index(i)]) for i in range(10)]
        for t in self.trials:
            #self.pearsons.append((t.depth, t.pearson))
            print t.trial_num, t.depth
        print "NEW PERSON"
        #print trial[0]
        '''
        for t in self.trials:
            #print t
            #print int(t[0][1]), t[1:]
            for a in t:
                pass
                #print a[0]
                if a[0] == "ranked_choices": print a[1][3]
                if a[0] == "quest_freeform": print a[1]
                print ''
            #self.trials[int(t[0][1])] = Trial(t)
        #for t in self.trials:
            #print t
        '''
class Person:
    def __init__(self, d):
        self.fullgames = []
        self.oneshots = []
        self.order = d['questiondata']['order']
        #print self.order
        #print d['questiondata']
        '''
        try:
            print d['questiondata'][u'comments-general']
            print d['questiondata'][u'comments-technical']
        except:
            pass
        '''
        for elem in d['data']:
            try:
                trialdata = elem['trialdata']
                if trialdata[0] == '20q': self.fullgames.append(trialdata)
            except:
                pass
            try:
                trialdata = elem['trialdata']
                if trialdata[0] == 'oneshot_data': self.oneshots.append(trialdata)
            except:
                pass
        self.ranknum = 0.0
        self.ranksum = 0.0
        self.ranknum_models = 0.0
        #self.models = [[np.zeros(10, dtype=np.float16), np.zeros(10, dtype=np.float16)] for i in range(10)]
        
        #for m in self.models: print np.sum(m[0]), m[0]
            
            #print f
    
    def analyze_fullgames(self):
        '''
        self.models.append( ("fullbayesian", np.zeros(10, dtype=np.float16)) )
        for i in range(8, 0, -1):
            self.models.append( ("clust" + str(i), np.zeros(10, dtype = np.float16)) )
        '''
        for full in self.fullgames[1:]:
            f = Fullgame(full)
            self.ranknum += len(f.questions)
            self.ranksum += f.ranksum
            for i  in range(len(f.models)):
                m = f.models[i]
                self.models[i][0] += m[2]
                self.models[i][1] += m[3]
                #print m[0], m[2]
                
            '''
            for i in range(len(self.models)):
                
                self.models[i][1] += f.models[i][2]
            '''
        '''
        for m in self.models:
            print m
        ''' 
            
    def analyze_oneshots(self):
        #print len(self.oneshots)
        
        return Oneshot(self.oneshots, self.order)
        
ranksum = 0.0
ranknum = 0.0
n = 0
n_p = 0
all_avg = []
#models = [[np.zeros(10, dtype=np.float16), np.zeros(10, dtype=np.float16)] for i in range(9)]
data = data[:3] + data[4:]
oneshots = []
'''
for i, d in zip(range(100), data):  
    p = Person(d)
    o = p.analyze_oneshots()
    oneshots.append(o)
with open("oneshots4.pickle", 'w') as oneshotsfile:
    pickle.dump(oneshots, oneshotsfile)

    #print '\n\n'
    #7,2,1,6,0,3,5,4,8,9
'''
'''
for i, d in zip(range(100), data):  
    p = Person(d)
    n += len(p.fullgames)
    n_p += 1
    ranksum += p.ranksum
    ranknum += p.ranknum
    
    #for x in range(len(models)):
    #    models[x][0] += p.models[x][0]
    #    models[x][1] += p.models[x][1]
    
    #m = models[-1]
    #print np.sum(m[0])
    #print m[0]
    #print m[0] / m[1]
print ranksum / ranknum, n, n_p
plt.hist(all_avg, bins=10)
plt.show()
'''
'''
for m in models:
    print "**********"
    print np.sum(m[0])
    print m[0]
    print m[1]
    print np.array(m[0]) / np.array(m[1])
'''

def get_all_answers():
    times = []
    comments = []
    all_answers = []
    sketchy_answers = []
    for d in data:
        their_answers = []
        try:
            their_answers = get_answers(d)
        except:
            continue
        
        all_answers += their_answers
        
        try:
            time = float(d['questiondata']['time'])
            times.append(time)
            if time < 300 and time > 200:
                sketchy_answers += their_answers
            comments.append(d['questiondata']['comments'])
        except:
            continue
        
    return all_answers, sketchy_answers, times, comments
'''
all_answers_file = open('pickled_data/all_answers_file.pickle', 'w')
sketchy_answers_file = open('pickled_data/sketchy_answers_file.pickle', 'w')
times_file = open('pickled_data/times_file.pickle', 'w')
comments_file = open('dapickled_datata/comments_file.pickle', 'w')

all_answers, sketchy_answers, times, comments = get_all_answers()

all_answers_file.write(pickle.dumps(all_answers))
sketchy_answers_file.write(pickle.dumps(sketchy_answers))
times_file.write(pickle.dumps(sketchy_answers))
comments_file.write(pickle.dumps(comments))

all_answers_file.close()
sketchy_answers_file.close()
times_file.close()
comments_file.close()
'''