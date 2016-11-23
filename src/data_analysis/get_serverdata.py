'''
Created on Sep 5, 2015

@author: alxcoh
'''
from import_all_files import *

analyze_randomchoice = False
analyzes = False
analyze_fullgame = False
analyze_fullgame_probs = False
import random
from runner_expected_utility import ExpectedUtilityPlayer
from runner_randomN import *

if analyzes:
    from tree_variational import *
    from runner_variational import *
    from runner_maxprob import *

elif analyze_fullgame:
    from runner_clust import *
    from tree_variational import *
    from runner_variational import *
    
#from sqlalchemy import create_engine, MetaData, Table
import numpy as np
import json
#from matplotlib import pyplot as plt
import cPickle as pickle
#from runner_nonbayesian import NonBayesianPlayer

#from matplotlib import pyplot as plt
import scipy.stats as scistats
from multiprocessing import Pool
import data_loader

data_matrix, data_dict, features, objects = data_loader.get_data()

fulldatafile = open(base_path + "src/analysis_files/datalogs/fulldata_correctorder.txt", 'w')


'''
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
'''
with open(base_path+"data/experiment_data.pickle", 'r') as data_file:
    data = pickle.load(data_file)
        
def get_answers(data_json):
    lines = data_json['data'];
    return [l['trialdata'] for l in lines[2:]]
    

datum = []
class Fullgame:
    def __init__(self, game):
        self.game = game
        #print self.game
        self.item = self.game[1][0][1][0]
        #print "item:", self.item
        self.questions = []
        self.ranksum = 0.0
        self.all_eig = []
        #self.questions_seen = self.get_questions_seen()
    
    def get_questions_seen(self):
        questions_seen = np.zeros(218)
        depth = 0
        #print "\nNew Game:", self.item
        for arr in self.game[1]:
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
                #print "rank:", rank
                k = (features.index(choice), resp)
                
                #print [features.index(q) for q in questionarr]
                for q in questionarr:
                    questions_seen[features.index(q)] += 1
                 
                depth += 1
        
        return questions_seen
    
    def analyze_percent(self):
        self.models = [ClustPlayer(9)]
        
        self.sumprobs = [0 for i in self.models]
        self.sumtotal = [0 for i in self.models]
      
        depth = 0
        print "New Game:", self.item
        question_index = 0
        for arr in self.game[1]:
            k = []
            if arr[0] == '20q_choice':
                question_index += 1
                #if question_index > 2: break
                choice = arr[1][3]
        
                resp = float(arr[1][4])
                #print arr[1][6]
                questionarr = sorted(arr[1][6], key = lambda x: -float(x[1]))
                questionarr = [e[0] for e in questionarr]
                
                rank = questionarr.index(choice)
                indx = questionarr.index(choice)
                self.questions.append( (questionarr, indx, resp) )
                self.ranksum += rank
                
                options_bynum = [str(features.index(f))for f in questionarr]
                choice_bynum = features.index(str(choice))
                #print "rank:", rank
                k = (features.index(choice), resp) 
                
                #print '*****'
                
                print '\n', features[k[0]], k[1]
                print questionarr
                
                for m, i in zip(self.models, range(1000)):
                    m.knowledge.append(k)
                    m.update_all()

                    
               
                for m, i in zip(self.models, range(1000)):
                    #print m.knowledge
                    
                    infogains = np.array([m.expected_gain(f) for f in options_bynum])
                    maxindx = list(infogains).index(max(list(infogains)))
                    print infogains, maxindx, rank
                    if rank == 0:
                        self.sumprobs[i] += 1
                    self.sumtotal[i] += 1
                
                depth += 1
                print ""

        print "Sumprobs: ", self.sumprobs, "\n***************\n\n"
        return self.sumprobs, self.sumtotal
    
    def analyze(self):
        self.models = []
        #self.models.append( ["fullbayesian", ClustPlayer(9), np.zeros(10, dtype=np.float16)] )
        '''
        for i in range(9, -1, -1):
            self.models.append( ["clust" + str(i), ClustPlayer(i), np.zeros(10, dtype = np.float16), np.zeros(10, dtype = np.float16)] )
        '''
        #Optimal Temperature:
        #[ 0.60785209  0.54815729  0.56051099  0.52031569  0.39981981  0.23249061 -0.08519039 -0.63554068 -1.41480307 -0.68202096]
        self.optimal_temp = [0.60785209,  0.54815729,  0.56051099,  0.52031569,  0.39981981,  0.23249061, -0.08519039, -0.63554068, -1.41480307, -0.68202096]
        self.models = [ClustPlayer(9), ClustPlayer(9), \
                       RandchoicePlayer(9, 20), RandomN(9, rands=randobjects), \
                       ClustPlayer(9)]

        self.sumprobs = [0 for i in self.models]
        self.sumtotal = [0 for i in self.models]
        self.sumspearman = 0
        self.numspearman = 0
        depth = 0
        print "New Game:", self.item
        question_index = 0
        for arr in self.game[1]:
            if arr[0] == '20q_choice':
                question_index += 1
                #if question_index > 2: break
                choice = arr[1][3]
        
                resp = float(arr[1][4])
                #print arr[1][6]
                questionarr = sorted(arr[1][6], key = lambda x: -float(x[1]))
                questionarr = [e[0] for e in questionarr]
                
                rank = questionarr.index(choice)
                indx = questionarr.index(choice)
                self.questions.append( (questionarr, indx, resp) )
                self.ranksum += rank
                
                options_bynum = [str(features.index(f))for f in questionarr]
                choice_bynum = features.index(str(choice))
                #print "rank:", rank
                k = (features.index(choice), resp)
               
                #print '*****'
                
                print '\n', features[k[0]], k[1]
                
                def softmax(choice, total, index, heat = 10):
                    #val = 1 if choice == max(total) else 0
                    val = math.e**(heat*choice) / np.sum([ math.e**(heat*c) for c in total ])
                    print "index:", index, "rank:", sorted(list(total))[::-1].index(choice), format(val, "0.3f"), format(choice, "0.3f"), [format(x, "0.3f") for x in total]
                    return val
                
                
                
                #if question_index < 3: continue
                print questionarr
                
                for m, i in zip(self.models, range(1000)):
                    m.knowledge.append(k)
                    m.update_all()

                    
                normal = np.array([self.models[8].expected_gain(f) for f in options_bynum])
                normal -= np.min(normal)
                normal /= np.sum(normal)
                for m, i in zip(self.models, range(1000)):
                    #print m.knowledge
                    
                    infogains = np.array([m.expected_gain(f) for f in options_bynum])
                    #infogains -= np.min(infogains)
                    #infogains /= np.max(infogains)
                    infogains -= np.min(infogains)
                    infogains /= np.max(infogains)
                    
                    spearman = scistats.spearmanr(normal, infogains)[0]
                    self.sumspearman += spearman
                    self.numspearman += 1.0
                    if spearman > 0.068 and i != 8: continue
                    #print "Spearman:", spearman
                    
                    prob_choose = softmax(infogains[indx], infogains, i, self.optimal_temp[i])
                    self.sumprobs[i] += math.log(prob_choose)
                    self.sumtotal[i] += 1
                
                depth += 1
                print ""

        print "Sumprobs: ", self.sumprobs, "\n***************\n\n"
        return self.sumprobs, self.sumtotal, self.sumspearman, self.numspearman



    def optimize(self, curtemp, randobjects):
        self.models = []
        first_time = True
        if len(self.all_eig) > 0: first_time = False
        print first_time
        #print first_time, self.all_eig
        #self.models.append( ["fullbayesian", ClustPlayer(9), np.zeros(10, dtype=np.float16)] )

#         for i in range(9, -1, -1):
#             self.models.append( ["clust" + str(i), ClustPlayer(i), np.zeros(10, dtype = np.float16), np.zeros(10, dtype = np.float16)] )
        #first is optimal, second is context insensitive
        '''self.models = [ClustPlayer(9), ClustPlayer(9), \
                       RandchoicePlayer(9, 20), RandomN(9, rands=randobjects), \
                       ClustPlayer(9)]
        '''
        self.models = [ClustPlayer(9)]

        self.gradient = [0 for m in self.models]
        self.sumtotal = [[0 for m in self.models] for d in range(10)]
        self.numtotal = [[0 for m in self.models] for d in range(10)]
        self.sumspear = [[0 for m in self.models] for d in range(10)]
        self.numspear = [[0 for m in self.models] for d in range(10)]
        depth = 0
        #print "New Game:", self.item
        question_index = 0
        real_index = -1
        for arr, game_index in zip(self.game[1], range(10000)):
            if arr[0] == '20q_choice':
                real_index += 1
                if first_time: eig_instance = []
                question_index += 1
                choice = arr[1][3]
                resp = float(arr[1][4])

                questionarr = sorted(arr[1][6], key = lambda x: -float(x[1]))
                questionarr = [e[0] for e in questionarr]
                
                rank = questionarr.index(choice)
                indx = rank
                self.questions.append( (questionarr, indx, resp) )
                self.ranksum += rank
                
                options_bynum = [str(features.index(f))for f in questionarr]
               
                k = (features.index(choice), resp)
                
                def gradient(choice, total, heat):
                    tosubtract = np.sum( [v*(math.e**(v*heat)) for v in total] )
                    tosubtract /= np.sum( [(math.e**(v*heat)) for v in total] )
                    return choice - tosubtract
                
                def softmax(choice, total, heat):
                    #val = 1 if choice == max(total) else 0
                    val = math.e**(heat*choice) / np.sum([ math.e**(heat*c) for c in total ])
                    return val
                
                
                #self.models[1].knowledge = []
                #self.models[1].update_all()
               
                    
                for m, i in zip(self.models, range(1000)):
                    
                    if first_time:
                        infogains = np.array([m.expected_gain(f) for f in options_bynum])
                        #print '\n', m.knowledge
                        #print options_bynum
                        #print infogains
                        eig_instance.append(infogains)
                        #print "eig computed manually", first_time, self.all_eig
                    else: 
                        #print "use stored eig", first_time
                        infogains = self.all_eig[real_index][i]
                    
                    chosen_val = infogains[indx]
                    
                    gradient_component = gradient(chosen_val, infogains, curtemp[i])
                    softmax_value = softmax(chosen_val, infogains, curtemp[i])
                    #if len(m.knowledge) == 0: print softmax_value
                    
                    self.gradient[i] += gradient_component
                    self.sumtotal[depth][i] += math.log(softmax_value)
                    self.numtotal[depth][i] += 1.0
                    
                    spearman = 0
                    self.sumspear[depth][i] += spearman
                    self.numspear[depth][i] += 1.0
                    
                    
                    
                    #print gradient_component, self.gradient[i], self.gradient
                if first_time:
                    for m, i in zip(self.models, range(1000)):
                        m.knowledge.append(k)
                        m.update_all()
                        

                depth += 1
            
                if first_time: self.all_eig.append(eig_instance)
        
        if first_time:
            pass
            #print self.all_eig

        return np.array(self.gradient), np.array(self.sumtotal), np.array(self.numtotal), np.array(self.sumspear), np.array(self.numspear)

class Trial:
    def __init__(self, t, randchoices = [] ):
        self.data = t[1]
        #print self.data
        self.trial_num = int(self.data[1][1][1])
        #print self.data[5]
        #print "data:", self.data
    
        self.questions = [elem[0] for elem in filter(lambda x: x[0] == 'questions_to_rank', self.data)[0][1][2]]
        self.ranked = filter(lambda x: x[0] == 'ranked_choices', self.data)[0][1][3]
        self.bonus = filter(lambda x: x[0] == 'bonus', self.data)
        #print self.bonus
        self.rankorder = [self.ranked.index(elem) for elem in self.questions]
        #print '\n\nRankorder:', self.rankorder
        #self.ranked_order = np.argsort([int(elem) for elem in filter(lambda x: x[0] == 'ranked_choices', self.data)[0][1][2]])
        #print '\n\n', self.ranked_order
        self.freeform = filter(lambda x: x[0] == 'quest_freeform', self.data)[0][1]
        #print self.trial_num, self.freeform
        self.qa = self.data[1][1][2]
        self.depth = len(self.qa)
        
        #print self.freeform
        #print self.ranked
        #print self.qa
        #print ""
        
        if True:
            #print '\n\nCreating models'
            #models = [VariationalPlayer(min(2**n, 1000)) for n in range(1, 7)] + [RandchoicePlayer(n, 20) for n in range(10)] + [ClustPlayer(9)]
            knowledge = []
            for k in self.qa:
                knowledge.append((int(float(k[0])), int(float(k[1]))))
            #print knowledge
            variational = False
            #models = [VariationalPlayer(knowledge) for n in range(7)] + [ClustPlayer(9)]
            #models = [ClustPlayer(9) for n in range(8)] + [NonBayesianPlayer()]
            #models = [ClustPlayer(9)]
            if randchoices != []:
                models = [RandomN(9, rands = randchoices)]
            else:
                models = [RandomN(9, n=20)]
            #models = [RandchoicePlayer(9)]
            #models = [MaxprobPlayer(9)]
            #models = [RandchoicePlayer(9, 20)]
            #models = [VariationalPlayer(knowledge)]
            #models = [ClustPlayer(9)]
            
            variational = False
            if variational:
                models[0].start(32)
            else: 
                models[-1].knowledge = knowledge
                models[-1].update_all()
                
            self.expected = []
            self.ordered = []
            self.pearson = []
            for m in models:
                self.expected.append( [m.expected_gain(r) for r in self.questions] )
                ordered = sorted(self.questions, key = lambda x: -self.expected[-1][self.questions.index(x)])
                self.ordered.append( [ordered.index(elem) for elem in self.questions] )
                #print self.ordered[-1]
                self.pearson.append( scistats.pearsonr([0, 1, 2, 3, 4, 5], self.ordered[-1])[0] )
                


        

class Oneshot:
    def __init__(self, trial, order, rands = []):
        #print order
        #print len(trial)

            #print t[1]
        #print '\n\n'
        #print order, '\n\n'
        self.trials = [Trial(trial[order.index(i)], randchoices = rands) for i in range(10)]
        '''
        for t in self.trials:
            #self.pearsons.append((t.depth, t.pearson))
            print "trialnum, depth:", t.trial_num, t.depth
        '''
        #print "PERSON COMPLETED"
        #nt trial[0]
        '''
        for t in self.trials:
            #print t
            #print int(t[0][1]), t[1:]
            for a in t:
                continue
                #print a[0]
                if a[0] == "ranked_choices": print a[1][3]
                if a[0] == "quest_freeform": print a[1]
                print ''
            #self.trials[int(t[0][1])] = Trial(t)
        #for t in self.trials:
            #print t
        '''
class Person:
    def __init__(self, d, nrands = 20):
        self.fullgames = []
        self.oneshots = []
        self.rands = np.array(random.sample(range(1000), nrands))
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
        if analyzes or analyze_randomchoice: self.analyzed_oneshots = self.analyze_oneshots()
        if not analyzes: self.fullgames = self.analyze_fullgames()
        '''
        self.questions_seen = np.zeros(len(features))
        print "\n\nlen:", len(self.fullgames)
        for f in self.fullgames:
            print list(f.questions_seen)[78]
            self.questions_seen += f.questions_seen

        print '\n',list(self.questions_seen)[78]
        '''
        #self.models = [[np.zeros(10, dtype=np.float16), np.zeros(10, dtype=np.float16)] for i in range(10)]
        
        #for m in self.models: print np.sum(m[0]), m[0]
            
            #print f
    
    def analyze_fullgames(self):
        '''
        self.models.append( ("fullbayesian", np.zeros(10, dtype=np.float16)) )
        for i in range(8, 0, -1):
            self.models.append( ("clust" + str(i), np.zeros(10, dtype = np.float16)) )
        '''
        #print "All games:", self.fullgames[1:]
        return [Fullgame(full) for full in self.fullgames[1:]]
            
    def analyze_oneshots(self):
        return Oneshot(self.oneshots, self.order, self.rands)
        
ranksum = 0.0
ranknum = 0.0
n = 0
n_p = 0
all_avg = []
#models = [[np.zeros(10, dtype=np.float16), np.zeros(10, dtype=np.float16)] for i in range(9)]
data = data[:3] + data[4:]
oneshots = []
people = []

def getPerson(data):
    return Person(data)

'''
people = [Person(d) for d in data]
times_seen = [list(p.questions_seen) for p in people]
with open(base_path+"experiment_data/times_seen.pickle", 'w') as peoplefile:
    pickle.dump(times_seen, peoplefile)
'''
if analyze_randomchoice:
    curN = 20
    costs = {}
    cors = {}
    while curN <= 1000:
        total_cost = 0
        sumcor = 0
        numcor = 0
        sumspear = 0
        for j in range(5):
            print "creating people"
            people = [[Person(datapiece, curN) for datapiece in data] for i in range(5)]
            print "people created"
            trials_model = [[] for i in range(10)]
            trials_people = [[] for i in range(10)]
            for person_set in people:
                for p in person_set:
                    for t, i in zip(p.analyzed_oneshots.trials, range(100)):
                        trials_model[i].append(t)
                        
            for p in people[0]:
                for t, i in zip(p.analyzed_oneshots.trials, range(100)):
                    trials_people[i].append(t)
    
                    
    
                    
            trials_model = sorted(trials_model, key = lambda x: x[0].depth)
            trials_people = sorted(trials_people, key = lambda x: x[0].depth)
            
            for trial_person, trial_model in zip(trials_people, trials_model):
                matrx_person = np.array([t.rankorder for t in trial_person])
                matrx_model = []
                for t in trial_model:
                    expected = t.expected[0]
                    sort_expect = sorted(expected)
                    toadd = [sort_expect.index(e) for e in expected]
                    #print expected, sort_expect, toadd
                    matrx_model.append(toadd)
                matrx_model = np.array(matrx_model)
                
                standard_error_person = list(np.std(matrx_person, axis=0))
                standard_error_model = list(np.std(matrx_model, axis=0))
                
                cost = np.sum([(sp - sm)**2 for sp, sm in zip(standard_error_person, standard_error_model)])
                #print "Standard error person:", standard_error_person
                #print "Standard error model:", standard_error_model
                #print "Cost:", cost
                #print ''
                person_ranks = [5-x for x in np.average(matrx_person, axis=0)]
                model_ranks = [x for x in np.average(matrx_model, axis=0)]
                cor_this = scistats.pearsonr(person_ranks, model_ranks)[0]
                spear_this = scistats.spearmanr(person_ranks, model_ranks)[0]
                print "For this trial:", cor_this, spear_this
                sumcor += cor_this
                sumspear += spear_this
                numcor += 1
                total_cost += cost
        print "\n\n**************\nCurN:", curN, "Cost:", total_cost, \
                    "Cor:", (sumcor/numcor), "Spear:", (sumspear/numcor), '\n************\n'

        costs[curN] = total_cost
        cors[curN] = sumcor/numcor
        
        curN += 1
    print "Costs:", costs
    print ''
    print "Cors:", cors

elif analyzes:
    p = Pool()
    #people = p.map(getPerson, data)
    people = map(getPerson, data)
    p.close()
    '''
    for i, d in zip(range(100), data):  
        p = Person(d)
        people.append(p)
    '''
    with open(base_path+"experiment_data/people_clust_random500.pickle", 'w') as peoplefile:
        pickle.dump(people, peoplefile)

    #print '\n\n'
    #7,2,1,6,0,3,5,4,8,9
elif analyze_fullgame:
    '''
    sumprobs = np.zeros(10)
    numprobs = np.zeros(10)
    sumspearman = 0
    numspearman = 0
    for datapiece in data:
        p = Person(datapiece)
        games = p.fullgames
        
        #print games
        
        for f in games:
            sumtot, numtot, spearman, nspearman = f.analyze()
            sumspearman += spearman
            numspearman += nspearman
            sumprobs += np.array(sumtot)
            numprobs += np.array(numtot)
    print "Sum probs:", sumprobs
    print "Num probs:", numprobs
    print "Net probs:", sumprobs/numprobs
    print "Average Spearman:", sumspearman/numspearman, sumspearman, numspearman
    '''
    num_models = 1
    curtemp = np.ones((23, num_models))
    curtemp *= 1
    
    alpha = 0.05
    iters = 0
    
    rand20 = random.sample(range(1000), 20)
    
    allgames = []
    for datapiece in data:
        #print "\n-------------------\nNew Person\n"
        p = Person(datapiece)
        games = p.fullgames
        allgames.append(games)

    while True:
        sumprobs = np.array([np.zeros(num_models) for d in range(10)])
        numprobs = np.array([np.zeros(num_models) for d in range(10)])
        sumspear = np.array([np.zeros(num_models) for d in range(10)])
        numspear = np.array([np.zeros(num_models) for d in range(10)])
        print "\nIteration started:", iters+1 
        print "Current Temperature:", curtemp
        gradient = np.zeros(num_models)
        
            
        for games, pindex in zip(allgames, range(1000)):
            for f in games:
                grad_inc, sumtot, numtot, sums, nums = f.optimize(curtemp[pindex], rand20)
                sumprobs += sumtot
                numprobs += numtot
                sumspear += sums
                numspear += nums
                gradient += grad_inc
                curtemp[pindex] += alpha * grad_inc
                #print "Gradient added:", gradient
                #print ''
                
        #print "\nTotal Gradient:", gradient
        #print "\nTo Add:", alpha*gradient
        #curtemp += alpha*gradient
        #print "\nNew Temp:", curtemp
        #print "\nCurrent sums:", sumprobs
        #print "\nCurrent nums:", numprobs
        print "\nCurrent probs:\n", sumprobs/numprobs
        print "\nAverage prob:", np.sum(sumprobs, axis=0) / np.sum(numprobs, axis=0)
        #print "\nAverage spear:", sumspear/numspear
        #print '\n**********************\n'
        iters += 1
        
#note: prob is 193/401
elif analyze_fullgame_probs:
    sumprobs = 0
    numprobs = 0
    

    for datapiece in data:
        print "\n-------------------\nNew Person\n"
        p = Person(datapiece)
        games = p.fullgames
        
        
        for f in games:
            s, n = f.analyze_percent()
            sumprobs += s[0]
            numprobs += n[0]
            print s, n
    print sumprobs
    print numprobs


else:
   
    for datapiece in data:
        personstr = "--------------\n"
        fullgames = []
        oneshots = []
        order = datapiece['questiondata']['order']

        personstr += "WorkerID:" + str(datapiece['workerId']) + "\n"
        personstr += "AssignmentID:" + str(datapiece['assignmentId']) + "\n"
        personstr += "Order:" + str(order) + "\n"


        for elem in datapiece['data']:
            try:
                trialdata = elem['trialdata']
                if trialdata[0] == '20q': fullgames.append(trialdata)
            except:
                pass
            try:
                trialdata = elem['trialdata']
                if trialdata[0] == 'oneshot_data': oneshots.append(trialdata)
            except:
                pass

        for f in fullgames:
            fullgamestr = ""
            item_str = f[1][0][1][0]
            item_int = objects.index(item_str)
            fullgamestr += "Item_str:" + str(item_str) + "\n"
            fullgamestr += "Item_int:" + str(item_int) + "\n"
            knowledge = []
            for arr, i in zip(f[1], range(len(f[1]))):
                questionstr = ""
                if arr[0] == '20q_choice':
                    choice_str = arr[1][3]
                    choice_int = features.index(choice_str)
                    questionarr = arr[1][6]
                    questionarr = [str(e[0]) for e in questionarr]
                    questionarr_int = [features.index(e) for e in questionarr]
                    resp = data_matrix[item_int, choice_int]
                    knowledge.append( (choice_int, resp) )
                    questionstr += "Choice_str:" + choice_str + "\n"
                    questionstr += "Choice_int:" + str(choice_int) + "\n"
                    questionstr += "QuestionOptions_str:" + str(questionarr) + "\n"
                    questionstr += "QuestionOptions_int:" + str(questionarr_int) + "\n"
                    questionstr += "Resp:" + str(resp) + "\n"
                
                fullgamestr += ("0000\n" if i > 0 and i < len(f[1])-1 else "") + questionstr

            fullgamestr += "Knowledge:" + str(knowledge) + "\n"
            personstr += "1111\n" + fullgamestr

        personstr += "2222\n"
        trials = [oneshots[i] for i in range(10)]
        oneshotstr = ""
        oneshotstr += "OneshotOrders:" + str(order) + "\n"

        for t in trials:
            trialstr = ""
            trialdata = t[1]
            trial_num = trialdata[1][1][1]
            
            questions = [int(elem[0]) for elem in filter(lambda x: x[0] == 'questions_to_rank', trialdata)[0][1][2]]
            questions_str = [features[e] for e in questions]

            ranked = filter(lambda x: x[0] == 'ranked_choices', trialdata)[0][1][3]
            ranked = [int(e) for e in ranked]

            bonus = filter(lambda x: x[0] == 'bonus', trialdata)
            rankorder = [ranked.index(elem) for elem in questions]

            freeform = filter(lambda x: x[0] == 'quest_freeform', trialdata)[0][1]
            qa = trialdata[1][1][2]
            depth = len(qa)

            knowledge = []
            for k in qa:
                knowledge.append((int(k[0]), float(k[1])))

            bonus_val = 0 if len(bonus) == 0 else 0.2
            freeform_val = str(freeform[2]).upper()
            qa = [[int(e[0]), float(e[1])] for e in qa]

            trialstr += "3333\n"
            trialstr += "TrialNum:" + str(trial_num) + "\n"
            trialstr += "Questions_int:" + str(questions) + "\n"
            trialstr += "Questions_str:" + str(questions_str) + "\n"
            trialstr += "Ranked:" + str(ranked) + "\n"
            trialstr += "Bonus:" + str(bonus_val) + "\n"
            trialstr += "Rankorder:" + str(rankorder) + "\n"
            trialstr += "Freeform:" + freeform_val + "\n"
            trialstr += "QuestionAnswerPairs:" + str(qa) + "\n"
            trialstr += "Depth:" + str(depth) + "\n"
            trialstr += "Knowledge:" + str(knowledge) + "\n"
            print depth
            oneshotstr += trialstr
        print '\n', order, '\n\n'
        personstr += oneshotstr

        fulldatafile.write("9999\n")
        fulldatafile.write(personstr)

    fulldatafile.close()

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