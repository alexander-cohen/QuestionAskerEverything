lines = []
from runner_numpy import *
with open("../src/datalogs/fulldata.txt", 'r') as fulldata:
	for l in fulldata:
		lines.append(l)


found_items = []
curlist = []
for l in lines:
	if l[:9] == 'Item_str:':
		split = l.split(":")
		obj = split[1].strip()
		#print obj
		curlist.append(items.index(obj))
	elif l[:14] == '--------------':
		found_items.append(curlist)
		curlist = []

found_items.append(curlist)
found_items = found_items[1:]

central = [[5, 'kitchen'], [1, 'hockey'], [17, 'shrimp'], [16, 'chicken'], [11, 'bark'], [2, 'cradle'], [4, 'garlic'], [6, 'vegetable'], [14, 'disease'], [3, 'tangerine']]
uncentral = [[6, 'wood'], [7, 'shop'], [9, 'flesh'], [16, 'mustard'], [5, 'coat'], [4, 'scarf'], [8, 'dandelion'], [2, 'football'], [3, 'canteen'], [19, 'pony']]

with open("datalogs/peopledata.pickle", 'r') as peopledata:
	people = pickle.load(peopledata)

central_games = []
uncentral_games = []

for c in central:
	person = people[c[0]-1]
	item = c[1]
	games = person["fullgames"]
	for g in games:
		if g[0] == item:
			for elem in g[2]:
				try:
					k = elem["Knowledge"]
				except:
					pass
			central_games.append((items.index(item), item, k))

	#print games, '\n\n'

for c in uncentral:
	person = people[c[0]-1]
	item = c[1]
	games = person["fullgames"]
	for g in games:
		if g[0] == item:
			for elem in g[2]:
				try:
					k = elem["Knowledge"]
				except:
					pass
			uncentral_games.append((items.index(item), item, k))
	#print games, '\n\n'

print "Central Games:\n", central_games, '\n\n'
print "Uncentral Games:\n", uncentral_games


