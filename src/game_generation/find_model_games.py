from runner_clust import *

central = ['doorway', 'hotel', 'bird', 'pimple','sauce','pill','potato','arm','avenue','kid']
uncentral = ['skateboard','poison','railroad','mattress','missile','bomb','wreckage','dolphin','bullet','gift']

central_games = []
uncentral_games = []

for c in central:
	player = ClustPlayer(9)
	for i in range(10):
		gains = player.expected_gains()
        best_feature = player.features_left[np.argmax(gains)]
		resp = player.query_dat_name(best_feature, c)
		print best_feature, resp
		player.add_knowledge(best_feature, resp)
	central_games.append( (c, player.knowledge) )

for c in uncentral:
	player = ClustPlayer(9)
	for i in range(10):
		gains = player.expected_gains()
        best_feature = player.features_left[np.argmax(gains)]
		resp = player.query_dat_name(best_feature, c)
		print best_feature, resp
		player.add_knowledge(best_feature, resp)
	uncentral_games.append( (c, player.knowledge) )

print central_games
print '\n\n'
print uncentral_games