import numpy as np
from multiprocessing import Pool

def analyze(f):
	lines = []
	with open(f, "r") as thefile:
		for l in thefile:
			lines.append(l)



	for i in range(len(lines)):
		if lines[i].strip() == "Average pearson (by: trial number):":
			thearr = lines[i+1] + lines[i+2]
			arr = [elem.strip('[] \n') for elem in thearr.split(" ")]
			newarr = [e for e in arr if e != '']
			cors = [float(e) for e in newarr]
			avg = np.average(np.array(cors))
			print avg
			return avg

cors = []
p = Pool()
cors = p.map(analyze, ["../analyzed_data/strict_only_consider-04:17:2016-17:04:23/random_subset" + str(n) + ".txt" for n in range(5, 500, 5)])
print cors

from matplotlib import pyplot as plt
import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

#cors = [0.43717393199999999, 0.60467291899999998, 0.68615401100000006, 0.71285831700000002, 0.73496028299999994, 0.74333388099999997, 0.75122598699999998, 0.75557629599999987, 0.75855456899999996, 0.7594363340000001, 0.76151323199999998, 0.76422879199999993, 0.76224351400000001, 0.76406300599999999, 0.76433782499999992, 0.76427768900000004, 0.76497170899999989, 0.76706741300000014, 0.76510460499999999, 0.76616238699999994, 0.76661028599999992, 0.765847638, 0.76603268300000005, 0.76411693700000005, 0.76639842199999997, 0.76536534000000001, 0.76554029599999995, 0.76529128200000007, 0.76390297200000001, 0.76414049900000003, 0.76166224500000002, 0.764208683, 0.76367863800000002, 0.76372217399999998, 0.76294356500000005, 0.76341569600000003, 0.76354788699999998, 0.76347433700000011, 0.76285093299999995, 0.7618947439999999, 0.76033139500000002, 0.76098213100000001, 0.76120411600000004, 0.76145053099999993, 0.76068379299999989, 0.75886986499999998, 0.76051360899999998, 0.76017099499999996, 0.75975292600000011, 0.75764265200000003, 0.75901764399999994, 0.75853187899999996, 0.75874070699999996, 0.75862729399999995, 0.75842145999999988, 0.75792706900000006, 0.75852051199999992, 0.757204918, 0.75788698500000007, 0.75701162499999997, 0.75704693099999987, 0.75653207100000008, 0.75657983400000006, 0.75557534999999998, 0.75553143999999994, 0.75543362799999991, 0.75577215200000003, 0.75511001200000005, 0.75527994799999987, 0.75521446700000006, 0.75549851599999995, 0.75483000399999989, 0.75363546400000003, 0.75454623599999993, 0.755061384, 0.75400202599999999, 0.75353564500000003, 0.75417020400000001, 0.75324226599999999, 0.75313358499999994, 0.75193462, 0.75364228799999999, 0.75304938200000004, 0.75341731800000011, 0.75327527500000013, 0.75389172199999999, 0.75161712400000003, 0.75265900499999994, 0.75137911700000015, 0.75228993599999994, 0.75223512100000001, 0.75157634299999998, 0.75189263500000003, 0.75165112500000009, 0.75086998100000002, 0.75164829600000005, 0.75169054000000002, 0.75137125199999999, 0.750251153]

fig = plt.figure(figsize=(10,6))
#fig.suptitle('EIG for subset size', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)

ax.set_xlabel('Subset Size (K)')
ax.set_ylabel('Average Pearson Correlation')

ax.plot(range(5, 100, 5), cors[:len(range(5, 100, 5))])
plt.axhline(0.777, ls='dashed')
ax.xaxis.set_ticks(range(5, 100, 15))
plt.savefig("plots/correlation_for_subset_size.pdf", format='pdf')


plt.show()