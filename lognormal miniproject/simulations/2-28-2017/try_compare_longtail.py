import json
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from firefly_pack.network_simulator_and_scorer import NetworkHelper
from scipy.interpolate import Rbf
import scipy.ndimage.filters as filt

# things to try
# compare distribution of scores
# compare pareto optimal solutions
# compare size of

results_files = ['2-28-2017 20x150  simulations 1.json', # for simulation i
                 '2-28-2017 20x150  simulations 2.json',
                 '2-28-2017 20x150  simulations 3.json']
config_files = ['2-28-2017 20x150  config.json',     # for simulation i
                 '2-28-2017 20x150  config.json',
                 '2-28-2017 20x150  config.json']
netconfig_files = ['2-28-2017 20x150  networkconfig 1.json', # for simulation i
                 '2-28-2017 20x150  networkconfig 2.json',
                 '2-28-2017 20x150  networkconfig 3.json']
input_files = ['2-28-2017 inputs.json',                 # for simulation i
                 '2-28-2017 inputs.json',
                 '2-28-2017 inputs.json']

N_sim = len(results_files)
N_bugs, N_gen, N_objectives, N_params = [],[],[],[]
histories, params = [],[]

# load in data
for i_sim in range(N_sim):

    # get score distribution
    ############ load the firefly simulations
    results_file = open(results_files[i_sim], 'r')
    with results_file as data_file:
        fireflyHistory = json.load(data_file)
    results_file.close()

    N_bugs.append(np.shape(fireflyHistory)[1])  # reflecting the new save format
    N_gen.append(np.shape(fireflyHistory)[0])
    # indexSwitch = [[dict() for i_bug in range(N_bugs)] for i_gen in range(N_gen)]  # hacky fix - needed the transpose of this for the new save-format (todo fix the plotting to reflect the new format so we can skip this step)
    indexSwitch = [[dict() for i_gen in range(N_gen[i_sim])] for i_bug in range(N_bugs[i_sim])]  # hacky fix - needed the transpose of this for the new save-format (todo fix the plotting to reflect the new format so we can skip this step)
    for i in range(N_gen[i_sim]):
        for j in range(N_bugs[i_sim]):
            indexSwitch[j][i] = fireflyHistory[i][j]
    fireflyHistory = indexSwitch
    histories.append(fireflyHistory)

    config_file = open(config_files[i_sim], 'r')
    with config_file as data_file:
        config_data = json.load(data_file)
    config_file.close()

    PARAMS = config_data['PARAMS']  # ok to delete config_data at this point if you want
    params.append(PARAMS)

    networkconfig_file = open(netconfig_files[i_sim], 'r')
    with networkconfig_file as data_file:
        networkconfig_data = json.load(data_file)
    networkconfig_file.close()

    N_objectives.append(np.shape(fireflyHistory[0][0]['score'])[0]) # but assume all comparisons share the same objective functions and params
    N_params.append(config_data['N_params'])

    print 'simulation ' + str(i_sim) + ':'
    print 'N_bugs: ', str(N_bugs[i_sim]), ' N_gen: ', str(N_gen[i_sim]), ' N_scores ', str(N_objectives[i_sim]), ' N_params ', str(N_params[i_sim])
    print 'lognormal sigma ', networkconfig_data['logrand_sigma']
    print 'lognormal mu ', networkconfig_data['logrand_mu']

# populate a more convenient score list
score_lists = []
bugidx_lists = []
genidx_lists = []
for i_sim in range(N_sim):
    this_score_list = np.zeros((N_objectives[i_sim], N_gen[i_sim]*N_bugs[i_sim]))
    this_bugidx_list = np.zeros((N_objectives[i_sim], N_gen[i_sim]*N_bugs[i_sim]))
    this_genidx_list = np.zeros((N_objectives[i_sim], N_gen[i_sim]*N_bugs[i_sim]))
    for i_obj in range(N_objectives[i_sim]):
        sampleIdx = 0
        for i_gen in range(N_gen[i_sim]):
                for i_bug in range(N_bugs[i_sim]):
                    if ~np.isinf(histories[i_sim][i_bug][i_gen]['score'][i_obj]): # skip nan's  (but wait, these nan's are valuable information about the param space so maybe we should include them)
                        if ~np.isnan(histories[i_sim][i_bug][i_gen]['score'][i_obj]):
                            this_score_list[i_obj][sampleIdx] = histories[i_sim][i_bug][i_gen]['score'][i_obj]
                            this_bugidx_list[i_obj][sampleIdx] = i_bug
                            this_genidx_list[i_obj][sampleIdx] = i_gen
                            sampleIdx += 1
    score_lists.append(this_score_list[:,:sampleIdx])
    bugidx_lists.append(this_bugidx_list[:,:sampleIdx])
    genidx_lists.append(this_genidx_list[:,:sampleIdx])

# determine a cutoff for drawing iso lines
CUTOFF_PERCENT = 75
cutoffs = np.zeros(N_objectives[0])
for i_obj in range(N_objectives[0]):
    merged_scores = []
    for i_sim in range(N_sim):
        merged_scores.append( score_lists[i_sim][i_obj][:] )
    cutoffs[i_obj] = np.percentile(merged_scores,CUTOFF_PERCENT)
print cutoffs

# plot the distributions of scores
MINS = [0,0]  # todo would be better to set these by finding the min and max of the scores lists
MAXES = [200000,1]
TICKS = [2000,0.01]
PRCT_HEIGHT = [0.00009, 40]
colors = [cm.Set1(x) for x in np.linspace(0,0.3,N_sim)]
print colors
for i_obj in range(N_objectives[0]):
    plt.figure()
    bins = np.arange(MINS[i_obj],MAXES[i_obj],TICKS[i_obj])

    for i_sim in range(N_sim):
        plt.hist(score_lists[i_sim][i_obj][:],bins,histtype='step',normed='true',color=colors[i_sim])  # temp - plot them on top of each other
    plt.plot([cutoffs[i_obj], cutoffs[i_obj]], [0, PRCT_HEIGHT[i_obj]],color = [0.1, 0.1, 0.1])

    plt.title('score distributions for obj ' + str(i_obj))
    plt.xlabel('score')
    plt.ylabel('P(score)')
    plt.show()

# compare regions of solution space: samples

obj_dim1 = 0
obj_dim2 = 1

colors = [cm.Set1(x) for x in np.linspace(0,0.1 * N_sim,N_sim)]
plt.figure()
samples = [[],[],[]]
for i_sim in range(N_sim):
    xx,yy = [],[]
    sample_idxs = []
    for idx,val in enumerate(score_lists[i_sim][obj_dim1]):
        x = score_lists[i_sim][obj_dim1][idx]
        y = score_lists[i_sim][obj_dim2][idx]

        if x > cutoffs[obj_dim1] and y > cutoffs[obj_dim2]: # if in top percent
            xx.append(x)
            yy.append(y)
            sample_idxs.append(idx)

    # plot the samples, color coded by sim
    samples[i_sim] = sample_idxs
    plt.scatter(xx,yy,color=colors[i_sim],alpha=0.5)
plt.xlabel('rate score')
plt.ylabel('asynchrony score')
plt.title('after subsampling')
plt.show()

#print samples
#print np.shape(samples)   # key for interpreting samples:   [bug1gen1 bug2gen1 bug3gen1 bug1gen2 bug2gen2 bug3gen2 ...]

# compare regions of parameter space

def sample2bugidx(sample,i_sim, i_obj):
    bugidx = bugidx_lists[i_sim][i_obj][sample]
    genidx = genidx_lists[i_sim][i_obj][sample]
    #bugidx = int(np.remainder(sample,  N_bugs[i_sim]))
    #genidx = int(np.floor(sample / N_bugs[i_sim]))
    return [bugidx,genidx]

# test the translator function (todo instead, make a map with this info when first processing)

i_sim = 2
i_obj = 1
sample_idx = 1
test_sample = samples[i_sim][sample_idx]
idxs = sample2bugidx(sample_idx,i_sim,1)
bugidx = int(idxs[0])
genidx = int(idxs[1])
print bugidx
print genidx
print np.shape(bugidx)
print np.shape(genidx)
# sanity check - does score match
print " score check : " + str(histories[0][bugidx][genidx]['score'][0])
print " score check : " + str(histories[0][bugidx][genidx]['score'][1])







