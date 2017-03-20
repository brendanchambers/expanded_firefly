from get_exemplars import get_exemplars
import json
import numpy as np
#

simulation_dir = '../lognormal miniproject/simulations/3-2-2017/' # location of the output and config files
config_filename = '2-28-2017 20x150  config.json'
save_filename = 'exemplars 3-5-2017.json'

# the simulation categories, as result files
simulations = ['2-28-2017 20x150  simulations 1.json',  # heaviest tail
               '2-28-2017 20x150  simulations 2.json',  # ...
               '2-28-2017 20x150  simulations 3.json']  # less extreme heavy tail
# the simulation categories, as result files
sim_configfile_names = ['2-28-2017 20x150  networkconfig 1.json',  # heaviest tail
               '2-28-2017 20x150  networkconfig 2.json',  # ...
               '2-28-2017 20x150  networkconfig 3.json']  # less extreme heavy tail
N_sim = len(simulations)

# pull out a subset of exemplars from each - for the purpose of generating longer recordings of activity
exemplars = [None]*N_sim
for i_sim, sim_name in enumerate(simulations):
    results_load_path = simulation_dir + sim_name
    sim_configfile_path = simulation_dir + sim_configfile_names[i_sim]
    firefly_configfile_path = simulation_dir + config_filename
    exemplars[i_sim] = get_exemplars(results_load_path, sim_configfile_path, firefly_configfile_path)
    print "exemplar i:" + str(exemplars[i_sim])
    print " size of exemplars i: " + str(np.shape(exemplars[i_sim]))


print "exemplars across simulations:" + str(exemplars)
print " size of exemplars: " + str(np.shape(exemplars))
'''
saveObject = {'exemplars':exemplars.tolist()}

# save the exemplars as json text
results_dir = 'analyses/3-2-2017/'
results_f_name = results_dir + 'exemplars.json'

resultsFile = open(results_f_name, 'w')
json.dump(saveObject, resultsFile, sort_keys=True, indent=2)
resultsFile.close()
'''
results_dir = 'analyses/3-2-2017/'
results_f_name = results_dir + 'exemplars.npy'
np.save(results_f_name,exemplars)

# todo