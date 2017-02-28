import numpy as np
import json
from pprint import pprint



###### firefly config

config_prefix = 'huge 2-16-2017' # for making the filestring
target_dir = '../simulations/2-21-2017/'

verbose = 'compare weight distributions' # optionally provide a general description of the current endeavor
PARAMS = ['p_ei','p_ie','p_ii','w_input','p_inpi','p_iinp'] # ,'lognorm_sigma'] # name for easier printing
#OBJECTIVES = ['stable duration','rate_score','asynchrony_score'] # '['asynchrony','stable duration'] # names for easier printing & reference
OBJECTIVES = ['rate_score']
                    # (for now the second obj dimension is not necessary)

N_gen = 150  # working towards 100+
N_bugs = 30
N_params = len(PARAMS)
N_objectives = len(OBJECTIVES)
#N_repetitions = 3 # repeat each simulation N_rep times to get a better estimate of obj score

# range for [p_ei, p_ie, p_ii, w_input]
#MEANS = [0.15, 0.15, 0.2, 10] # ,-1] # for each param  # WARNING trying a uniform distribution instead
#STDS = [0.1, 0.1, 0.1, 6] # ,3]
#MAXES = [0.5, 0.5, 0.5, 12] # 13] # , 5]
#MINS = [0.1, 0.1, 0.1, 6] # 3] # , -10]  # sigma must be > 0
MINS = [0.1, 0.1, 0.15, 0.9, 0.1, 0.1]
MAXES = [0.5, 0.4, 0.35, 1.1, 0.5, 0.4]
    # example hand-tuned solution to search around during testing:
#network_params = [0.3475, 0.1375, 0.1855, 1.5, 0.34, 0.165] # p_ei, p_ie, p_ii, w_input, p_inpi, p_iinp  # tau_e > tau_i




characteristic_scales = np.zeros((N_params,)) # note this gets saved as a list (for serialization)
                                                #  todo fix code to unpack it
                                                        #  note I think I took care of this, double check
for i_param in range(N_params):
    characteristic_scales[i_param] = MAXES[i_param] - MINS[i_param]
    #characteristic_scales[i_param] = 2*STDS[i_param]

alpha = 0.075 # 0.035 # NOTE alpha gets scaled for each param in Firefly Dynamics function
beta = 6  # >4 yields chaotic firefly dynamics
absorption = 0.6 # somewhere around 0.5 is good according to Yang

annealing_constant = 0.995 # currently only beta is being annealed

############# network config












########## saving

# save the strings
#saveName = title + ' verbose info.json'
#verboseInfoFile = open(saveName,'w')
#json.dump(verbose,verboseInfoFile,indent=2)
#verboseInfoFile.close()

# repackage the config constants into a dictionary
config_dict = {"N_gen":N_gen,"N_bugs":N_bugs,"N_params":N_params,"N_objectives":N_objectives,
               #"N_repetitions":N_repetitions,
               #"MEANS":MEANS,"STDS":STDS,
               "MAXES":MAXES,"MINS":MINS,"characteristic_scales": characteristic_scales.tolist(),
               "alpha":alpha,"beta":beta,"absorption":absorption,"annealing_constant":annealing_constant,
               "verbose":verbose,"PARAMS":PARAMS,"OBJECTIVES":OBJECTIVES,"config_prefix":config_prefix,"target_dir":target_dir}

# save the result as json formatted data
saveName = target_dir + config_prefix + ' config.json'
configFile = open(saveName,'w')
json.dump(config_dict,configFile,sort_keys=True,indent=2)
configFile.close()