import time
import numpy as np

from firefly_pack.network_simulator_and_scorer import NetworkHelper
import matplotlib.pyplot as plt

# switches:
network_config_filename = 'simulations/2-27-2017/2-27-2017 40x90  networkconfig 1.json'
inputcurrents_filestring = 'simulations/2-27-2017/2-27-2017 inputs.json'

# init
network_helper = NetworkHelper(network_config_filename) # this object does all the simulation work

if network_helper.__class__.cell_inputs == None: # if not yet initialized
    network_helper.initializeInputs(inputcurrents_filestring)  # initialize inputs as a global static variable - currently this still gets copied when it gets made into a timed array

N_objectives = 2 # todo load from some file instead
N_repeats = 5
score_vectors = np.zeros((N_repeats, N_objectives))
for i_repeat in range(N_repeats):
    #print " i_repeat " + str(i_repeat)
    startTime = time.time()

    # a clean pipeline omitting the firefly stuff
    network_params = [0.3475, 0.26, 0.1855, 1.1, 0.34, 0.165] # p_ei, p_ie, p_ii, w_input, p_inpi, p_iinp  # tau_e > tau_i
    #network_params = [0.25, 0.12, 0.19, 1.5, 0.34, 0.18] # p_ei, p_ie, p_ii, w_input, p_inpi, p_iinp  # tau_e = tau_i

    score_vector = network_helper.simulateActivity(network_params,verboseplot=False)
    score_vectors[i_repeat,:] = score_vector

    print "score_vector: " + str(score_vector)

    stopTime = time.time()
    running_time = stopTime - startTime
    print "running time: " + str(running_time)

print "score vectors: " + str(score_vectors)


doPlot = True
n_bins = 25
if doPlot:
    for i_dim in range(N_objectives):
        plt.figure()
        plt.hist(score_vectors[:,i_dim],n_bins)
        plt.title('objective ' + str(i_dim) + ' score variability at init conditions')
        plt.show()

        # make a case for how many samples we need