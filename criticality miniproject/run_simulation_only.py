import time

from firefly_pack.network_simulator_and_scorer import NetworkHelper

# switches:
network_config_filename = 'simulations/2-16-2017/2-16-2017 networkconfig 1.json'
inputcurrents_filestring = 'simulations/2-16-2017/input_currents 2-16-2017.json'

# init
network_helper = NetworkHelper(network_config_filename) # this object does all the simulation work

if network_helper.__class__.cell_inputs == None: # if not yet initialized
    network_helper.initializeInputs(inputcurrents_filestring)  # initialize inputs as a global static variable - currently this still gets copied when it gets made into a timed array

startTime = time.time()

# a clean pipeline omitting the firefly stuff
network_params = [0.395, 0.18, 0.285, 1, 0.37, 0.225] # p_ei, p_ie, p_ii, w_input, p_inpi, p_iinp  # tau_e > tau_i
#network_params = [0.25, 0.12, 0.19, 1.5, 0.34, 0.18] # p_ei, p_ie, p_ii, w_input, p_inpi, p_iinp  # tau_e = tau_i

score_vector = network_helper.simulateActivity(network_params,verboseplot=True)

print "score_vector: " + str(score_vector)


stopTime = time.time()
running_time = stopTime - startTime
print "running time: " + str(running_time)







#### todo try checking measuring zipf's law