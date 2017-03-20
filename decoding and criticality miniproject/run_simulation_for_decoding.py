from firefly_pack.network_simulator_and_scorer import NetworkHelper
import time

def run_simulation_for_decoding(networkconfig_filestring,inputcurrents_filestring,save_filepath,params):


    network_helper = NetworkHelper(networkconfig_filestring)  # this object does all the simulation work
    if network_helper.__class__.cell_inputs == None:  # if not yet initialized
        network_helper.initializeInputs(inputcurrents_filestring)  # initialize inputs as a global static variable - currently this still gets copied when it gets made into a timed array

    startTime = time.time()

    network_helper.simulate_activity_for_decoding(params, save_filepath)

    endTime = time.time()
    return endTime - startTime

