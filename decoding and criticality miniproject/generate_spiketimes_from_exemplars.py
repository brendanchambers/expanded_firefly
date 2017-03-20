import numpy as np
from generate_inputs import generate_inputs
from run_simulation_for_decoding import run_simulation_for_decoding

network_config_filepath = '../lognormal miniproject/simulations/3-2-2017/2-28-2017 20x150  networkconfig 1.json'
input_config_filepath = '../lognormal miniproject/simulations/3-2-2017/2-28-2017 inputs.json'

analysis_dir = 'analyses/3-20-2017/spiketimes/'
exemplar_file_path = analysis_dir + '../exemplars.npy'
exemplars = np.load(exemplar_file_path)

N_repeats = 1000

N_sim_families = 1 # np.shape(exemplars)[0] # temp, just look at a single model-type
N_inputs = 2 # number of unique inputs to generate
N_winners = 1 # np.shape(exemplars)[1]
#N_params = np.shape(exemplars)[2]

# plot parameters for each simulation family todo

# compare parameters for each simulation family todo



# run some test activity

    # generate inputs
for i in range(N_inputs):
    name_postfix = 'input ' + str(i) + '.json'
    generate_inputs(analysis_dir,name_postfix)

for i_input in range(N_inputs):
    for i_sim_family in range(N_sim_families):
        for i_winner in range(N_winners):

            params = exemplars[i_sim_family][i_winner][:]
            #input_filepath = input_config_filepath # testing the original input
            input_filepath = analysis_dir + 'input ' + str(i_input) + '.json'

            for i_repeat in range(N_repeats):
                save_filepath = analysis_dir + 'family ' + str(i_sim_family) + ' input ' + str(i_input) + ' winner ' + str(i_winner) + ' ' + str(i_repeat) # todo postfix?
                running_time = run_simulation_for_decoding(network_config_filepath,input_filepath,save_filepath,params) # todo
                print 'finished input ' + str(i_input) + ' family ' + str(i_sim_family) + ' winner ' + str(i_winner) + ' in ' +  str(running_time) + ' seconds'




# check dynamics and compare distributions (consider box plots) todo
