import json
import time

import numpy as np

from firefly_pack.firefly_dynamics_rescaled import firefly_dynamics_rescaled
from firefly_pack.network_simulator_and_scorer import NetworkHelper


#@profile
def run_firefly1():
    dir = 'simulations/3-2-2017/'
    config_filestring = dir + '2-28-2017 80x60  config.json'
    networkconfig_filestring = dir + '2-28-2017 20x150  networkconfig 3.json'
    inputcurrents_filestring = dir + '2-28-2017 inputs.json'
    results_postfix = ' simulations 3.json'


    ############### load the firefly config file
    config_file = open(config_filestring,'r')
    with config_file as data_file:
        firefly_config = json.load(data_file)
    config_file.close()

    print('CONFIG FILE: ' + firefly_config['verbose'])
    print('from ' + config_filestring)

    save_prefix = firefly_config['config_prefix']
    results_f_name = dir + save_prefix + results_postfix

    N_gen = firefly_config['N_gen']  # this is kind of dumb I realize , but hopefully worth it for readability
    N_bugs = firefly_config['N_bugs']
    N_params = firefly_config['N_params']
    N_objectives = firefly_config['N_objectives']

    # range for rosenbrock
    #MEANS = firefly_config['MEANS']
    #STDS = firefly_config['STDS']
    MAXES = firefly_config['MAXES']
    MINS = firefly_config['MINS']

    characteristic_scales = np.array(firefly_config['characteristic_scales']) # note this gets saved as a list (for serialization)

    alpha = firefly_config['alpha'] # NOTE alpha gets scaled by char scale for each param in Firefly Dynamics function
    beta = firefly_config['beta']  # >4 yields chaotic firefly dynamics
    #print "beta t0 : " + str(beta)
    absorption = firefly_config['absorption'] # somewhere around 0.5 is good according to Yang

    annealing_constant = firefly_config['annealing_constant'] # currently only beta is being annealed

    ############ initializations
    # population = sp.randn(N_params, N_bugs)
    population = np.random.rand(N_params, N_bugs)
    #for i_bug in range(N_bugs):
    for i_param in range(N_params):
        # population[:,i_bug] *= STDS   # gaussian scatter around the means using the stds
        # population[:,i_bug] += MEANS]
        #population[i_param, i_bug] *= (MAXES[i_param] - MINS[i_param])
        #population[i_param, i_bug] += MINS[i_param]
        population[i_param,:] = np.linspace(MINS[i_param],MAXES[i_param],N_bugs) # Yang stresses that even spacing is important


    # check bounds on parameter values (NOTE shouldn't have to do this here in the initialization step now that it's switched to a uniform dist)
    '''
    for i_param in range(N_params):
        for i_fly in range(N_bugs):
            if population[i_param,i_fly] < MINS[i_param]:
                population[i_param,i_fly] = MINS[i_param]
            if population[i_param,i_fly] > MAXES[i_param]:
                population[i_param,i_fly] = MAXES[i_param]
    print 'initial population: ', population
    '''

    scoreVectors = np.zeros((N_bugs, N_objectives))
    attractionTerms = np.zeros((N_bugs, N_params))
    noiseTerms = np.zeros((N_bugs, N_params))

    oneGen = [dict() for i_bug in range(N_bugs)] # write this dictionary to the json simulations file after each generation

    resultsFile = open(results_f_name, 'w')
    resultsFile.write("[") # for formatting a valid json object

    network_helper = NetworkHelper(networkconfig_filestring) # this object does all the simulation work
    if network_helper.__class__.cell_inputs == None: # if not yet initialized
        network_helper.initializeInputs(inputcurrents_filestring)  # initialize inputs as a global static variable - currently this still gets copied when it gets made into a timed array

    startTime = time.time()

    ################ run the firefly algorithm
    for i_gen in range(0, N_gen):

        print 'generation: ' , i_gen

        # handle meta-heuristics (right now we are just annealing beta)
        beta *= annealing_constant
        alpha *= annealing_constant # let's anneal alpha too
        #print "annealing constant: " + str(annealing_constant)
        #print "beta updated: " + str(beta)

        for i_fly in range(0, N_bugs): # old note: better to enumerate the firebugs directly
            scoreVectors[i_fly,:] = network_helper.simulateActivity(population[:,i_fly],verboseplot=False)
            #scoreVectors[i_fly, 1] = rosenbrock_obj(population[:, i_fly]) # temp just use the same obj for both


        result = firefly_dynamics_rescaled(population, scoreVectors, alpha, beta, absorption, characteristic_scales,  # todo could remove characteristic scales
                                               MAXES, MINS)

        newPopulation = result['newPopulation']
        attractionTerms = result['attractionTerms']
        noiseTerms = result['noiseTerms']
        population = newPopulation
        updateCounts = result['dominatedByOthers'].tolist()

        # check bounds on parameter values
        for i_param in range(N_params):
            for i_fly in range(N_bugs):
                if population[i_param,i_fly] < MINS[i_param]:
                    population[i_param,i_fly] = MINS[i_param]
                if population[i_param,i_fly] > MAXES[i_param]:
                    population[i_param,i_fly] = MAXES[i_param]




        print 'pareto IDs : ' + str(result['paretoIDs'])
        print 'cull IDs : ' + str(result['cullIDs'])

        # keep track of progress for plotting etc
        for i_fly in range(0,N_bugs):
            oneGen[i_fly] = {'noise':np.copy(noiseTerms[i_fly,:]).tolist(),'attraction':np.copy(attractionTerms[i_fly,:]).tolist(),
                                            'alpha':alpha,'beta':beta,'absorption':absorption,
                                            'score':np.copy(scoreVectors[i_fly,:]).tolist(),'params':np.copy(population[:,i_fly]).tolist(),
                                            'gen':i_gen,'fly':i_fly,'updateCounts':updateCounts} # ,'paretoIDs':paretoIDs,'cullIDs':cullIDs}
                                            # NOTE could maybe improve efficiency here...do we need to copy?

        json.dump(oneGen, resultsFile, sort_keys=True, indent=2)
        if i_gen < (N_gen - 1):
            resultsFile.write(",")

        # todo keep track of the pareto front

    ############### save the simulations for plotting and post-hoc analysis

    print 'elapsed time for firefly alg: ', time.time() - startTime, ' seconds'

    resultsFile.write("]")
    resultsFile.close()

run_firefly1()