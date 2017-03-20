import json
import numpy as np
# todo make this more elaborate

def get_exemplars(simulation_name, sim_configfile_name, firefly_configfile_name, N_exemplars=3, plot_verbose=True):



    # temporary quick solution - just grab the 5 best scores and use those params
    ############ load the firefly simulations
    results_file = open(simulation_name, 'r')
    with results_file as data_file:
        fireflyHistory = json.load(data_file)
    results_file.close()

    N_bugs = np.shape(fireflyHistory)[1]  # reflecting the new save format
    N_gen = np.shape(fireflyHistory)[0]
    # indexSwitch = [[dict() for i_bug in range(N_bugs)] for i_gen in range(N_gen)]  # hacky fix - needed the transpose of this for the new save-format (todo fix the plotting to reflect the new format so we can skip this step)
    indexSwitch = [[dict() for i_gen in range(N_gen)] for i_bug in range(
        N_bugs)]  # hacky fix - needed the transpose of this for the new save-format (todo fix the plotting to reflect the new format so we can skip this step)
    for i in range(N_gen):
        for j in range(N_bugs):
            indexSwitch[j][i] = fireflyHistory[i][j]
    fireflyHistory = indexSwitch

    config_file = open(firefly_configfile_name, 'r')
    with config_file as data_file:
        config_data = json.load(data_file)
    config_file.close()

    PARAMS = config_data['PARAMS']  # ok to delete config_data at this point if you want

    networkconfig_file = open(sim_configfile_name, 'r')
    with networkconfig_file as data_file:
        networkconfig_data = json.load(data_file)
    networkconfig_file.close()

    N_bugs = np.shape(fireflyHistory)[0]
    N_gen = np.shape(fireflyHistory)[1]  # todo this is going to be backwards
    N_objectives = np.shape(fireflyHistory[0][0]['score'])[0]
    N_params = config_data['N_params']
    print 'N_bugs: ', N_bugs, ' N_gen: ', N_gen, ' N_scores ', N_objectives
    print 'lognormal sigma ', networkconfig_data['logrand_sigma']
    print 'lognormal mu ', networkconfig_data['logrand_mu']

    # find the K winners
        # for now just concatenate the obj functions - but ideally we want to combine them somehow, or use the pareto front or something todo
    exemplar_idxs = np.zeros((N_objectives*N_exemplars,2))  # N_obj x N_exemp x (i_bug, i_gen)
    exemplar_params = np.zeros((N_objectives*N_exemplars,N_params))

    # copied from plot_firefly_fun:
    print 'shape of firefly history', np.shape(fireflyHistory)
    # do this the stupid way for now because I don't understand iterators yet
    allScores = np.zeros((N_bugs, N_gen))
    bugIdxs = np.zeros(N_objectives, )
    genIdxs = np.zeros(N_objectives, )  # init
    for i_obj in range(0, N_objectives):

        for i_gen in range(0, N_gen):
            for i_bug in range(0, N_bugs):
                allScores[i_bug, i_gen] = fireflyHistory[i_bug][i_gen]['score'][i_obj]  # read these into a more convenient format

        for i_winner in range(N_exemplars):

            print "objective " + str(i_obj) + " "  # todo grab the names from OBJECTIVES in firefly config

            bestOverall = np.nanargmax(allScores)  # argmax thinks nans > inf  # todo why are there nans in here anyway?
            unraveledIdx = np.unravel_index(bestOverall, (N_bugs, N_gen))
            print 'best overall:', bestOverall, ' score: ', allScores[unraveledIdx]
            bugIdxs[i_obj] = unraveledIdx[0]
            genIdxs[i_obj] = unraveledIdx[1]  # keep these for simulation at the very end
            # print "best overall match? " + str(allScores[bugIdxs[i_obj]][genIdxs[i_obj]]) # test unraveling  # ok it's working
            print bestOverall
            print np.shape(allScores.flatten())
            print 'winning params: ', fireflyHistory[int(bugIdxs[i_obj])][int(genIdxs[i_obj])]['params']

            print np.shape(exemplar_idxs)
            this_idx = i_obj*N_exemplars + i_winner
            exemplar_idxs[this_idx][0] = int(bugIdxs[i_obj])
            exemplar_idxs[this_idx][1] = int(genIdxs[i_obj])
            #exemplar_idxs[i_obj][i_winner][0] = int(bugIdxs[i_obj])
            #exemplar_idxs[i_obj][i_winner][1] = int(genIdxs[i_obj])
            allScores[bugIdxs[i_obj]][genIdxs[i_obj]] = np.nan # mask this winner out so the next iteration finds a different solution

            for i_param in range(N_params):
                winning_bug = int(bugIdxs[i_obj])
                winning_gen = int(genIdxs[i_obj])
                paramval = fireflyHistory[winning_bug][winning_gen]['params'][i_param]
                exemplar_params[this_idx][i_param] = paramval

    if plot_verbose:
        print 'plotting under construction (todo)'
        # todo
        # show the 2D cuts and plot the max scores on top
        # ultimately this would be cool to do with 2D cut contour maps, with exemplars colorcoded on top

    return exemplar_params
    #return exemplar_idxs