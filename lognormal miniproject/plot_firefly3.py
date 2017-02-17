import json

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from network_simulator_and_scorer import NetworkHelper

#results_filestring = 'firefly cleanup 12-6-2016 simulations.json'
#networkconfig_filestring = 'firefly cleanup 12-6-2016 networkconfig.json'
#config_filestring = 'firefly cleanup 12-6-2016 config.json'

dir = 'simulations/2-10-2017/'
config_filestring = dir + '2-13-2017 config.json'
networkconfig_filestring = dir + '2-13-2017 networkconfig 3.json'
inputcurrents_filestring = dir + 'input_currents 2-3-2017.json'
results_filestring = dir + '2-13-2017 simulations 3.json'

verboseplot = True
simulatewinner = True

dimx = 3 # for the 2D cuts # todo could grab the names from the config file for the figure labels
dimy = 0


############ load the firefly simulations
results_file = open(results_filestring,'r')
with results_file as data_file:
    fireflyHistory = json.load(data_file)
results_file.close()

N_bugs = np.shape(fireflyHistory)[1] # reflecting the new save format
N_gen = np.shape(fireflyHistory)[0]
#indexSwitch = [[dict() for i_bug in range(N_bugs)] for i_gen in range(N_gen)]  # hacky fix - needed the transpose of this for the new save-format (todo fix the plotting to reflect the new format so we can skip this step)
indexSwitch = [[dict() for i_gen in range(N_gen)] for i_bug in range(N_bugs)]  # hacky fix - needed the transpose of this for the new save-format (todo fix the plotting to reflect the new format so we can skip this step)
for i in range(N_gen):
    for j in range(N_bugs):
        indexSwitch[j][i] = fireflyHistory[i][j]
fireflyHistory = indexSwitch

config_file = open(config_filestring,'r')
with config_file as data_file:
    config_data = json.load(data_file)
config_file.close()

PARAMS = config_data['PARAMS'] # ok to delete config_data at this point if you want

networkconfig_file = open(networkconfig_filestring,'r')
with networkconfig_file as data_file:
    networkconfig_data = json.load(data_file)
networkconfig_file.close()

N_bugs = np.shape(fireflyHistory)[0]
N_gen = np.shape(fireflyHistory)[1]  # todo this is going to be backwards
N_objectives = np.shape(fireflyHistory[0][0]['score'])[0]
N_params = config_data['N_params']
print 'N_bugs: ' , N_bugs , ' N_gen: ' , N_gen, ' N_scores ' , N_objectives
print 'lognormal sigma ' , networkconfig_data['logrand_sigma']
print 'lognormal mu ' , networkconfig_data['logrand_mu']



############## determine the winner

print 'shape of firefly history', np.shape(fireflyHistory)
# do this the stupid way for now because I don't understand iterators yet
allScores = np.zeros((N_bugs,N_gen))
bugIdxs = np.zeros(N_objectives, )
genIdxs = np.zeros(N_objectives, ) # init
for i_obj in range(0,N_objectives):
    print "objective " + str(i_obj) + " " # todo grab the names from OBJECTIVES in firefly config
    for i_gen in range(0,N_gen):
        for i_bug in range(0,N_bugs):
            allScores[i_bug, i_gen] = fireflyHistory[i_bug][i_gen]['score'][i_obj]  # don't need this later so it's ok to overwrite it
    bestOverall = np.nanargmax(allScores) # argmax thinks nans > inf  # todo why are there nans in here anyway?
    unraveledIdx = np.unravel_index(bestOverall,(N_bugs,N_gen))
    print 'best overall:', bestOverall,' score: ', allScores[unraveledIdx]
    bugIdxs[i_obj] = unraveledIdx[0]
    genIdxs[i_obj] = unraveledIdx[1] # keep these for simulation at the very end
    #print "best overall match? " + str(allScores[bugIdxs[i_obj]][genIdxs[i_obj]]) # test unraveling  # ok it's working
    print 'winning params: ', fireflyHistory[ int(bugIdxs[i_obj]) ][ int(genIdxs[i_obj]) ]['params']



# plot some of the progression (2D cut)
if verboseplot:

    num_flies_to_plot = 1

    plt.figure(0)
    for i_fly in range(num_flies_to_plot):  # todo select indices randomly
        xx = np.zeros((N_gen,))
        yy = np.zeros((N_gen,))

        colorList = iter(plt.cm.rainbow(np.linspace(0, 1, N_gen)))
        flyID = np.random.randint(0, N_bugs)
        for i_gen in range(N_gen):
            xx[i_gen] = fireflyHistory[flyID][i_gen]['params'][dimx]
            yy[i_gen] = fireflyHistory[flyID][i_gen]['params'][dimy]

            thisColor = next(colorList)

            plt.plot(fireflyHistory[flyID][i_gen]['params'][dimx],  # color code by generation
                     fireflyHistory[flyID][i_gen]['params'][dimy],
                     color=thisColor,marker='o')

            if i_gen < (N_gen-1):
                x1 = fireflyHistory[flyID][i_gen]['params'][dimx]
                x2 = fireflyHistory[flyID][i_gen+1]['params'][dimx]
                y1 = fireflyHistory[flyID][i_gen]['params'][dimy]
                y2 = fireflyHistory[flyID][i_gen+1]['params'][dimy]
                plt.plot([x1,x2],[y1,y2],linestyle='-',color=thisColor,alpha=0.4)

        #plt.plot(xx,yy,alpha=0.4) # continuous line for this fly

    xxlabel = PARAMS[dimx]
    yylabel = PARAMS[dimy]
    plt.xlabel(xxlabel)
    plt.ylabel(yylabel)
    plt.title('full history of a firefly')
    plt.show()


    f, axarr = plt.subplots(N_params, N_params)
    flyID = np.random.randint(0, N_bugs)
    for i_row in range(N_params):
        print "i_row = " + str(i_row) + "..."
        for i_col in range(N_params):
            if i_col >= i_row:

                xx = np.zeros((N_gen,))
                yy = np.zeros((N_gen,))


                colorList = iter(plt.cm.rainbow(np.linspace(0, 1, N_gen)))
                for i_gen in range(N_gen):
                    xx[i_gen] = fireflyHistory[flyID][i_gen]['params'][i_row]
                    yy[i_gen] = fireflyHistory[flyID][i_gen]['params'][i_col]

                    thisColor = next(colorList)

                    axarr[i_row, i_col].plot(fireflyHistory[flyID][i_gen]['params'][i_row],  # color code by generation
                             fireflyHistory[flyID][i_gen]['params'][i_col],
                             color=thisColor, marker='.')

                axarr[i_row,i_col].plot(xx, yy, alpha=0.15)  # continuous line for this fly


                xxlabel = PARAMS[i_row]
                yylabel = PARAMS[i_col]
                axarr[i_row, i_col].set_xlabel(xxlabel)
                axarr[i_row, i_col].set_ylabel(yylabel)
                axarr[i_row, i_col].set_title('a firefly path across every 2d param cut ')

    plt.show()



############# plot all information about a single firefly
if verboseplot:

    i_bug = flyID # the last random bug chosen above
    f, axarr = plt.subplots(6, sharex=True)
    yy = []
    for i in range(N_gen):
        yy.append(fireflyHistory[i_bug][i]['noise']) # bet there's a more elegant way to do this
    axarr[0].plot(range(N_gen),yy)
    axarr[0].set_ylabel('noise terms')

    yy = []
    for i in range(N_gen):
        yy.append(fireflyHistory[i_bug][i]['attraction']) # bet there's a more elegant way to do this
    axarr[1].plot(range(N_gen),yy)
    axarr[1].set_ylabel('attraction terms')

    yy = []
    for i in range(N_gen):
        yy.append(fireflyHistory[i_bug][i]['params']) # bet there's a more elegant way to do this
    axarr[2].plot(range(N_gen),yy)
    axarr[2].set_ylabel('params')

    yy = []
    for i in range(N_gen):
        yy.append(fireflyHistory[i_bug][i]['alpha']) # bet there's a more elegant way to do this
    axarr[3].plot(range(N_gen),yy)
    axarr[3].set_ylabel('alpha')

    yy = []
    for i in range(N_gen):
        yy.append(fireflyHistory[i_bug][i]['beta']) # bet there's a more elegant way to do this
    axarr[4].plot(range(N_gen),yy)
    axarr[4].set_ylabel('beta')

    yy = []
    for i in range(N_gen):
        yy.append(fireflyHistory[i_bug][i]['absorption']) # bet there's a more elegant way to do this
    axarr[5].plot(range(N_gen),yy)
    axarr[5].set_ylabel('absorption')

    plt.xlabel('generations')
    plt.show()

#  todo compare magnitude of attraction vs noise across parameters - seems like something is wrong with the scaling
if verboseplot:

    i_bug = flyID # the last random bug chosen above
    f, axarr = plt.subplots(4, sharex=True)

    # param 1
    yy1 = []
    yy2 = []
    yy3 = []
    for i in range(N_gen):
        yy1.append(fireflyHistory[i_bug][i]['attraction'][0])
        yy2.append(fireflyHistory[i_bug][i]['noise'][0])
        yy3.append(fireflyHistory[i_bug][i]['params'][0])
    axarr[0].plot(range(N_gen), yy1)
    axarr[0].plot(range(N_gen), yy2)
    axarr[0].plot(range(N_gen), yy3)
    axarr[0].set_ylabel('dimension 0')

    # param 2
    yy1 = []
    yy2 = []
    yy3 = []
    for i in range(N_gen):
        yy1.append(fireflyHistory[i_bug][i]['attraction'][1])
        yy2.append(fireflyHistory[i_bug][i]['noise'][1])
        yy3.append(fireflyHistory[i_bug][i]['params'][1])
    axarr[1].plot(range(N_gen), yy1)
    axarr[1].plot(range(N_gen), yy2)
    axarr[1].plot(range(N_gen), yy3)
    axarr[1].set_ylabel('dimension 1')

    # param 3
    yy1 = []
    yy2 = []
    yy3 = []
    for i in range(N_gen):
        yy1.append(fireflyHistory[i_bug][i]['attraction'][2])
        yy2.append(fireflyHistory[i_bug][i]['noise'][2])
        yy3.append(fireflyHistory[i_bug][i]['params'][2])
    axarr[2].plot(range(N_gen), yy1)
    axarr[2].plot(range(N_gen), yy2)
    axarr[2].plot(range(N_gen), yy3)
    axarr[2].set_ylabel('dimension 2')

    # param 4
    yy1 = []
    yy2 = []
    yy3 = []
    for i in range(N_gen):
        yy1.append(fireflyHistory[i_bug][i]['attraction'][3])
        yy2.append(fireflyHistory[i_bug][i]['noise'][3])
        yy3.append(fireflyHistory[i_bug][i]['params'][3])
    axarr[3].plot(range(N_gen), yy1)
    axarr[3].plot(range(N_gen), yy2)
    axarr[3].plot(range(N_gen), yy3)
    axarr[3].set_ylabel('dimension 3')

    plt.title('red: param val  blue: attraction term   green: noise term')
    plt.xlabel('generations')
    plt.show()



##################### plot distribution of scores

#paretoScores = np.zeros((N_objectives, N_bugs, N_gen))
for i_objective in range(0, N_objectives):
    print "obj " + str(i_objective) + " " # todo grab name automatically

    scores_inf_removed = [] # np.zeros((N_bugs*N_gen,))
    entry_counter = 0   # todo this doesn't seem right
    #for entry in fireflyHistory:
    for i_bug in range(N_bugs):
        for i_gen in range(N_gen):
            if ~np.isinf(fireflyHistory[i_bug][i_gen]['score'][i_objective]):
                if ~np.isnan(fireflyHistory[i_bug][i_gen]['score'][i_objective]):
                    #scores_inf_removed[entry_counter] = entry[i_gen]['score'][i_objective]
                    scores_inf_removed.append(fireflyHistory[i_bug][i_gen]['score'][i_objective])
                    entry_counter += 1
    #paretoScores[i_objective] = np.reshape(scores,(N_bugs,N_gen)) # todo check if this reshaping is correct
    print "num processed entries: " + str(entry_counter)

    if verboseplot:
        try:
            num_bins = 25
            plt.figure
            plt.hist(scores_inf_removed,num_bins,histtype='step')  # histogram(scores)
            tick = np.percentile(scores_inf_removed,90) # display upper quartile
            print "tick: " + str(tick)
            plt.plot([tick, tick],[0, 600],linestyle='-',linewidth=1,color='r')
            plt.title('objective ' + str(i_objective))
            plt.ylabel('log count')
            plt.yscale('log',nonposy='clip')
            plt.title('objective ' + str(i_objective))
            plt.show()
        except ValueError:
            print "error plotting hist. possiblly illegal range on scores"

    #minScore = np.nanmin(paretoScores[i_objective][~np.isnan(paretoScores[i_objective])]) # make sure we're not looking at nans
    print " distribution of scores after removing nan's and inf's: "
    minScore = np.min(scores_inf_removed)  # make sure we're not looking at nans
    print "min score " + str(minScore)
    #medianScore = np.nanmedian(paretoScores[i_objective][~np.isnan(paretoScores[i_objective])]) # going to use this below:
    medianScore = np.median(scores_inf_removed)
    #paretoScores[i_objective][~np.isnan(paretoScores[i_objective])])  # going to use this below:
    print "median score " + str(medianScore)
    #maxScore = np.max(paretoScores[i_objective][~np.isnan(paretoScores[i_objective])])
    maxScore = np.max(scores_inf_removed)
    print "max score " + str(maxScore)

    if verboseplot:   # plot the spatial extent

        # define a colormap m
        PRCTILE = 25
        MIN = np.percentile(scores_inf_removed,PRCTILE)  # warning - here we're using this inf_removed list but below we switch back to fireflyHistory
        MAX = np.max(scores_inf_removed)                    # todo organize this better
        print "lower cutoff value: " + str(MIN)
        norm = mpl.colors.Normalize(vmin=MIN, vmax=MAX)
        cmap = cm.rainbow
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        plt.figure()
        for i_gen in range(N_gen):
            for i_fly in range(N_bugs):
                if ~np.isinf(fireflyHistory[i_fly][i_gen]['score'][i_objective]):
                    if ~np.isnan(fireflyHistory[i_fly][i_gen]['score'][i_objective]):
                        xx = fireflyHistory[i_fly][i_gen]['params'][dimx]
                        yy = fireflyHistory[i_fly][i_gen]['params'][dimy]  # note; try color code by generation and fly
                        z = fireflyHistory[i_fly][i_gen]['score'][i_objective]
                        if z > MIN: # for better visibility, don't plot the bad samples
                            thisColor = m.to_rgba(z)
                            plt.plot(xx, yy, '.', color=thisColor,alpha=0.5)

        xxlabel = PARAMS[dimx]
        yylabel = PARAMS[dimy]
        plt.xlabel(xxlabel)
        plt.ylabel(yylabel)
        plt.title('scores of the fireflies for obj ' + str(i_obj))
        #a = np.linspace(-15, 1, 10).reshape(1, -1)
        #a = np.vstack((a, a))
        # plt.imshow(a, aspect='auto', cmap=plt.get_cmap(m), origin='lower')
        # todo get colorscale showing
        plt.show()


        # IN PROGRESS: display pairs of parameters
        TOP = np.percentile(scores_inf_removed, 90)
        MID = np.percentile(scores_inf_removed, 80)
        BOTTOM = np.percentile(scores_inf_removed, 10)
        f, axarr = plt.subplots(N_params, N_params)
        for i_row in range(N_params):
            print "i_row = " + str(i_row) + "..."
            for i_col in range(N_params):
                if i_col >= i_row:
                    for i_gen in range(N_gen):  # plot all the bugs you want to see
                        for i_fly in range(N_bugs):
                            if ~np.isinf(fireflyHistory[i_fly][i_gen]['score'][i_objective]):
                                if ~np.isnan(fireflyHistory[i_fly][i_gen]['score'][i_objective]):
                                    xx = fireflyHistory[i_fly][i_gen]['params'][i_row]
                                    yy = fireflyHistory[i_fly][i_gen]['params'][i_col]  # note; try color code by generation and fly
                                    z = fireflyHistory[i_fly][i_gen]['score'][i_objective]
                                    if z > TOP:
                                        thisColor = 'g'  # plot some rough contours
                                        axarr[i_row, i_col].plot(xx, yy, '.', color=thisColor,alpha=0.5)
                                    elif z > MID:
                                        thisColor = 'b'
                                        axarr[i_row, i_col].plot(xx, yy, '.', color=thisColor,alpha=0.5)
                                    elif z < BOTTOM:
                                        thisColor = 'r'
                                        axarr[i_row, i_col].plot(xx, yy, '.', color=thisColor,alpha=0.5)
                    xxlabel = PARAMS[i_row]
                    yylabel = PARAMS[i_col]
                    axarr[i_row, i_col].set_xlabel(xxlabel)
                    axarr[i_row, i_col].set_ylabel(yylabel)
                    axarr[i_row, i_col].set_title('scores of the fireflies for obj ' + str(i_obj))

        plt.show()
        # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
        #plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
        #plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
'''
######################## map of the solution space: ...
dimx = 0
dimy = 1

for i_obj in range(0,N_objectives):

    MIN = np.nanmedian(paretoScores[i_objective][~np.isnan(paretoScores[i_objective])])   # chop off some of the range so the plot is easier to read
    MAX = np.max(paretoScores[i_objective][~np.isnan(paretoScores[i_objective])])
    norm = mpl.colors.Normalize(vmin=MIN, vmax=MAX)
    cmap = cm.rainbow
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    plt.figure()
    for i_gen in range(N_gen):
        for i_fly in range(N_bugs):
            xx = fireflyHistory[i_fly][i_gen]['params'][dimx]  # todo color code by generation and fly
            yy = fireflyHistory[i_fly][i_gen]['params'][dimy]  # todo color code by generation and fly
            z = fireflyHistory[i_fly][i_gen]['score'][i_obj] # just the first score
            thisColor = m.to_rgba(z)
            plt.plot(xx,yy,'o',color=thisColor)

    xxlabel = PARAMS[dimx]
    yylabel = PARAMS[dimy]
    plt.xlabel(xxlabel)
    plt.ylabel(yylabel)
    plt.title('scores of the fireflies for obj ' + str(i_obj))
    a = np.linspace(-15,1,10).reshape(1,-1)
    a = np.vstack((a,a))
    #plt.imshow(a, aspect='auto', cmap=plt.get_cmap(m), origin='lower')
        # todo get colorscale showing
    plt.show()

'''

###################### plot score trajectories
plt.figure()
for i_dim in range(0,N_objectives):
    # todo add score vs time plots
    trace = np.zeros((N_gen,))
    for i_bug in range(N_bugs):
        for i_gen in range(N_gen):
            trace[i_gen] = fireflyHistory[i_bug][i_gen]['score'][i_dim]
        plt.plot(trace,alpha=0.3)
    #plt.yscale('log', nonposy='clip')
    plt.title('score trajectories for dim ' + str(i_dim))
    plt.show()



# plot the best example a few times
if simulatewinner:

    print 'simulating winning fly'
    network_helper = NetworkHelper(networkconfig_filestring)
    if network_helper.__class__.cell_inputs == None:
        network_helper.initializeInputs(inputcurrents_filestring)


    for i_obj in range(0,N_objectives):
        print "objective " + str(i_obj)
        print 'first trial = '
        network_helper.simulateActivity(fireflyHistory[ int(bugIdxs[i_obj]) ][ int(genIdxs[i_obj]) ]['params'], verboseplot=True)

        print 'second trial = '
        network_helper.simulateActivity(fireflyHistory[ int(bugIdxs[i_obj]) ][ int(genIdxs[i_obj]) ]['params'], verboseplot=True)

