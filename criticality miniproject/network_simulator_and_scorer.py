import json
import numpy as np
from brian2 import *
import matplotlib.pyplot as plt
import time
import scipy.ndimage.filters as filt
import scipy.stats as stats
from brian2.units.allunits import *

from memory_profiler import profile

# in this version of our quest, only lognorm_sigma and lognorm_mu will be changing
# function of arguments lognorm_sigma and lognorm_mu


#networkconfig_filestring =

class NetworkHelper:

    cell_inputs = None # intially null - set after construction so that we know N_e, N, etc

    def __init__(self, networkconfig_filestring):

        # load the networkconfig file
        networkconfig_file = open(networkconfig_filestring, 'r')
        with networkconfig_file as data_file:
            network_config = json.load(data_file)
        networkconfig_file.close()

        print('network config file from ' + networkconfig_filestring)

        self.N_input = network_config['N_input'] # WARNING this needs to match N_target in the input gen file (todo fix this issue)
        print "N_input " + str(self.N_input)
        self.N_e = network_config['N_e']
        self.N_i = network_config['N_i']
        self.N = network_config['N']

        parsed = network_config['duration'].split(' ') # read in string representation
        print str(parsed[0]) + ' * ' + str(parsed[1])
        self.duration = eval(str(parsed[0]) + ' * ' + str(parsed[1])) # convert to physical units
        parsed = network_config['input_duration'].split(' ')
        self.input_duration = eval(parsed[0] + ' * ' + parsed[1])
        #parsed = network_config['input_rate'].split(' ')
        #self.input_rate = eval(parsed[0] + ' * ' + parsed[1])

        parsed = network_config['initial_Vm']
        print 'initial voltage distribution rule: ' + parsed
        self.initial_Vm = eval(parsed)

        parsed = network_config['C'].split(' ')
        self.C = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['gL'].split(' ')
        self.gL = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['taum'].split(' ')
        self.taum = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['EL'].split(' ')
        self.EL = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['VT'].split(' ')
        self.VT = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['DeltaT'].split(' ')
        self.DeltaT = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['Vcut'].split(' ')
        self.Vcut = eval(parsed[0] + ' * ' + parsed[1])

        #parsed = network_config['w_input'].split(' ')
        #self.w_input = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['we'].split(' ')
        self.we = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['wi'].split(' ')
        self.wi = eval(parsed[0] + ' * ' + parsed[1])

        # todo the non-physical units
        #self.p_connect_input = network_config['p_connect_input']
        self.p_connect_ee = network_config['p_connect_ee']
        #self.p_connect_ie = network_config['p_connect_ie']
        #self.p_connect_ei = network_config['p_connect_ei']
        #self.p_connect_ii = network_config['p_connect_ii']

        self.logrand_sigma = network_config['logrand_sigma']
        self.logrand_mu = network_config['logrand_mu']
        self.LOG_RAND_sigmaInh =  network_config['LOG_RAND_sigmaInh']
        self.LOG_RAND_muInh = network_config['LOG_RAND_muInh']

        parsed = network_config['tauw'].split(' ')
        self.tauw = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['a'].split(' ')
        self.a = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['b'].split(' ')
        self.b = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['Vr'].split(' ')
        self.Vr = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['EE'].split(' ')
        self.EE = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['EI'].split(' ')
        self.EI = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['taue'].split(' ')
        self.taue = eval(parsed[0] + ' * ' + parsed[1])
        parsed = network_config['taui'].split(' ')
        self.taui = eval(parsed[0] + ' * ' + parsed[1])

        self.eqs = Equations(str(network_config['eqs'])) # dynamics of the model


    def initializeInputs(self, input_filestring):

        input_file = open(input_filestring, 'r') # load in inputs
        with input_file as data_file:
            input_info = json.load(data_file)
        input_file.close()
        input_conductances = np.asarray(input_info['input_conductances'])
        N_targets = input_info['N_targets'] # WARNING this needs to match N_inputs in the network config file (todo fix this issue)


        #block_duration = 150 * ms  # todo this should all be set in the network config file
        block_duration = 300*ms
        input_duration = 50 * ms
        t_res = defaultclock.dt
        total_blocks = int(np.ceil(self.duration * 1.0 / block_duration))
        total_steps = (self.duration / t_res) + 1
        steps_per_block = (block_duration / t_res) + 1
        num_input_steps = (input_duration / t_res) + 1
        NetworkHelper.cell_inputs = np.zeros((total_steps, self.N))
        cur_step = 0 # index into the timed array (row)
        for i_blocks in range(total_blocks):

            if i_blocks == total_blocks - 1:   # if this is the last block and the math doesn't work out, handle the truncation:
                steps_remaining = total_steps - cur_step + 1
                if steps_remaining < num_input_steps:
                    NetworkHelper.cell_inputs[cur_step:,:N_targets] = input_conductances[:][:steps_remaining].T  # todo generate random indices for the targets
                else:
                    NetworkHelper.cell_inputs[cur_step:(cur_step + num_input_steps),:N_targets] = input_conductances.T
            else:

                NetworkHelper.cell_inputs[cur_step:(cur_step + num_input_steps),:N_targets] = input_conductances.T
                cur_step += steps_per_block

        NetworkHelper.cell_inputs[:, N_targets:] = 0  # set exc-non-input and inhibitory to zero for impinging input conductance # todo don't need this now
            # todo should probably randomly assign instead of using first N_target entries - this could be more important for the small world topologies



    #@profile
    def  simulateActivity(self, input_args, verboseplot=False):
        p_ei = input_args[0]
        p_ie = input_args[1]
        p_ii = input_args[2]
        w_input = input_args[3] * nS
        p_inhi = input_args[4] # todo consider ordering these in a more logical way at some point
        p_iinh = input_args[5]
        ########### define the neurons and connections
        logrand_mu = log(1) - 0.5*(self.logrand_sigma**2) # this establishes mean(W) = 1, regardless of sigma

        # having trouble with the self stuff in passing string arguments to Brian2
        #  so just pursue the course of least resistance - rename the variables NOTE is there a better way

        initial_Vm = self.initial_Vm

        C = self.C
        gL = self.gL
        taum = self.taum
        EL = self.EL
        VT = self.VT
        DeltaT = self.DeltaT
        Vcut = self.Vcut

        tauw = self.tauw
        a = self.a
        b = self.b
        Vr = self.Vr
        EE = self.EE
        EI = self.EI
        taue = self.taue
        taui = self.taui

        #w_input = self.w_input
        we = self.we
        wi = self.wi

        eqs = self.eqs  # dynamics of the model

        g_input_timedArray = TimedArray(NetworkHelper.cell_inputs * w_input, dt=defaultclock.dt)  # w_input is in nS, giving these the correct units
        neurons = NeuronGroup(self.N,
                              model=eqs, threshold='vm>Vcut', reset="vm=Vr; w+=b",
                              refractory=1 * ms, method='rk4')

        # Defining the subgroups i.e. (e_input e i)
        P_input = neurons[:self.N_input] # the new input population (which is a subset of the exc population) # todo add N_inh
        Pe = neurons[self.N_input:self.N_e]  # non-inh subset of the excitatory subpopulation
                    # so, Pe has size (N_e - N_input) + 1
        Pi = neurons[self.N_e:self.N] # inh neurons

        # DEFINE THE SYNAPSE GROUPS
        # exc -> exc
        C_inp_inp = Synapses(P_input, P_input, model='''alpha : 1''', on_pre='gE+=(alpha*we)')   # treat all excitatory connections the same:
        C_inp_e = Synapses(P_input, Pe, model='''alpha : 1''', on_pre='gE+=(alpha*we)')
        C_e_inp = Synapses(Pe, P_input, model='''alpha : 1''', on_pre='gE+=(alpha*we)')
        Cee = Synapses(Pe, Pe, model='''alpha : 1''', on_pre='gE+=(alpha*we)') # note: we will set alpha using the lognorm draws
        # exc -> inh
        C_inp_i = Synapses(P_input, Pi, model='''alpha : 1''', on_pre='gE+=(alpha*we)')
        Cei = Synapses(Pe, Pi, model='''alpha : 1''', on_pre='gE+=(alpha*we)')
        # inh -> exc
        C_i_inp = Synapses(Pi, P_input, model='''alpha : 1''', on_pre='gI+=(alpha*wi)')
        Cie = Synapses(Pi, Pe, model='''alpha : 1''', on_pre='gI+=(alpha*wi)')
        # inh -> inh
        Cii = Synapses(Pi, Pi, model='''alpha : 1''', on_pre='gI+=(alpha*wi)')

        # DEFINE THE BINARY TOPOLOGY
        # all exc connections
        C_inp_inp.connect(p=self.p_connect_ee,condition='i!=j')   # no autapses      NOTE (additional argument we might want: condition='i!=j') # note - we can also specify vectors Pre and Post, so that Pre(i) -> Post(i) ... e.g. Cee.connect(i=Pre, j=Post)
        C_inp_e.connect(p=self.p_connect_ee)
        C_e_inp.connect(p=self.p_connect_ee)
        Cee.connect(p=self.p_connect_ee,condition='i!=j') # no autapses

                                                        #Cei.connect(p=self.p_connect_ei)  # artifact from when we weren't optimizing over these params
                                                        #Cii.connect(p=self.p_connect_ii)
                                                        #Cie.connect(p=self.p_connect_ie)
        C_inp_i.connect(p=p_inhi)
        Cei.connect(p=p_ei)

        C_i_inp.connect(p=p_iinh)
        Cie.connect(p=p_ie) # passed in as argument

        Cii.connect(p=p_ii,condition='i!=j')

        # GET THE SIZES OF THESE FOR THE LOGNORM DRAW
        N_inp_inp = len(C_inp_inp)
        N_inp_e = len(C_inp_e)
        N_e_inp = len(C_e_inp)
        N_ee = len(Cee)
        N_inp_i = len(C_inp_i)
        N_ei = len(Cei)
        N_i_inp = len(C_i_inp)
        N_ie = len(Cie)
        N_ii = len(Cii)

        # DEFINE THE DISTRIBUTION OF WEIGHTS
        # todo do this in a loop instead of one big draw, b/c it's a killer on memory -- todo run speed comparisons
        C_inp_inp.alpha = numpy.random.lognormal(self.logrand_mu, self.logrand_sigma, N_inp_inp)
        C_inp_e.alpha = numpy.random.lognormal(self.logrand_mu, self.logrand_sigma, N_inp_e)
        C_e_inp.alpha = numpy.random.lognormal(self.logrand_mu, self.logrand_sigma, N_e_inp)
        Cee.alpha = numpy.random.lognormal(self.logrand_mu, self.logrand_sigma, N_ee)

        C_inp_i.alpha = numpy.random.lognormal(self.logrand_mu, self.logrand_sigma, N_inp_i)
        Cei.alpha = numpy.random.lognormal(self.logrand_mu, self.logrand_sigma, N_ei)

        C_i_inp.alpha = numpy.random.lognormal(self.logrand_mu, self.logrand_sigma, N_i_inp)
        Cie.alpha = numpy.random.lognormal(self.LOG_RAND_muInh, self.LOG_RAND_sigmaInh, N_ie)

        Cii.alpha = numpy.random.lognormal(self.LOG_RAND_muInh, self.LOG_RAND_sigmaInh, N_ii)  # NOTE mean of these distributions = 1




        ################# run the simulation ####################
        ###### initialization
        neurons.vm = self.initial_Vm
        neurons.gE = 0 * nS
        neurons.gI = 0 * nS
        # neurons.I = input_current * nA # current injection during input period

        ###### recording activity
        s_mon = SpikeMonitor(neurons)  # keep track of population firing
        #P_patch = neurons[(self.N_e - 1):(self.N_e + 1)]  # random exc and inh cell #todo chose these randomly
        P_patch = neurons[0:2] # todo # go back to patching an inhibitory neuron too
        dt_patch = 0.1 * ms  # 1/sampling frequency  in ms
        patch_monitor = StateMonitor(P_patch, variables=('vm', 'gE', 'gI', 'w', 'g_input'), record=True,
                                     dt=dt_patch)  # keep track of a few cells
        #inputRate_monitor = PopulationRateMonitor(input_units)

        ###### stimulation period
        #run(self.input_duration)
        #run(self.input_duration)

        ######  non-stimulation period  % todo define some input connectivity
        # neurons.I = 0 * nA # current injection turned off
        # input_units = PoissonGroup(N_input, 0*Hz) # turn off input firing
        # Cinput = Synapses(input_units, Pe, on_pre = 'gE+=w_input')
        # Cinput.connect(p=p_connect_input)

        # input_units = PoissonGroup(N_input, 0*Hz) # silence the inputs - this doesn't work
        # (# note - looks like the TimedArray Brian2 class has potential to be a better solution for dynamic inputs)
        run(self.duration) #  - self.input_duration)  # make everything go

        # device.build(directory='output', compile=True, run=True, debug=False) # for the c++ build (omitting during testing)
        # todo try to get this back in if we want it to run faster








        ################ plotting    todo don't actually need this here    ####################
        if verboseplot:
            ##### spiking activity
            plt.figure
            plt.plot(s_mon.t / ms, s_mon.i, '.k')
            plt.xlabel('Time (ms)')
            plt.ylabel('Neuron index')
            plt.title('population spiking')
            plt.show()

            ####  example excitatory neuron
            tt = patch_monitor.t / ms
            f, axarr = plt.subplots(5, sharex=True)
            plt.title('an excitatory neuron')
            axarr[0].plot(tt, patch_monitor[0].vm, 'k')
            axarr[0].set_ylabel('Vm')
            axarr[1].plot(tt, patch_monitor[0].gE, 'b')
            axarr[1].set_ylabel('gE')
            axarr[2].plot(tt, patch_monitor[0].gI, 'r')
            axarr[2].set_ylabel('gI')
            axarr[3].plot(tt, patch_monitor[0].w, 'ro')
            axarr[3].set_ylabel('adaptation w')
            #axarr[4].plot(tt, inputRate_monitor.smooth_rate(width=5 * ms) / Hz, 'k')
            axarr[4].plot(tt, patch_monitor[0].g_input,'k')
            axarr[4].set_ylabel('g_input')
            plt.xlabel('time (ms)')
            plt.show()

            #####  example inhibitory neuron
            tt = patch_monitor.t / ms
            f, axarr = plt.subplots(5, sharex=True)
            plt.title('an inhibitory neuron')
            axarr[0].plot(tt, patch_monitor[1].vm, 'k')
            axarr[0].set_ylabel('Vm')
            axarr[1].plot(tt, patch_monitor[1].gE, 'b')
            axarr[1].set_ylabel('gE')
            axarr[2].plot(tt, patch_monitor[1].gI, 'r')
            axarr[2].set_ylabel('gI')
            axarr[3].plot(tt, patch_monitor[1].w, 'ro')
            axarr[3].set_ylabel('adaptation w')
            #axarr[4].plot(tt, inputRate_monitor.smooth_rate(width=5 * ms) / Hz, 'k')
            axarr[4].plot(tt, patch_monitor[1].g_input, 'k')
            axarr[4].set_ylabel('g_input')
            plt.xlabel('time (ms)')
            plt.show()







############################################################################################
#        scoring
############################################################################################

        do_score_duration = False  # just to make it a little easier to play with which obj functions we include
        do_score_rate = True
        do_score_asynchrony = False

        ############### reformat the spike times, bin, and smooth

        # todo smooth? subsample? (subsampling might be cheaper but smoothing so that bin size doesn't affect the score would be nice)

        # BIN SIZE
        binwidth = 1 # ms
        numBins = np.ceil((self.duration/ms) / binwidth)+1 #  https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.convolve1d.html
        raster = np.zeros((self.N_e,numBins))

        for i_spike in range(0,len(s_mon.i)):
            cellIdx = s_mon.i[i_spike]
            if cellIdx < self.N_e: # only look at exc neurons
                binIdx = np.round((s_mon.t[i_spike] / ms) / binwidth)
                raster[cellIdx][binIdx] += 1.0  #
        raster = raster * (1.0/(binwidth/1000.0)) # convert to Hz # todo this is still wonky

        avgRates = np.nansum(raster,1) / (self.duration/ms / 1000) # avg rates in Hz during the trial (remove ms units and convert to s)
        numActiveNeurons = np.sum(np.where(avgRates > 0.01,1,0)) # previously used number of active neurons but currently ignoring this

        smoothSigma = 3  # ms
        sigmaBins = smoothSigma / binwidth # warning will this be a problem if it's not an integer? round if necessary
        raster_smooth = filt.gaussian_filter1d(raster,sigma=smoothSigma,axis=1) # todo should smooth BEFORE rebinning
        sumSmoothRate = np.sum(raster_smooth,0) # todo set this up relative to N_e

        meanSmoothRate = (1./self.N_e)*sumSmoothRate # note, does the division make f.p. roundoff worse?

        IGNITION_THRESH = 0.5  # avg firing rate (Hz) among active cells -> to count as an ignition
        QUENCH_THRESH = 0.5
        PAROXYSM_THRESH = 15 # maximum allowed rate
                    #   note triple check that these are normalized correctly with the smoothing kernel


        # redesigning this - count total number of bins with stable firing
        stableBins = []
        for idx,val in enumerate(meanSmoothRate):
            if val > QUENCH_THRESH:
                if val < PAROXYSM_THRESH:
                    stableBins.append(idx)

        numStableBins = len(stableBins)
        if do_score_duration:
            stable_duration_score = numStableBins * binwidth # total stable duration

        activeBins = []
        for idx, val in enumerate(meanSmoothRate):
            if val >= PAROXYSM_THRESH:
                activeBins.append(idx)
        numActiveBins = len(activeBins)

        # add information about sum square rates
        maxPossibleSpiking = PAROXYSM_THRESH * PAROXYSM_THRESH * numStableBins  # get max possible sum square spiking
        if do_score_rate:
            totalSpiking = 0
            for idx in activeBins:
                totalSpiking += (np.power(meanSmoothRate[idx], 2))  # sum square spiking
            rate_score = maxPossibleSpiking - totalSpiking  # reward low levels of firing spread over many bins



        ########################## plotting for stable duration score
        if verboseplot:
            # check the processed raster
            plt.figure
            plt.imshow(raster,interpolation='nearest',aspect='auto')
            plt.tight_layout()
            plt.gray()
            plt.title('how does the raster look before smoothing')
            plt.show()

            plt.figure
            plt.imshow(raster_smooth, interpolation='nearest', aspect='auto')
            plt.tight_layout()
            plt.gray()
            plt.title('how does the raster look after smoothing')
            plt.show()

            #print "ignition frame: " + str(ignitionFrame)
            #print "quench frame: " + str(quenchFrame)
            #print "paroxysm frame: " + str(paroxysmFrame)

            plt.figure
            plt.plot(meanSmoothRate)
            title('mean smooth rate')
            plt.show()



        #####################  compute corr coeffs of rates  # todo do this better
        if do_score_asynchrony:
            CORR_COEFF_SAMPLE = 300 # how many neurons should we compute corr coeffs for?
            BINS_SAMPLE = 100
            if numActiveNeurons < CORR_COEFF_SAMPLE:  # impose min number of active neurons
                asynchrony_score = -np.inf # no meaningful definition for null trials
            elif numStableBins < BINS_SAMPLE:
                asynchrony_score = -np.inf   # impose min number of stable bins
            else:
                #corrcoeffs = np.corrcoef(raster[0:self.N_e][stablePeriodBegin:stablePeriodEnd])  # look at excitatory cells only
                sampleIdxs = np.argpartition(avgRates,-CORR_COEFF_SAMPLE)[-CORR_COEFF_SAMPLE:] # K highest rates from high to low
                corrcoeffs = np.corrcoef(raster_smooth[sampleIdxs][:]) # corr coeffs among sample neurons
                if len(corrcoeffs) > 1:
                    np.fill_diagonal(corrcoeffs, np.nan) # mask out the self-self comparisons (replace with nans)
                #print "sample diagonal element - " , corrcoeffs[5][5] # testing

                asynchrony_score = 1 - mean(np.abs(corrcoeffs[~np.isnan(corrcoeffs)])) # this isn't working well
                # asynchrony_score = -(np.sum(np.abs(corrcoeffs[~np.isnan(corrcoeffs)])))  # minimize the corr corrcoeffs (= maximize -1 * sumsquare corrcoeffs)
                # todo how about based on power in the autocorr function

                if verboseplot:
                    print 'shape of the corroeffs ', shape(corrcoeffs)
                    print " mean corr coeff " + str(mean(corrcoeffs[~np.isnan(corrcoeffs)]))
                    print " mean abs corr coeff " + str(mean(np.abs(corrcoeffs[~np.isnan(corrcoeffs)])))

                    plt.figure()
                    plt.hist(corrcoeffs[~np.isnan(corrcoeffs)], bins=40)
                    plt.show()
                    plt.title('distributon of correlation coefficients')  # todo add landmarks to plot



###########################   other ideas - CV of ISIs, zipf's law, scaling with sample size, etc



###########################      package it all up

        NUM_COMPONENTS = 1  # 3  # temp - asynchrony and null # todo need to read this in automatically in the constructor
        score_components = np.zeros((NUM_COMPONENTS,))
        #score_components[0] = asynchrony_score # asynchrony_score # it's named like this because the plan used to be, pareto front
        #score_components[1] = stable_duration_score # temp

        #score_components[0] = stable_duration_score
        #score_components[1] = rate_score
        #score_components[2] = asynchrony_score
        score_components[0] = rate_score
        for i_component in range(NUM_COMPONENTS):
            if np.isnan(score_components[i_component]):
                score_components[i_component] = -np.inf

        #if verboseplot:
            #print "asynchrony_score " + str(asynchrony_score)
        if do_score_duration:
            print " stable duration score " + str(stable_duration_score)
        if do_score_rate:
            print "rate score " + str(rate_score)
        if do_score_asynchrony:
            print "asynchrony score " + str(asynchrony_score)

        return score_components

