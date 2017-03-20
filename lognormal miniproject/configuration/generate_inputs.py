import json

from brian2 import *
import scipy.ndimage.filters as filt
import scipy.stats as stats
from brian2.units.allunits import *
from scipy import signal
from tempfile import TemporaryFile


# NOTE this is currently parameterized by
# uniform distribution of poisson firing rates
# 0.2 p_connect from poisson to exc
# and some uniform jitter in the connection weight

# goal for the inputs - let's do something that looks more like these: http://science.sciencemag.org/content/312/5780/1622.full

target_dir = '../simulations/3-2-2017/'
save_postfix = "2-28-2017 inputs.json"
save_name = target_dir + save_postfix

# design parameters
N_input = 200 # number of poisson thalamic inputs
inputMean = 10*Hz # hz # for the Poisson rates
my_rates = np.random.uniform(0,inputMean/Hz * 2,N_input) * Hz # draw a uniform bag of firing rates, idealizing stimululs tuning
tau_input = 3 # ms # todo it would be nice to use a depressing burst

w_poisson_out = 500 # scaling to get the conductances into the biologically plausible range
        # nS

DURATION = 50*ms
T_RES = 0.1*ms
N_targets = 400 # WARNING this has to batched by hand to N_inputs in the network config files (todo fix this issue)
p_target_receives_input = 0.15 # todo what level of input correlation does this induce (probably too much)

# init
input_connectivity = np.random.rand(N_input, N_targets)
active_recipients = np.arange(N_targets)
#for i_recipient in range(N_input):
#    if active_recipients(i_recipient) < p_target_receives_input:
#        active_recipients.append(i_recipient)

for i_input in range(N_input):
    for i_target in active_recipients:
        if input_connectivity[i_input][i_target] < p_target_receives_input:
            input_connectivity[i_input][i_target] = 1 + 0.2*np.random.rand() # a little bit of heterogeneity # todo think about this todo no negative weights
        else:
            input_connectivity[i_input][i_target] = 0

# instantiate a poisson population

PoissonInputGroup = PoissonGroup(N_input, my_rates)
p_mon = SpikeMonitor(PoissonInputGroup)

run(DURATION)

plt.figure()
plt.plot(p_mon.t / ms, p_mon.i, '.k')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.title('population spiking')
plt.show()

tres_input = 0.1 # ms
N_inputbins = DURATION/ms / tres_input + 1
input_conductances = np.zeros((N_targets, N_inputbins)) # init
input_current_components = np.zeros((N_input, N_inputbins))

for idx,t in enumerate(p_mon.t):
    spike_id = p_mon.i[idx]
    #t
    binned_idx = divmod(t/ms,tres_input)[0] # the whole part is the idx  (t0 = 0th bin)
    input_current_components[spike_id][binned_idx] += 1
# binned
plt.figure()
plt.imshow(input_current_components,aspect='auto')
plt.title('poisson activity after binning')
plt.show()

# convolve with exponentials  (todo it would be cool to convolve with a depressing burst)
tau_filterInSamples = tau_input / tres_input
RANGE_FRACTION = 0.01 # cutoff the exp function when this much of the y-range is left
M = -(tau_filterInSamples*np.log(RANGE_FRACTION) - 1) # include 99% of range and solve for number of samples ... todo account for dt
exp_kernel = signal.exponential(M, 0, tau_filterInSamples, False)
exp_kernel = exp_kernel * 1.0 / np.sum(exp_kernel) # make sure integral = 1 s
plt.figure()
plt.plot(exp_kernel)
title('filtering kernel for input currents')
plt.show()

                            # = N_poisson, N_targets.T   **   N_poisson, T
input_current_components = w_poisson_out * filt.convolve1d(input_current_components, exp_kernel, axis=1) # oops, this probably wasn't matrix multiplication

for i_target in range(N_targets): # row in input cells
    for i_source in range(N_input):
        if input_connectivity[i_source][i_target] > 0:
            input_conductances[i_target][:] += input_current_components[i_source][:] * input_connectivity[i_source][i_target] # scale by the connectio weight


# plot
plt.figure()
plt.imshow(input_conductances, aspect='auto')
plt.title('sum input activity')
plt.show()

plt.figure()
idx = 1
plt.plot(input_conductances[idx][:])
plt.title('neuron number ' + str(idx))
plt.ylabel('input conductance (nS)')
plt.show()

C = 1 # get a rough idea of the induced voltage at rest
V_clamp = 60 # mV
E_input = 0 # mV
input_currents = input_conductances * (V_clamp - E_input) / C

plt.figure()
idx = 1
plt.plot(input_conductances[idx][:])
plt.title('neuron number ' + str(idx))
plt.ylabel('effective voltage induced (mV)')
plt.show()



# save as json

savefile = open(save_name,'w')
save_object = {"input_conductances":input_conductances.tolist(), "dt": T_RES / ms, #  "target_fraction":target_fraction,
                            "p_target_receives_input":p_target_receives_input,
                            "inputMean":inputMean/Hz,  "tau_input":tau_input, # "inputStd":inputStd/Hz,
                            "N_targets":N_targets,"w_poisson":w_poisson_out}
json.dump(save_object,savefile,sort_keys=True,indent=2)


