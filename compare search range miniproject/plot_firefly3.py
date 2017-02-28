#results_filestring = 'firefly cleanup 12-6-2016 simulations.json'
#networkconfig_filestring = 'firefly cleanup 12-6-2016 networkconfig.json'
#config_filestring = 'firefly cleanup 12-6-2016 config.json'
from firefly_pack.plot_firefly_fun import plot_firefly_fun

dir = 'simulations/2-16-2017/'
config_filestring = dir + 'huge 2-16-2017 config.json'
networkconfig_filestring = dir + '2-16-2017 networkconfig 1.json'
inputcurrents_filestring = dir + 'input_currents 2-16-2017.json'
results_filestring = dir + 'huge 2-16-2017 simulations 3.json'

plot_firefly_fun(dir,config_filestring,
                 networkconfig_filestring,
                 inputcurrents_filestring,
                 results_filestring)

