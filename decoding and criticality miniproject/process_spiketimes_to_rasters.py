import numpy as np
import math
import matplotlib.pyplot as plt

# zeroth draft of decoding attempts:
plotverbose = False

N_families = 1 # temp while, trying things out
N_inputs = 2
N_winners = 1 # temp, while testing things out
N_repeats = 500

spiketimes_dir = 'analyses/3-2-2017/spiketimes/'
rasters_dir = 'analyses/3-2-2017/rasters/' # for saving the binned rasters

t_range = [75, 100] # ms   (throwing away the last 50 ms)
cell_range = [0,3200]
T_RES = 25 # ms

duration = t_range[1] - t_range[0] + 1 # ms
N_cells = cell_range[1] - cell_range[0] + 1
N_bins = math.ceil(duration / T_RES) # todo save these config parameters in the rasters directory

for i_family in range(N_families):
    print "processing family " + str(i_family) + "..."
    for i_input in range(N_inputs):
        for i_winner in range(N_winners):
            for i_repeat in range(N_repeats):

                postfix = 'family ' + str(i_family) + ' input ' + str(i_input) + ' winner ' + str(i_winner) + ' ' + str(i_repeat) + '.npy'
                spiketimes = np.load(spiketimes_dir + postfix)  # (t (ms), index)

                raster = np.zeros((N_cells, N_bins)) # todo need to write these params to a file for record keeping

                for [t,i] in spiketimes.T: # enum over columns
                    i = int(i) # make sure python knows this is an integer
                    if t >= t_range[0] and t < t_range[1]: # check for the temporal epoch we are interested in binning
                        if i >= cell_range[0] and i < cell_range[1]: # check for the subset of cells we are interested in

                            cell_idx = i - cell_range[0]
                            t_idx = math.floor( (t - t_range[0]) / T_RES)

                            raster[cell_idx][t_idx] += 1
                if plotverbose:
                    plt.figure()
                    plt.imshow(raster,aspect='auto')
                    plt.title('test view of raster binning')
                    plt.show()

                np.save(rasters_dir + postfix,raster) # todo save everything in one step instead



