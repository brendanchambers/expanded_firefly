import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt


rasters_dir = 'analyses/3-2-2017/rasters/'
perform_decoding = True
N_families = 1 # temp
N_inputs = 2 # todo there are 3 so we can figure out multi-class, work up to that
N_winners = 1 # temp, while getting thigns working
N_repeats = 500
N_bins = 10 # todo read this all in automatically
T_RES = 25 # ms

N_repeats_for_train = 250  # portion of repeats to use for training
N_repeats_for_test = N_repeats - N_repeats_for_train

rasters_dir = 'analyses/3-2-2017/rasters/' # for saving the binned rasters

# init
postfix = 'family 0 input 0 winner 0 0.npy'# load any one of these, just to grab N_cells, N_bins
raster = np.load(rasters_dir + postfix)  # load raster
N_bins = np.shape(raster)[1]
N_cells = np.shape(raster)[0]
N_x = N_bins*N_cells
num_train_vectors = N_repeats_for_train*N_inputs
num_test_vectors = N_repeats_for_test*N_inputs

plt.figure()
plt.imshow(raster,aspect='auto',interpolation='None')
plt.title('sample raster')
plt.xlabel('bins')
plt.ylabel('sample cells')

if perform_decoding:

    # generate the training vectors and labels

    #for i_winner in range(N_winners):  # one decoder for each winner
    i_winner = 0
    #for i_family in range(N_families): # one decoder for each family
    i_family = 0
    training_vectors = np.zeros((num_train_vectors, N_x))
    training_labels = np.zeros((num_train_vectors,))
    train_idx = 0
    for i_input in range(N_inputs):
        print 'constructing training vectors for input ' + str(i_input) + '...'
        for i_repeat in range(N_repeats_for_train):

            postfix = 'family ' + str(i_family) + ' input ' + str(i_input) + ' winner ' + str(i_winner) + ' ' + str(i_repeat) + '.npy'
            raster = np.load(rasters_dir + postfix)  # load raster

            input_vector = raster.flatten()
            training_vectors[train_idx][:] = input_vector
            training_labels[train_idx] = i_input

            train_idx += 1



    # build an SVM
    print 'performing svm training...'
    my_svm = svm.SVC()
    my_svc = my_svm.fit(training_vectors, training_labels)




    # generate the test vectors
    #for i_winner in range(N_winners):  # one decoder for each winner
    i_winner = 0
    #for i_family in range(N_families): # one decoder for each family
    i_family = 0
    test_vectors = np.zeros((num_test_vectors, N_x))
    test_labels = np.zeros((num_test_vectors,))
    test_idx = 0
    for i_input in range(N_inputs):
        print ' constructing test vectors for input ' + str(i_input) + '...'
        for i_repeat in range(N_repeats_for_train+1,N_repeats):

            postfix = 'family ' + str(i_family) + ' input ' + str(i_input) + ' winner ' + str(i_winner) + ' ' + str(i_repeat) + '.npy'
            raster = np.load(rasters_dir + postfix)  # load raster

            input_vector = raster.flatten()
            test_vectors[test_idx][:] = input_vector
            test_labels[test_idx] = i_input

            test_idx += 1

    # check the test vectors
    print 'performing svm classication...'
    test_classifications = my_svm.predict(test_vectors) # test_vectors)

    save_dir = 'analyses/3-2-2017/analysis/test_results.npy'
    test_results = {'test_classifications': test_classifications, 'test_labels': training_labels} # test_labels}
    np.save(save_dir, test_results)

else: # otherwise, load in previous decoding

    load_dir = 'analyses/3-2-2017/analysis/test_results.npy'
    test_results_load = np.load(load_dir)
    test_classifications = test_results_load[()]['test_classifications']
    test_labels = test_results_load[()]['test_labels']

print "shape of test classifications " + str(np.shape(test_classifications))
print "shape of test_labels " + str(np.shape(test_labels))
#print 'predicted: ' + str(test_classifications)
#print 'actual: ' + str(test_labels)



viz_alignment = np.stack((test_classifications,test_labels))
plt.figure()
plt.imshow(viz_alignment,aspect='auto',interpolation='none')
plt.title('test classifications; test_labels')
plt.show()

# % correct
percent_correct = np.mean( test_classifications==test_labels )
print "percent correct: " + str(percent_correct)

# TP   FP
# TN   FN

# take a look at the support vectors
support_vectors = my_svm.support_vectors_
sv_weights = my_svm.coef_ # this will break for non-linear kernels
print 'shape of score vectors: ' + str(np.shape(support_vectors))
print 'shape of weight alpha s : ' + str(np.shape(sv_weights))

plt.figure()
plt.imshow(support_vectors,aspect='auto')
plt.title('support vectors')
plt.show()



