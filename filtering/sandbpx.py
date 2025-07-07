from scipy.io import loadmat

# adjust this to wherever your file actually lives
mat_path = r"C:\Users\mzinn1\Desktop\Datasets\roi_classifier_training\2-L\suite2p\plane0\full_prediction_F.npy.mat"
# load the .mat into a dict
data = loadmat(mat_path)

# inspect top‐level keys (should include your spike_probs)
print(data.keys())
# → dict_keys(['__header__','__version__','__globals__','spike_prob'])

# now extract the array
spike_prob = data['spike_prob']   # shape (n_neurons, n_timepoints)

# e.g. look at the first cell’s inferred rate:
print(spike_prob[0, :])
