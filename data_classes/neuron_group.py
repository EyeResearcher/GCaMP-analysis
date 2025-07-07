class NeuronGroup:
    def __init__(self, filtered_indices, filtered_stat_file, filtered_f_array):
        self.filtered_indices = filtered_indices
        self.filtered_stat_file = filtered_stat_file
        self.filtered_f_array = filtered_f_array
        self.avg_sttc = None
        self.spike_frequency = None
        self.avg_amplitude = None
        self.weighted_avg_amplitude = None
        self.avg_dtw_cost = None
        self.dtw_cost_matrix = None
        self.distance_dict = None
        self.zipped_neuron_amp = []
