import os
import numpy as np
import tensorflow as tf
print(tf.__version__)  # Ensure TensorFlow is imported correctly
from Cascade.cascade2p import config, utils
from Cascade.cascade2p.cascade import get_model_paths  # or adjust if your import path differs
from scipy.ndimage import gaussian_filter, binary_dilation
class CascadePredictor:
    """
    Load all Keras models for a given Cascade2p model_name + model_folder,
    then predict on new traces without reloading from disk each time.
    """
    def __init__(self, model_name: str = "Global_EXC_30Hz_smoothing100ms_high_noise", model_folder: str = "Pretrained_models"):
        self.model_name   = model_name
        self.model_folder = model_folder
        self._load_config()
        self._load_models()

    def _load_config(self):
        # read in the YAML config for this model
        model_path = os.path.join(self.model_folder, self.model_name)
        cfg_file = os.path.join(model_path, "config.yaml")
        self.cfg = config.read_config(cfg_file)

        # pull out the key parameters we’ll need
        self.noise_levels = self.cfg["noise_levels"]
        self.ensemble_size = self.cfg["ensemble_size"]
        self.before_frac = self.cfg["before_frac"]
        self.window_size = self.cfg["windowsize"]
        self.smoothing = self.cfg["smoothing"]
        self.batch_size = self.cfg["batch_size"]

    def _load_models(self):
        """
        Uses get_model_paths() to find every .h5 for each noise level,
        loads them into memory once, and keeps them in a dict.
        """
        load_model = tf.keras.models.load_model
        model_path = os.path.join(self.model_folder, self.model_name)
        paths_dict = get_model_paths(model_path)

        self.models = {}
        for noise, paths in paths_dict.items():
            self.models[noise] = [ load_model(p) for p in paths ]

    def predict(self,
                traces: np.ndarray,
                threshold: int = 0,
                padding: float = np.nan,
                verbosity: int = 1) -> np.ndarray:
        """
        traces: shape (n_rois, n_frames)
        returns:
          Y_predict: shape (n_rois, n_frames)
        """
        # --- MATCH noise level per neuron (same as original predict) ---
        # calculate noise if needed
        trace_noise_levels = utils.calculate_noise_levels(traces, self.cfg["sampling_rate"])
        diffs = (trace_noise_levels[:, None] - np.array(self.noise_levels)[None, :])
        best_idx = np.argmin(np.abs(diffs), axis=1)

        # preprocess windows
        XX = utils.preprocess_traces(traces,
                                     before_frac=self.before_frac,
                                     window_size=self.window_size)

        Y_predict = np.zeros((XX.shape[0], XX.shape[1]))

        # loop by noise level
        for i, noise_level in enumerate(self.noise_levels):
            neuron_idxs = np.where(best_idx == i)[0]
            if neuron_idxs.size == 0:
                continue

            # reshape and predict with each ensemble member
            XX_sel = XX[neuron_idxs].reshape(-1, XX.shape[2])
            # match Keras input dims if needed
            if XX_sel.ndim == 2:
                XX_sel = np.expand_dims(XX_sel, 2)

            # average across ensemble
            preds = np.zeros((len(neuron_idxs), XX.shape[1]))
            for model in self.models[noise_level]:
                flat = model.predict(XX_sel, batch_size=self.batch_size,
                                     verbose=verbosity)
                preds += flat.reshape(len(neuron_idxs), XX.shape[1])
            preds /= len(self.models[noise_level])
            Y_predict[neuron_idxs] = preds

        # thresholding logic (same as original)
        if threshold is False:
            pass
        elif threshold == 1:

            single = np.zeros(1001); single[501] = 1
            sfilt = gaussian_filter(single, sigma=self.smoothing * self.cfg["sampling_rate"])
            thr = sfilt.max() / np.e
            for n in range(Y_predict.shape[0]):
                mask = Y_predict[n] > thr
                mask = binary_dilation(mask, iterations=int(self.smoothing * self.cfg["sampling_rate"]))
                Y_predict[n, ~mask] = 0
        else:
            Y_predict[Y_predict < 0] = 0

        # pad edges
        bf = int(self.before_frac * self.window_size)
        aft = XX.shape[1] - bf
        Y_predict[:, :bf] = padding
        Y_predict[:, -bf:] = padding

        return Y_predict
