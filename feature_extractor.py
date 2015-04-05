import numpy as np

def fold_time_series(time_point, period, div_period):
    return time_point - 1.0 * int(time_point / (period / div_period)) * period / div_period


class FeatureExtractor():

    def __init__(self):
        pass

    def fit(self, X_dict):
        pass

    def transform(self, X_dict):


        cols = [
            'magnitude_b',
            'magnitude_r',
            'period',
            'asym_b',
            'asym_r',
            'log_p_not_variable',
            'sigma_flux_b',
            'sigma_flux_r',
            'quality',
            'div_period',
        ]

        time_features_labels = (    ("time_points_r", "light_points_r"), ("time_points_b", "light_points_b") )
        nb_freq =  10
        nb_time_features = len(time_features_labels) * nb_freq

        X_transformed = np.zeros((len(X_dict), len(cols) + nb_time_features))

        for i, example in enumerate(X_dict):

            features = [example[c] for c in cols]

            period = example["period"]
            div_period = example["div_period"]

            for time_label, amplitude_label in time_features_labels:
                time_points = example[time_label]
                #time_points_folded = [fold_time_series(time_point, period, div_period)
                #                      for time_point in time_points]
                #time_points_folded = np.array(time_points_folded)
                amplitudes = np.array(example[amplitude_label])

                missing_values = np.isnan(amplitudes)

                #time_points_folded = time_points_folded[np.logical_not(missing_values)][:, np.newaxis]
                amplitudes = amplitudes[np.logical_not(missing_values)]

                freq = np.fft.fft(amplitudes)
                freq_magnitude = np.abs(freq)
                freq = map(lambda f:f[1],
                           sorted(enumerate(freq.tolist()), key=(lambda (i, n):freq_magnitude[i]), reverse=True))
                freq = np.array(freq)
                freq = freq[0:nb_freq]
                features.extend(np.abs(freq))

            X_transformed[i] = features
        return X_transformed
