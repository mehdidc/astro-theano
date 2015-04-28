import numpy as np
from pyearth.earth import Earth
def unfold_sample(x, color):
    """Operates inplace"""
    real_period = x['period'] / x['div_period']
    phase = (x['time_points_%s' % color] % real_period) / real_period
    order = np.argsort(phase)
    x['phase_%s' % color] = phase[order]
    x['light_points_%s' % color] = np.array(x['light_points_%s' % color])[order]
    x['error_points_%s' % color] = np.array(x['error_points_%s' % color])[order]
    x['time_points_%s' % color] = np.array(x['time_points_%s' % color])[order]


def binify(bins, a, b):
    a_dig = np.digitize(a, bins) - 1
    not_empty_bins = np.unique(a_dig)
    a_bin = np.array([np.mean(a[a_dig == i]) for i in not_empty_bins])
    b_bin = np.array([np.mean(b[a_dig == i]) for i in not_empty_bins])
    return a_bin, b_bin


class FeatureExtractor(object):

    def __init__(self):
        pass

    def fit(self, X_dict, y):
        pass

    def transform(self, X_dict):
        X = []
        for i, x in enumerate(X_dict):
            real_period = x['period'] / x['div_period']
            x_new = [x['magnitude_b'], x['magnitude_r'], real_period,
                     x['asym_b'], x['asym_r'], x['log_p_not_variable'],
                     x['sigma_flux_b'], x['sigma_flux_r'],
                     x['quality'], x['div_period'] ]

            for color in ['r', 'b']:
                unfold_sample(x, color=color)
                x_train = x['phase_' + color]
                y_train = x['light_points_' + color]
                y_sigma = x['error_points_' + color]

                num_bins = 64
                bins = np.linspace(0, 1, num_bins + 1)

                model = Earth(penalty=0.3,
                              max_terms=10,
                              thresh=0,
                              smooth=True,
                              check_every=5,
                              max_degree=10)
                x_train, y_train  = binify(bins, x_train, y_train)

                time_points_ = np.concatenate((x_train - 1.,
                                                x_train,
                                                x_train + 1.), axis=0)
                light_points_ = np.concatenate((y_train,
                                                y_train,
                                                y_train), axis=0)

                model.fit(time_points_[:, np.newaxis], light_points_)

                t = np.arange(-1., 2., 0.01)
                y=model.predict(t)
                i_max = y.argmax()

                t_ = t
                y_ = np.concatenate( (y[i_max:], y[0:i_max]), axis=0 )

                x_new.append(t[i_max])
                amplitude = max(y_) - min(y_)
                x_new.append(amplitude)
                y_ /= amplitude

                #plt.plot(time_points_, light_points_, c='red')
                #plt.plot(t_, y_, c='green')

                #plt.show()

                for p in y_:
                    x_new.append(p)

            X.append(x_new)
        return np.array(X)
