import numpy as np
from pyearth.earth import Earth

def fold_time_series(time_point, period, div_period):
    real_period = period / div_period
    return time_point % real_period  # modulo real_period

def unfold_sample(x, color):
    """Operates inplace"""
    real_period = x['period'] / x['div_period']
    phase = (x['time_points_%s' % color] % real_period) / real_period
    order = np.argsort(phase)
    x['phase_%s' % color] = phase[order]
    x['light_points_%s' % color] = np.array(x['light_points_%s' % color])[order]
    x['error_points_%s' % color] = np.array(x['error_points_%s' % color])[order]
    x['time_points_%s' % color] = np.array(x['time_points_%s' % color])[order]

class FeatureExtractor(object):

    def __init__(self):
        pass

    def fit(self, X_dict, y):
        pass

    def transform(self, X_dict):
        
        X = []
        ii = 0
        for x in X_dict:
            real_period = x['period'] / x['div_period']
            x_new = [x['magnitude_b'], x['magnitude_r'], real_period,
		     x['asym_b'], x['asym_r'], x['log_p_not_variable'],
		     x['sigma_flux_b'], x['sigma_flux_r'],
		     x['quality'], x['div_period']
	    ]

            for color in ['r', 'b']:
                unfold_sample(x, color=color)
                x_train = x['phase_' + color]
                y_train = x['light_points_' + color]
                y_sigma = x['error_points_' + color]
                
                model = Earth(penalty=2, max_terms=30, smooth=True, endspan=1, max_degree=50)

                time_points_ = np.concatenate((x_train - 1., 
                                               x_train, 
                                               x_train + 1.), axis=0)
                light_points_ = np.concatenate((y_train, 
                                                y_train, 
                                                y_train), axis=0)

                model.fit(time_points_[:, np.newaxis], light_points_)
                
                t = np.arange(-1., 2., 0.01)
                y=model.predict(t)
                i_max=(t[y.argmax()])
                t_ = np.arange(i_max, i_max+1, 0.01)
                y_ = model.predict(t_)                
                x_new.append(i_max)
                amplitude = max(y_) - min(y_)
                x_new.append(amplitude)

                for p in y_:
                    x_new.append(p)
            X.append(x_new)
        return np.array(X)
