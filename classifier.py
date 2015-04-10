from sklearn.base import BaseEstimator

import os
os.environ["OMP_NUM_THREADS"] = "1"
from lasagne.easy import SimpleNeuralNet
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
import theano


class Classifier(BaseEstimator):

    def __init__(self):
        neural_net = SimpleNeuralNet(nb_hidden_list=[54, 32],
                                     max_nb_epochs=254,
                                     batch_size=100,
                                     learning_rate=0.74586477334620205)
        self.clf = Pipeline([
            ('imputer', Imputer()),
            ('scaler', StandardScaler()),
            ('nnet', neural_net)
        ])

    def __getattr__(self, attrname):
        return getattr(self.clf, attrname)

    def fit(self, X, y):
        X = X.astype(theano.config.floatX)
        y = y.astype(np.int32)
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        X = X.astype(theano.config.floatX)
        return self.clf.predict(X)

    def predict_proba(self, X):
        X = X.astype(theano.config.floatX)
        return self.clf.predict_proba(X)
