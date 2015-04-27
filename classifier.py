from sklearn.base import BaseEstimator

import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
import theano

class Classifier(BaseEstimator):

    def __init__(self):

        self.clf = Pipeline([
            ('imputer', Imputer()),
            ('rf', RandomForestClassifier(n_estimators=100)),
        ])

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
