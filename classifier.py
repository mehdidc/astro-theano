from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator

from sklearn.base import BaseEstimator
 
import os
os.environ["OMP_NUM_THREADS"] = "1"
from lasagne.easy import SimpleNeuralNet
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import Pipeline
import theano
 
class Classifier(BaseEstimator):
 
    def __init__(self):

	nnet = Pipeline([
            ('imputer', Imputer()),
            ('scaler', StandardScaler()),
            ('neuralnet', SimpleNeuralNet(nb_hidden_list=[30],
                                          max_nb_epochs=50,
                                          batch_size=100,
                                          learning_rate=1.)),
        ])
	
        self.clf = Pipeline([
            ('rf', AdaBoostClassifier(base_estimator=nnet,
				      n_estimators=10))
        ])
 
    def fit(self, X, y):
        self.clf.fit(X, y)
 
    def predict(self, X):
        return self.clf.predict(X)
 
    def predict_proba(self, X):
        return self.clf.predict_proba(X)
