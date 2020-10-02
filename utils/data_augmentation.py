import numpy as np 
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsRegressor

class RegressionSMOTE(BaseEstimator):
    
    def __init__(self, n, k, sigma, **kwargs):
        self.n = n
        self.k = k
        self.sigma = sigma
        self.random_state = kwargs.get('random_state', np.random.random())
        
    def fit(self, X, y):
        self.knn = KNeighborsRegressor(self.k, 'distance').fit(X, y)

    def transform(self, X, y):
        np.random.seed(self.random_state)
        
        ix = np.random.choice(len(X), self.n)
        nn = self.knn.kneighbors(X[ix], return_distance=False)
        newY = self.knn.predict(X[ix])
        nni = np.random.choice(self.k, self.n)
        ix2 = np.array([n[i] for n, i in zip(nn, nni)])
        
        dif = X[ix] - X[ix2]
        gap = np.random.rand(self.n, 1)
        newX = X[ix] + dif*gap
        newX = newX + np.random.rand(*newX.shape)*self.sigma
        
        return newX, newY
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
    
class MovingAverageAugmentor(BaseEstimator):
    
    def __init__(self, windowsize=2):
        self.windowsize = windowsize
        
    def fit(self, X, y):
        pass
    
    def transform(self, X, y):
        return (
            pd.DataFrame(X).rolling(window=self.windowsize).mean().iloc[self.windowsize-1:].values,
            pd.DataFrame(y).rolling(window=self.windowsize).mean().iloc[self.windowsize-1:].values
        )
    
    def fit_transform(self, X, y):
        return self.transform(X, y)    