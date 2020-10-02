import numpy as np
import impyute
from sklearn.base import BaseEstimator

class Imputer(BaseEstimator):
    
    def __init__(self, strategy, **params):
        """
        params: 
            strategy == 'constant':
                constant: float
            strategy == 'em':
                loops: int, Number of em iterations to run before breaking
            strategy == 'fast_knn':
                k : int
                eps: nonnegative float, Return approximate nearest neighbors; the kth returned value is guaranteed to be no further than (1+eps) times the distance to the real kth nearest neighbor
                p : float, 1<=p, Which Minkowski p-norm to use
                distance_upper_bound : nonnegative float, Return only neighbors within this distance
                leafsize: int, The number of points at which the algorithm switches over to brute-force
            strategy == 'locf':
                axis
            strategy == 'moving_window':
                wsize: int, Window size
                errors: {raise, coerce, ignore}, Errors will occur with the indexing of the windows
                
        """
        self.all_strategies = [
            'constant', 'random', 
            'mean', 'median', 'mode', 'buck_iterative', 'em', 'fast_knn',
            'locf', 'moving_window'
        ]
        
        if strategy not in self.all_strategies:
            raise Exception('Invalid Strategy... strategy can be one of the follwing:\n', *self.all_strategies)
        
        if strategy == 'constant' and 'constant' not in params.keys():
            raise Exception('Provide a constant value to impute')
        
        self.strategy = strategy
        self.params = params
        
    def fit(self, X):
        return self
    
    def transform(self, X):
        if self.strategy == 'constant':
            X[np.where(np.isnan(X))] = self.params['constant']
            return X
        if self.strategy == 'random':
            return impyute.imputation.cs.random(X)
        if self.strategy == 'mean':
            return impyute.imputation.cs.mean(X)
        if self.strategy == 'median':
            return impyute.imputation.cs.median(X)
        if self.strategy == 'mode':
            return impyute.imputation.cs.mode(X)
        if self.strategy == 'buck_iterative':
            return impyute.imputation.cs.buck_iterative(X)
        if self.strategy == 'em':
            return impyute.imputation.cs.em(X, loops=self.params.get('loops', 50))
        if self.strategy == 'fast_knn':
            return impyute.imputation.cs.em(
                X, k=self.params.get('k', 3), eps=self.params.get('eps', 0), p=self.params.get('p', 2), 
                distance_upper_bound=self.params.get('distance_upper_bound', float('inf')), 
                leafsize=self.params.get('leafsize', 10)
            )
        if self.strategy == 'locf':
            return impyute.imputation.ts.locf(X, axis=self.params.get('axis', 0))
        if self.strategy == 'moving_window':
            return impyute.imputation.ts.moving_window(
                X, wsize=self.params.get('wsize', 5), errors=self.params.get('errors', 'coerce')
            )
    
    def fit_transform(self, X):
        return self.transform(X)