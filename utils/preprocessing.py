import numpy as np
import pandas as pd
from scipy import stats

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline

from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

def get_column_names(feature_name, columns):
    val = feature_name.split('_')[1]
    col_idx = int(feature_name.split('_')[0][1:])
    return f'{columns[col_idx]}_{val}'

class Preprocessor():
    
    def __init__(self, return_df=True):
        self.return_df = return_df
        
        self.impute_median = SimpleImputer(strategy='median')
        self.impute_const = SimpleImputer(strategy='constant')
        self.ss = StandardScaler()
        self.ohe = OneHotEncoder(handle_unknown='ignore')
        
        self.num_cols = make_column_selector(dtype_include='number')
        self.cat_cols = make_column_selector(dtype_exclude='number')
        
        self.prerocessor = make_column_transformer(
            (make_pipeline(self.impute_median, self.ss), self.num_cols),
            (make_pipeline(self.impute_const, self.ohe), self.cat_cols),
        )
        
    def fit(self, X, y):
        return self.prerocessor.fit(X)
        
    def transform(self, X, y):
        if return_df:
            return pd.DataFrame(
                self.prerocessor.transform(X).todense(),
                columns=self.num_cols(X)+list(map(
                    lambda x: get_column_names(x, self.cat_cols(X)),
                    self.prerocessor.transformers_[1][1][1].get_feature_names()
                ))
            ), y
        return X, y
        
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
    
class OutlierRemover():
    """
    strategy:
        z_score
        inter_quartile_range
        isolation_forest
        elliptic_envelope
        local_outlier_factor
        one_class_svm
    params:
        Isolation Forest:
            n_estimators
        EllipticEnvelope:
            contamination
        LocalOutlierFactor:
            n_neighbors
        OneClassSVM   
            kernel
            degree
            gamma
            

    """
    def __init__(self, strategy, **params):
        self.all_strategies = [
            'z_score', 'inter_quartile_range', 
            'isolation_forest', 'elliptic_envelope', 'local_outlier_factor', 'one_class_svm'
        ]
        
        if strategy not in self.all_strategies:
            raise Exception(
                'Invalid Strategy... strategy can be one of the follwing:\n', *self.all_strategies
            )
        
        self.strategy = strategy
        self.params = params
        
        if strategy == 'isolation_forest':
            self.outlier_remover = IsolationForest(
                n_estimators=params.get('n_estimators', 100), bootstrap=True, random_state=19
            )
        if strategy == 'elliptic_envelope':
            self.outlier_remover = EllipticEnvelope(
                contamination=params.get('contamination', 0.1), random_state=19
            )
        if strategy == 'local_outlier_factor':
            self.outlier_remover = LocalOutlierFactor(
                contamination=params.get('n_neighbors', 20), random_state=19
            )
        if strategy == 'one_class_svm':
            self.outlier_remover = OneClassSVM(
                kernel=params.get('kernel', 'rbf'), degree=params.get('degree', 3),
                gamma=params.get('gamma', 'scale')
            )
        
    def fit(self, X, y):
        if self.strategy not in ['z_score', 'inter_quartile_range']:
            return self.outlier_remover.fit(X)
        return self
    
    def transform(self, X, y):
        if self.strategy not in ['z_score', 'inter_quartile_range']:
            y_hat = self.outlier_remover.predict(X)
            mask = y_hat != -1
            X, y = X.iloc[mask, :], y.iloc[mask]
            return X, y
        if self.strategy == 'z_score':
            z = pd.DataFrame(np.abs(stats.zscore(X)))
            idx = X[z <= 3].dropna().index,
            return X.iloc[idx], y.iloc[idx] 
        if self.strategy == 'inter_quartile_range':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            idx = X[(X >= (Q1 - 1.5 * IQR)) & (X <= (Q3 + 1.5 * IQR))].dropna().index
            return X.iloc[idx], y.iloc[idx]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)