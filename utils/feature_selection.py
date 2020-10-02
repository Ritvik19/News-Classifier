import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score

from sklearn.feature_selection import chi2, f_classif, mutual_info_classif  
from sklearn.feature_selection import f_regression, mutual_info_regression  
from sklearn.feature_selection import SelectKBest, SelectPercentile 

class UnivariateFeatureSelction:
    def __init__(self, n_features, problem_type, scoring):
        """  
        Custom univariate feature selection wrapper on
        different univariate feature selection models from  scikit-learn.
        :param n_features: SelectPercentile if float else SelectKBest  
        :param problem_type: classification or regression  
        :param scoring: scoring function, string 
        """
    
        if problem_type == "classification":  
            valid_scoring = {
                "f_classif": f_classif,
                "chi2": chi2,  "mutual_info_classif": mutual_info_classif  
            }  
        else:  
            valid_scoring = {
                "f_regression": f_regression,
                "mutual_info_regression": mutual_info_regression  
            } 
            
        if scoring not in valid_scoring:  
            raise Exception("Invalid scoring function") 
            
        #if n_features is int, we use selectkbest 
        #if n_features is float, we use selectpercentile 
        if isinstance(n_features, int):  
            self.selection = SelectKBest(valid_scoring[scoring], k=n_features)
        elif isinstance(n_features, float):  
            self.selection = SelectPercentile(valid_scoring[scoring],  percentile=int(n_features * 100))
        else:
            raise Exception("Invalid type of feature")

    def fit(self, X, y):
        return self.selection.fit(X, y)

    def transform(self, X): 
        return self.selection.transform(X)  

    def fit_transform(self, X, y):  
        return self.selection.fit_transform(X, y)
    
class GreedyFeatureSelection():
    
    def __init__(self, problem_type, epsilon=0.00001):
        self.epsilon = epsilon
        self.problem_type = problem_type
        if problem_type not in ['regression', 'classification']:
            raise Exception('Invalid problem type')
            
    def evaluate_score(self, X, y):
        if self.problem_type == 'regression':
            model = LinearRegression().fit(X, y)
            score = r2_score(y, model.predict(X))
        else:
            model = LogisticRegression().fit(X, y)
            score = roc_auc_score(y, model.predict_proba(X)[:, 1])
        return score
    
    def fit(self, X, y):
        good_features = []
        best_scores = []
        
        num_features = X.shape[1]
        
        while True:
            this_feature = None
            best_score = 0
            
            for feature in range(num_features):
                if feature in good_features:
                    continue
                
                selected_features = good_features + [feature]
                Xtrain = X.iloc[:, selected_features]
                score = self.evaluate_score(Xtrain, y)
                if score > best_score:
                    this_feature = feature
                    best_score = score
            
            if this_feature is not None:
                good_features.append(this_feature)
                best_scores.append(best_score)
                
            if len(best_scores) > 2:
                if best_scores[-1] - best_scores[-2] < self.epsilon:
                    break
                    
        self.best_scores = best_scores[:-1]
        self.good_features = good_features[:-1]
        return self
    
    def transform(self, X, y):
        return X.iloc[:, self.good_features], y
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(self, X, y)    