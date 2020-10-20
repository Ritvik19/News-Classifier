import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
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

    def transform(self, X, y=None): 
        return self.selection.transform(X, y)  

    def fit_transform(self, X, y):  
        return self.selection.fit_transform(X, y)
    
    def get_support(self):
        return self.selection.get_support()
    
class GreedyFeatureSelection():
    
    def __init__(self, problem_type, epsilon=0.0001, random_state=None):
        self.epsilon = epsilon
        self.problem_type = problem_type
        self.random_state = random_state
        if problem_type not in ['regression', 'classification']:
            raise Exception('Invalid problem type')
            
    def evaluate_score(self, X, y):
        if self.problem_type == 'regression':
            return LinearRegression().fit(X, y).score(X, y)
        else:
            return LogisticRegression(
                max_iter=100_000, class_weight='balanced', random_state=self.random_state
            ).fit(X, y).score(X, y)
    
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
            
            else:    
                break
            
            if len(best_scores) > 2:
                if best_scores[-1] - best_scores[-2] < self.epsilon:
                    break
                    
        self.best_scores = best_scores[:-1]
        self.good_features = good_features[:-1]
        return self
    
    def transform(self, X, y=None):
        return X.iloc[:, self.good_features]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)    
    
    def get_support(self):
        return [True if x in self.good_features else False for x in range(self.num_features)]
    
class SelectImportantFeatures():
    
    def __init__(self, problem_type, random_state=None):
        self.problem_type = problem_type
        self.random_state = random_state
        if problem_type not in ['regression', 'classification']:
            raise Exception('Invalid problem type')
        
    def fit(self, X, y):
        if self.problem_type == 'regression':
            model = DecisionTreeRegressor(random_state=self.random_state)
        else:
            model = DecisionTreeClassifier(class_weight='balanced', random_state=self.random_state)
        model.fit(X, y)
        self.support = (model.feature_importances_ > 0)
        self.indices = np.where(self.support)[0]
        
    def transform(self, X, y=None):
        return X[:, self.indices]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)
    
    def get_support(self):
        return self.support
        
