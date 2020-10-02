from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ShuffleSplit

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from xgboost import XGBClassifier, XGBRFClassifier, XGBRegressor, XGBRFRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

def HyperOptPipeline(algo, n_iter=-1):
    if algo in ['linreg', 'logreg', 'svr', 'svc']:
        ss = StandardScaler()
        mms = MinMaxScaler()
        
    if algo == 'linreg':
        model_linreg = LinearRegression()
        model_lasso = Lasso()
        model_ridge = Ridge()
        model_elasticnet = ElasticNet()
        
        params = [
            {
                'scaler': [ss, mms],
                'estimator': [model_linreg]
            },{
                'scaler': [ss, mms],
                'estimator__alpha': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                'estimator': [model_lasso]
            },{
                'scaler': [ss, mms],
                'estimator__alpha': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                'estimator': [model_ridge]
            },{
                'scaler': [ss, mms],
                'estimator__alpha': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                'estimator__l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'estimator': [model_elasticnet]
            }
        ]
        
        pipeline = Pipeline([('scaler', ss), ('estimator', model_linreg)])
        
    if algo == 'logreg':
        model_logreg = LogisticRegression(class_weight='balanced', solver='saga', max_iter=100_000)

        params = [
            {
                'scaler': [ss, mms],
                'estimator__penalty': ['l1', 'l2'],
                'estimator__C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
            },
            {
                'scaler': [ss, mms],
                'estimator__penalty': ['elasticnet'],
                'estimator__C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                'estimator__l1_ratio': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            },
            {
                'scaler': [ss, mms],
                'estimator__penalty': ['none'],
            },
        ]
        
        pipeline = Pipeline([('scaler', ss), ('estimator', model_logreg)])

    if algo in ['svc', 'svr']:
        
        model = SVC(class_weight='balanced') if algo == 'svc' else SVR()
        
        params = [
            {
                'scaler': [ss, mms],
                'estimator__kernel': ['linear'],
                'estimator__C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
            },
            {
                'scaler': [ss, mms],
                'estimator__kernel': ['rbf', 'sigmoid'],
                'estimator__gamma': ['scale', 'auto'],
                'estimator__C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
            },
            {
                'scaler': [ss, mms],
                'estimator__kernel': ['poly'],
                'estimator__gamma': ['scale', 'auto'],
                'estimator__C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
                'estimator__degree': [2, 3, 4, 5]
            }
        ]
        
        pipeline = Pipeline([('scaler', ss), ('estimator', model)])
        
    if algo in ['ctree', 'rtree']:
        if algo == 'ctree':
            model_rf = RandomForestClassifier(class_weight='balanced')
            model_gb = GradientBoostingClassifier()
            model_et = ExtraTreesClassifier(class_weight='balanced')
            model_xgb = XGBClassifier()
            model_xgbrf = XGBRFClassifier()
            model_cb = CatBoostClassifier(bootstrap_type='Bernoulli')
            model_lgbm = LGBMClassifier(class_weight='balanced')
        else:
            model_rf = RandomForestRegressor()
            model_gb = GradientBoostingRegressor()
            model_et = ExtraTreesRegressor()
            model_xgb = XGBRegressor()
            model_xgbrf = XGBRFRegressor()
            model_cb = CatBoostRegressor(bootstrap_type='Bernoulli')
            model_lgbm = LGBMRegressor()

        params =  [
            {
                'estimator': [model_rf],
                'estimator__n_estimators': [10, 50, 100, 250, 500],
                'estimator__max_depth': [5, 10, 15, 25, 30, None],
                'estimator__min_samples_split': [1.0, 2, 5, 10, 15, 100],
                'estimator__min_samples_leaf': [1, 2, 5, 10],
            },
            {
                'estimator': [model_gb],
                'estimator__n_estimators': [10, 50, 100, 250, 500],
                'estimator__learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
                'estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'estimator__max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
                'estimator__min_samples_split': [1.0, 2, 5, 10, 15, 100],
                'estimator__min_samples_leaf': [1, 2, 5, 10],
            },
            {
                'estimator': [model_et],
                'estimator__n_estimators': [10, 50, 100, 250, 500],
                'estimator__max_depth': [5, 10, 15, 25, 30, None],
                'estimator__min_samples_split': [1.0, 2, 5, 10, 15, 100],
                'estimator__min_samples_leaf': [1, 2, 5, 10],
            },
            {
                'estimator': [model_xgb],
                'estimator__n_estimators': [10, 50, 100, 250, 500],
                'estimator__learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
                'estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'estimator__max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
                'estimator__gamma': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                'estimator__min_child_weight': [1, 3, 5, 7],
                'estimator__reg_lambda': [0.01, 0.1, 1.0],
                'estimator__reg_alpha': [0, 0.1, 0.5, 1.0],
            },
            {
                'estimator': [model_xgbrf],
                'estimator__n_estimators': [10, 50, 100, 250, 500],
                'estimator__learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
                'estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'estimator__max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
                'estimator__gamma': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                'estimator__min_child_weight': [1, 3, 5, 7],
                'estimator__reg_lambda': [0.01, 0.1, 1.0],
                'estimator__reg_alpha': [0, 0.1, 0.5, 1.0],
            },
            {
                'estimator': [model_cb],
                'estimator__n_estimators': [10, 50, 100, 250, 500],
                'estimator__learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
                'estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'estimator__max_depth': [3, 5, 7, 9, 12, 15, 16],
                'estimator__reg_lambda': [0.01, 0.1, 1.0],
            },
            {
                'estimator': [model_lgbm],
                'estimator__n_estimators': [10, 50, 100, 250, 500],
                'estimator__learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
                'estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'estimator__min_child_samples': [1, 2, 5, 10, 15, 100],
                'estimator__min_child_weight': [1, 3, 5, 7],
                'estimator__reg_lambda': [0.01, 0.1, 1.0],
                'estimator__reg_alpha': [0, 0.1, 0.5, 1.0],
            } 
        ]  
        
        pipeline = Pipeline([('estimator', model_rf)]) 
    
    n_params = 0        
    for param_dict in params:    
        n = 1
        for v in param_dict.values():
            n *= len(v)
        n_params += n
        
    print(n_params, 'parameter settings identified')
    if n_iter == -1:
        return GridSearchCV(pipeline, params, cv=ShuffleSplit(test_size=0.1, n_splits=1, random_state=19))
    return RandomizedSearchCV(pipeline, params, n_iter=n_iter, cv=ShuffleSplit(test_size=0.1, n_splits=1, random_state=19), random_state=19)