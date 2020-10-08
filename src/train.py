import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import load_data, models, model_dispatcher
from tqdm.auto import tqdm

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline

import argparse, os

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')

file_handler = logging.FileHandler('../training.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def train_text(dataset_id, algorithm, problem='bc', n_iter=10):
    X, y = load_data.load_text(dataset_id)

    logger.debug('dataset loaded')
    
    performance = {
        'accuracy' : [],
        'log_loss' : [],
        'auroc' : []
    }
    cm = []

    if len(models.models[algorithm]) == 2:
        preprocessor = models.models[algorithm][0]
    else:
        preprocessor = Pipeline([
            ('vectorizer', models.models[algorithm][0]),
            ('feature_selector', models.models[algorithm][1]),
        ])

    X = preprocessor.fit_transform(X, y)
    num_features = X.shape[1]
    
    if type(preprocessor)  == Pipeline:
        vocab = pd.Series(
            preprocessor['vectorizer'].get_feature_names()
        )[preprocessor['feature_selector'].get_support()].values
    else:
        vocab = preprocessor.get_feature_names()
    
    model_dispatcher.save_preprocessing(
        preprocessor, num_features, vocab, algorithm, dataset_id,
    )

    logger.debug('data preprocessed')

    model = models.models[algorithm][-1]
    
    if problem == 'ml':
        kf = KFold(n_splits=10, shuffle=True, random_state=19)
    else:
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=19)
        
    
    logger.debug('training started')
        
    iter_count = 0
    for train_indices, test_indices in tqdm(kf.split(X, y)):
        X_train = X[train_indices]
        y_train = y[train_indices]

        X_test = X[test_indices]
        y_test = y[test_indices]

        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_pred_ = model.predict_proba(X_test)

        try:
            performance['log_loss'].append(log_loss(y_test, y_pred_))
            if problem in ['mc', 'ml']:            
                performance['auroc'].append(roc_auc_score(y_test, y_pred_, multi_class='ovr'))
            else:
                performance['auroc'].append(roc_auc_score(y_test, y_pred_[:, 1]))
        except:
            pass
        performance['accuracy'].append(accuracy_score(y_test, y_pred))
        
        if problem == 'ml':
            _ = multilabel_confusion_matrix(y_test, y_pred)
        else:
            _ = confusion_matrix(y_test, y_pred)
        
        cm.append(_)
        try:
            logger.debug(f'AUROC {performance["auroc"][-1]}')
        except:
            pass
        logger.debug(f'Accuracy {performance["accuracy"][-1]}')
            
        if type(model) in [GridSearchCV, RandomizedSearchCV]:
            logger.debug(model.best_params_)
            model_dispatcher.save_model(model.best_estimator_, algorithm, dataset_id)
            break
        
        iter_count += 1    
        logger.debug(f'{iter_count} iterations done')
        if iter_count == n_iter:
            model_dispatcher.save_model(model, algorithm, dataset_id)
            break    
        
        model_dispatcher.save_model(model, algorithm, dataset_id)        
                
    cm = np.mean(cm, axis=0)
    if problem == 'ml':
        cm = [normalize(matrix, norm='l1')*100 for matrix in cm]
        
    model_dispatcher.save_results_classification(
        classification_report(y_test, y_pred), cm, performance, algorithm, dataset_id, problem=problem
    )
    
    logger.debug('model dispatched')   
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    
    parser.add_argument('-a', '--algo', metavar='algo', type=str, help='algorithm to run')
    parser.add_argument('-d', '--dataset', metavar='data', type=str, help='dataset to train on')
    parser.add_argument('-p', '--problem', metavar='problem', nargs='?', type=str, help='type of problem')
    parser.add_argument('-t', '--type', metavar='type', nargs='?', type=str, help='type of dataset')
    parser.add_argument('-n', '--n_iter', metavar='n_iter', nargs='?', type=str, help='number of iterations to train')
    
    args = parser.parse_args()
    logger.debug(args)

    train_text(args.dataset.strip(), args.algo.strip(), args.problem.strip(), int(args.n_iter.strip()))
