import numpy as np
import matplotlib.pyplot as plt
import load_data, models, model_dispatcher
from tqdm.auto import tqdm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import normalize

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model

import argparse, os

def train(dataset_id, algorithm, problem='bc', datatype='tab'):
    if datatype == 'tab':
        X, y = load_data.load_data(dataset_id)
    elif datatype == 'txt':
        X, y = load_data.load_vects(dataset_id)
    
    if problem in ['bc', 'mc', 'ml']:
        performance = {
            'accuracy' : [],
            'log_loss' : [],
            'auroc' : []
        }
        cm = []
    elif problem == 'rg':
        performance = {
            'mse': [],
            'mae': [],
            'r2score': [],
        }
    
    model = models.models[algorithm]
    
    if problem in ['ml', 'rg']:
        kf = KFold(n_splits=10, shuffle=True, random_state=19)
    else:
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=19)
        
        
    for train_indices, test_indices in tqdm(kf.split(X, y)):
        X_train = X[train_indices]
        y_train = y[train_indices]

        X_test = X[test_indices]
        y_test = y[test_indices]

        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        if problem in ['bc', 'mc', 'ml']:
            y_pred_ = model.predict_proba(X_test)
    
            performance['log_loss'].append(log_loss(y_test, y_pred_))
            performance['accuracy'].append(accuracy_score(y_test, y_pred))
            
            if problem in ['mc', 'ml']:            
                performance['auroc'].append(roc_auc_score(y_test, y_pred_, multi_class='ovr'))
            else:
                performance['auroc'].append(roc_auc_score(y_test, y_pred_[:, 1]))
        
            if problem == 'ml':
                _ = multilabel_confusion_matrix(y_test, y_pred)
            else:
                _ = confusion_matrix(y_test, y_pred)
        
            cm.append(_)
            print('AUROC', performance['auroc'][-1])
        
        else:
            performance['mse'].append(mean_squared_error(y_test, y_pred))
            performance['mae'].append(mean_absolute_error(y_test, y_pred))
            performance['r2score'].append(r2_score(y_test, y_pred))
            print('R2Score', performance['r2score'][-1])
            
        if type(model) in [GridSearchCV, RandomizedSearchCV]:
            print(model.best_params_)
            model_dispatcher.save_model(model.best_estimator_, algorithm, dataset_id)
            break
        
        model_dispatcher.save_model(model, algorithm, dataset_id)        
                
    if problem in ['bc', 'mc', 'ml']:                
        cm = np.mean(cm, axis=0)
        if problem == 'ml':
            cm = [normalize(matrix, norm='l1')*100 for matrix in cm]
            
        model_dispatcher.save_results_classification(
            classification_report(y_test, y_pred), cm, performance, algorithm, dataset_id, problem=problem
        )
    else:
        model_dispatcher.save_results_regression(
            performance, algorithm, dataset_id, y_test.reshape(-1), y_pred
        )
        
        
def train_cnn(dataset_id, name, image_size=(28, 28, 1)):
    if os.path.exists(f'../data/{dataset_id}/training'):
        output_classes = len(os.listdir(f'../data/{dataset_id}/training'))
    else:
        output_classes = len(os.listdir(f'../data/{dataset_id}'))
        
    output_directory = os.path.join(os.path.dirname(os.getcwd()), 'performance', f'{name}-{dataset_id}')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)        
        
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=7)
    filepath = f'../models/{name}-{dataset_id}.h5'
    ckpt = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    rlp = ReduceLROnPlateau(monitor='loss', patience=3)
    
    neuralnetwork = models.models[name](image_size, output_classes)
    
    plot_model(
        neuralnetwork, to_file=os.path.join(output_directory, 'arch.png'), 
        show_shapes=True, show_layer_names=True
    )
    
    training_set, validation_set = load_data.load_images(dataset_id, image_size)
    
    history = neuralnetwork.fit_generator(
        training_set, validation_data=validation_set,
        callbacks=[es, ckpt, rlp], epochs=10,
    )
    
    fig, ax = plt.subplots(2, 1, figsize=(20, 12))
    fig.suptitle('Model Performance', fontsize=24) 

    ax[0].plot(history.history['loss'], label='t-loss')
    ax[0].plot(history.history['val_loss'], label='v-loss')
    ax[0].set_title('Loss', fontsize=18)
    ax[0].set_ylabel('Loss')

    ax[1].plot(history.history['acc'], label='t-acc')
    ax[1].plot(history.history['val_acc'], label='v-acc')
    ax[1].set_title('Score', fontsize=18)
    ax[1].set_ylabel('Score')

    for i in range(2):
        ax[i].grid()
        ax[i].legend()
        ax[i].set_xlabel('Epochs')
        
    fig.savefig(os.path.join(output_directory, f'Model-Performance.png'))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    
    parser.add_argument('-a', '--algo', metavar='algo', type=str, help='algorithm to run')
    parser.add_argument('-d', '--dataset', metavar='data', type=str, help='dataset to train on')
    parser.add_argument('-p', '--problem', metavar='problem', nargs='?', type=str, help='type of problem')
    parser.add_argument('-t', '--type', metavar='type', nargs='?', type=str, help='type of dataset')
    
    args = parser.parse_args()
    print(args)
    train(args.dataset.strip(), args.algo.strip(), args.problem.strip(), args.type.strip())