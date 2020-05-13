from sklearn.metrics import accuracy_score, confusion_matrix, hamming_loss, hinge_loss, log_loss
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.multiclass import OneVsRestClassifier

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

skf = StratifiedKFold(n_splits=11, shuffle=True, random_state=101)
kf = KFold(n_splits=11, shuffle=True, random_state=101)

def train_model(model, vects, target, labels, **kwargs):
    model_performance = {
        'loss': [],
        'accuracy': [],
    }
    
    for train_indices, test_indices in skf.split(vects, target):
        X_train = vects[train_indices]
        y_train = target[train_indices]

        X_test = vects[test_indices]
        y_test = target[test_indices]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
               
        model_performance['loss'].append(hamming_loss(y_test, y_pred))
        model_performance['accuracy'].append(accuracy_score(y_test, y_pred))

    fig = plt.figure(figsize=(20, 6))

    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)

    ax1.plot(model_performance['loss'], label='loss per iteration')
    ax1.plot(np.ones(10)*np.mean(model_performance['loss']), '--', label='mean loss')

    ax1.plot(model_performance['accuracy'], label='accuracy per iteration')
    ax1.plot(np.ones(10)*np.mean(model_performance['accuracy']), '--', label='mean accuracy')

    ax1.grid()
    ax1.legend()
    ax1.set_xlabel('fold')
    ax1.set_ylabel('value')
    ax1.set_title('Model Performance')

    cm = normalize(confusion_matrix(y_test, y_pred), axis=1, norm='l1')*100
    
    ax2 = plt.subplot2grid((1, 3), (0, 2))
    sns.heatmap(cm, annot=True, square=True, ax=ax2, cmap='Blues')
    ax2.set_title(f'Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_xticklabels(labels, rotation = 45)
    ax2.set_ylabel('Actual')
    ax2.set_yticklabels(labels, rotation = 45)
    
    return model_performance, cm, model

def train_model_one_vs_rest(model, vects, target, labels, **kwargs):
    model_performance = {
        'loss': [],
        'accuracy': [],
    }
    
    model = OneVsRestClassifier(model)

    for train_indices, test_indices in kf.split(vects, target):
        X_train = vects[train_indices]
        y_train = target[train_indices]

        X_test = vects[test_indices]
        y_test = target[test_indices]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_performance['loss'].append(hamming_loss(y_test, y_pred))
        model_performance['accuracy'].append(accuracy_score(y_test, y_pred))

    fig = plt.figure(figsize=(20, 18))

    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)

    ax1.plot(model_performance['loss'], label='loss per iteration')
    ax1.plot(np.ones(10)*np.mean(model_performance['loss']), '--', label='mean loss')

    ax1.plot(model_performance['accuracy'], label='accuracy per iteration')
    ax1.plot(np.ones(10)*np.mean(model_performance['accuracy']), '--', label='mean accuracy')

    ax1.grid()
    ax1.legend()
    ax1.set_xlabel('fold')
    ax1.set_ylabel('value')
    ax1.set_title('Model Performance')

    cm = []
    
    cm.append(normalize(confusion_matrix(y_test[:, 0], y_pred[:, 0]), axis=1, norm='l1')*100)
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    sns.heatmap(cm[-1], annot=True, square=True, ax=ax2, cmap='Blues')
    ax2.set_title(f'Confusion Matrix \'{labels[0]}\'')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    for i, l in enumerate(labels[1:]):
        cm.append(normalize(confusion_matrix(y_test[:, i+1], y_pred[:, i+1]), axis=1, norm='l1')*100)
        ax2 = plt.subplot2grid((3, 3), (i//3+1, i%3))
        sns.heatmap(cm[-1], annot=True, square=True, ax=ax2, cmap='Blues')
        ax2.set_title(f'Confusion Matrix \'{l}\'')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
    
    return model_performance, cm, model