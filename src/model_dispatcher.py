import pickle, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def save_model(model, name, dataset_id):
    pickle.dump(model, open(f'../models/{name}-{dataset_id}.pkl', 'wb'))
    
def save_results_classification(classification_report, confusion_matrix, performance, name, dataset_id, problem=None):
    output_directory = os.path.join(os.path.dirname(os.getcwd()), 'performance', f'{name}-{dataset_id}')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    with open(os.path.join(output_directory, 'Classification-Report.txt'), 'w') as f:
        f.write(f'Log loss: {round(np.mean(performance["log_loss"]), 4)}\n')
        f.write(f'AUROC: {round(np.mean(performance["auroc"]), 4)}\n')
        f.write('\n\nClassification Report\n\n')
        f.write((classification_report))
    if problem == 'ml':
        for i in range(len(confusion_matrix)):
            fig, ax = plt.subplots()
            fig.suptitle(f'Class {i+1}')
            sns.heatmap(confusion_matrix[i], annot=True, square=True, ax=ax, cmap='Blues')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            fig.savefig(os.path.join(output_directory, f'Confusion-Matrix-{i+1}.png'))
    else:
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix, annot=True, square=True, ax=ax, cmap='Blues')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        fig.savefig(os.path.join(output_directory, 'Confusion-Matrix.png'))
            
def save_results_regression(performance, name, dataset_id, y_true, y_pred):
    output_directory = os.path.join(os.path.dirname(os.getcwd()), 'performance', f'{name}-{dataset_id}')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
        
    with open(os.path.join(output_directory, 'Regression-Report.txt'), 'w') as f:
        f.write(f'MSE: {round(np.mean(performance["mse"]), 4)}\n')
        f.write(f'MAE: {round(np.mean(performance["mae"]), 4)}\n')
        f.write(f'R2: {round(np.mean(performance["r2score"]), 4)}\n')
        
        fig, ax = plt.subplots()
        fig.suptitle(f'Distribution of Error')
        sns.distplot(y_true-y_pred, kde=False, ax=ax)
        fig.savefig(os.path.join(output_directory, f'Error-Distribution.png'))
        
        fig, ax = plt.subplots()
        fig.suptitle(f'Actual Vs Predicted')
        sns.scatterplot(y_true, y_pred)
        mn = min(np.min(y_true), np.min(y_pred))
        mx = max(np.max(y_true), np.max(y_pred))
        line = np.linspace(mn, mx, len(y_true))
        ax.plot(line, line, 'orange')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        fig.savefig(os.path.join(output_directory, f'Actual-Vs-Predicted.png'))
            