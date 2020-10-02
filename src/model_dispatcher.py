import pickle, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def save_model(model, name, dataset_id):
    pickle.dump(model, open(f'../models/{name}-{dataset_id}.pkl', 'wb'))

def save_preprocessing(preprocessor, num_features, vocab, name, dataset_id):
    output_directory = os.path.join(os.path.dirname(os.getcwd()), 'performance', f'{name}-{dataset_id}')
    model_directory = os.path.join(os.path.dirname(os.getcwd()), 'models')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
        
    with open(os.path.join(output_directory, 'Preprocessing-Report.txt'), 'w') as f:
        f.write(f'<pre>{preprocessor}</pre>\n')
        f.write(f'<p><b>Num Features: </b>{num_features}</p>\n')

    pickle.dump(preprocessor, open(os.path.join(model_directory, f'Preprocessor-{name}-{dataset_id}.pkl'), 'wb'))
    
    wordcloud = WordCloud(background_color='white', max_words=100000, width=1600, height=800).generate(' '.join((vocab)))
    wordcloud.to_file(os.path.join(output_directory, 'Vocab.png'))
        
def save_results_classification(classification_report, confusion_matrix, performance, name, dataset_id, problem=None):
    output_directory = os.path.join(os.path.dirname(os.getcwd()), 'performance', f'{name}-{dataset_id}')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    
    with open(os.path.join(output_directory, 'Classification-Report.txt'), 'w') as f:
        f.write('<div class="row">\n')
        f.write((f'<div class="col-md-6"><pre>\n{classification_report}\n</pre></div>\n'))
        f.write('<div class="col-md-6"><table class="table">\n')
        f.write(f'<tr><th>Log loss</th><td>{round(np.mean(performance["log_loss"]), 4)}</td></tr>\n')
        f.write(f'<tr><th>AUROC</th><td>{round(np.mean(performance["auroc"]), 4)}</td></tr>\n')
        f.write('</table></div></div>')
        
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
    y_pred = y_pred.reshape(-1)
    output_directory = os.path.join(os.path.dirname(os.getcwd()), 'performance', f'{name}-{dataset_id}')
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
        
    with open(os.path.join(output_directory, 'Regression-Report.txt'), 'w') as f:
        f.write('<table class="table">\n')
        f.write(f'<tr><th>MSE</th><td>{round(np.mean(performance["mse"]), 4)}</td></tr>\n')
        f.write(f'<tr><th>MAE</th><td>{round(np.mean(performance["mae"]), 4)}</td></tr>\n')
        f.write(f'<tr><th>R2</th><td>{round(np.mean(performance["r2score"]), 4)}</td></tr>\n')
        f.write('</table>')
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
            