import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def check_null(data):
    print(data.isnull().sum())
    fig = plt.figure(figsize=(20,6))
    sns.heatmap(data.isnull(), cmap='viridis', yticklabels=False, cbar=False)
    
def univariate_categorical(data, col):
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(col, fontsize=24)

    ax1 = plt.subplot2grid((1, 3), (0, 0))
    data[col].value_counts().plot.pie(legend=True, autopct='%1.0f%%', ax=ax1)

    ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    data[col].value_counts().plot.bar()
    ax2.grid()
    x_offset = -0.03
    y_offset = 0.05
    for p in ax2.patches:
        b = p.get_bbox()
        val = "{:.2f}".format(b.y1 + b.y0)        
        ax2.annotate(val, ((b.x0 + b.x1)/2 + x_offset, b.y1 + y_offset))
        
def univariate_continuous(data, col):
    fig, ax = plt.subplots(figsize=(20, 6))
    fig.suptitle(col, fontsize=24)
    
    sns.distplot(data['Fare'], ax=ax)
    ax.grid()
    
def bivariate_continuous_categorical(data, x, y):
    fig, ax = plt.subplots(nrows=2, figsize=(20,12))
    fig.suptitle(f'{x} vs {y}', fontsize=24)
    sns.distplot(data[x], ax=ax[0])
    ax[0].grid()
    ax[0].set_title(f'Univariate Analysis {x}', fontsize=18)

    for i in data[y].unique():
        sns.distplot(data[data[y] == i][x], ax=ax[1], label=i)
    ax[1].set_title(f'Bivariate Analysis {y} vs {x}', fontsize=18)
    ax[1].grid()
    ax[1].legend()
    
def bivariate_categorical_continuous(data, x, y):
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(f'{x} vs {y}', fontsize=24)

    ax1 = plt.subplot2grid((1, 3), (0, 0))
    ax1.set_title(f'Univariate Analysis {x}', fontsize=18)
    data[x].value_counts().plot.pie(legend=True, autopct='%1.0f%%', ax=ax1)

    ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)

    for i in data[x].unique():
        sns.distplot(data[data[x] == i][y], ax=ax2, label=i)
    ax2.set_title(f'Bivariate Analysis {y} vs {x}', fontsize=18)
    ax2.grid()
    ax2.legend()

def bivariate_continuous_continuous(data, x, y):
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(f'{x} vs {y}', fontsize=24)
    
    ax1 = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    sns.distplot(data[x], ax=ax1)
    ax1.grid()
    ax1.set_title(f'Univariate Analysis {x}', fontsize=18)

    ax2 = plt.subplot2grid((1, 3), (0, 2))
    data[[x, y]].plot.scatter(x=x, y=y, ax=ax2, grid=True)
    ax2.set_title(f'Bivariate Analysis {y} vs {x}', fontsize=18)
    
def bivariate_categorical_categorical(data, x, y):
    fig, ax = plt.subplots(1, 2, figsize=(20, 6))
    fig.suptitle(f'{x} vs {y}', fontsize=24)
    
    data[x].value_counts().plot.bar(ax=ax[0], grid=True)
    ax[0].set_title(f'Univariate Analysis {x}', fontsize=18)
    sns.heatmap(pd.crosstab(data[x], data[y]), ax=ax[1], annot=True, cmap='Blues', square=True, fmt='d')
    ax[1].set_title('Bivariate Analysis {y} vs {x}', fontsize=18)
    
def wordcloud_df(data, title=''):
    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(title, fontsize=24)

    ax1 = plt.subplot2grid((1, 3), (0, 0))
    pd.Series(' '.join(data).split()).value_counts().head(20)[::-1].plot.barh(ax=ax1, grid=True)
    ax1.set_title('Most Frequent Words', fontsize=18)

    ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    wordcloud = WordCloud().generate(' '.join((data.values)))
    ax2.imshow(wordcloud)
    ax2.set_title('Word Cloud', fontsize=18)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_visible(False)
    
def correlation_heatmap(data):
    fig, ax = plt.subplots(figsize=(10, 10))

    mask = np.zeros_like(data.corr(), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask[np.diag_indices_from(mask)] = True

    sns.heatmap(data.corr(), mask=mask, cmap="RdBu", ax=ax, annot=True, square=True, center=0)