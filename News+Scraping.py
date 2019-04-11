
# coding: utf-8

# In[1]:


import requests
import bs4
import pandas as pd
from IPython.display import clear_output


# In[2]:


urls = [
    'https://inshorts.com/en/read',
    'https://inshorts.com/en/read/national',
    'https://inshorts.com/en/read/sports',
    'https://inshorts.com/en/read/world',
    'https://inshorts.com/en/read/politics',
    'https://inshorts.com/en/read/technology',
    'https://inshorts.com/en/read/startup',
    'https://inshorts.com/en/read/entertainment',
    'https://inshorts.com/en/read/miscellaneous',
    'https://inshorts.com/en/read/hatke',
    'https://inshorts.com/en/read/science',
    'https://inshorts.com/en/read/automobile'
]


# In[3]:


print(len(urls))


# In[4]:


# news = pd.DataFrame(columns=['news', 'national', 'sports', 'world', 'politics', 'technology', 'entertainment', 'hatke'])
# news.to_csv('InshortsScraped.csv', index=False)


# In[5]:


news = pd.read_csv('InshortsScraped.csv')


# In[6]:


def scrape(news_class, news):
    news_ = pd.DataFrame(columns=news.columns)
    res = requests.get(urls[news_class])
    if res.status_code == requests.codes.ok:
        ressoup = bs4.BeautifulSoup(res.text, 'lxml')
        elems = ressoup.select('.news-card-title.news-right-box')
        for i, e in enumerate(elems):
            x = e.getText().strip().split('\n')[0]
            #print(i, str(x))
            if news_class == 0:
                news_.loc[i] = [x, 0, 0, 0, 0, 0, 0, 0]
            elif news_class == 1:
                news_.loc[i] = [x, 1, 0, 0, 0, 0, 0, 0]
            elif news_class == 2:
                news_.loc[i] = [x, 0, 1, 0, 0, 0, 0, 0]
            elif news_class == 3:
                news_.loc[i] = [x, 0, 0, 1, 0, 0, 0, 0]
            elif news_class == 4:
                news_.loc[i] = [x, 0, 0, 0, 1, 0, 0, 0]
            elif news_class == 5:
                news_.loc[i] = [x, 0, 0, 0, 0, 1, 0, 0]
            elif news_class == 6:
                news_.loc[i] = [x, 0, 0, 0, 0, 1, 0, 0]
            elif news_class == 7:
                news_.loc[i] = [x, 0, 0, 0, 0, 0, 1, 0]
            elif news_class == 8:
                news_.loc[i] = [x, 0, 0, 0, 0, 0, 0, 0]
            elif news_class == 9:
                news_.loc[i] = [x, 0, 0, 0, 0, 0, 0, 1]
            elif news_class == 10:
                news_.loc[i] = [x, 0, 0, 0, 0, 1, 0, 0]
            elif news_class == 11:
                news_.loc[i] = [x, 0, 0, 0, 0, 1, 0, 0]
        news = pd.concat([news, news_], axis=0)
        print(len(news))
    else:
        print('Something went wrong')
    return news


# In[7]:


for i in range(12):
    print('Current Progress', i+1, '/ 12')
    news = scrape(i, news)
    clear_output(wait=True)
print('done')


# In[8]:




# In[9]:


news.head()


# In[10]:


news = news.reset_index()


# In[11]:


news = news.drop(['index'], axis=1)


# In[12]:


n = len(news)
print(n)


# In[13]:


for i in range(n):
    for j in range(i+1, n):
        if news.iloc[i]['news'] == news.iloc[j]['news']:
            print('-------')
            print(i, j)
            news.iloc[i][1:] = (news.iloc[i][1:]|news.iloc[j][1:]).astype('int')
            news.iloc[j][1:] = (news.iloc[i][1:]|news.iloc[j][1:]).astype('int')
            

print()
# In[14]:


news


# In[15]:


news = news.drop_duplicates()
print(len(news))


# In[16]:


news.to_csv('InshortsScraped.csv', index=False)


# In[ ]:




