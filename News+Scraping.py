import requests
import bs4
import pandas as pd
from IPython.display import clear_output
import time
from tqdm.auto import tqdm
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
FILEPATH = "E:/Scrapped-Data/InshortsScraped.csv"
news = pd.read_csv(FILEPATH)

n1 = len(news)
print(n1)


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
    else:
        print('Something went wrong')
    return news

for i in range(12):
    print('Current Progress', i+1, '/ 12', end=' -> ')
    news = scrape(i, news)
print('done')

news = news.reset_index()
news = news.drop(['index'], axis=1)
n2 = len(news)
print(n2)

start = time.time()
for i in tqdm(range(n2)):
    for j in range(i+1, n2):
        if news.iloc[i]['news'] == news.iloc[j]['news']:
            print(i, j)
            news.iloc[i][1:] = (news.iloc[i][1:]|news.iloc[j][1:]).astype('int')
            news.iloc[j][1:] = (news.iloc[i][1:]|news.iloc[j][1:]).astype('int')
            

print()
news = news.drop_duplicates()
n0 = len(news)

end = time.time()
news.to_csv(FILEPATH, index=False)
print((end - start)/60, 'Minutes')
print('Initial', n1)
print('After Scraping', n2)
print('Reduced', n0)
print(n0-n1, 'New Articles')

