# News-Classifier

Tag News Articles with Categories to which they might belong using Machine Learning
___

### Data Preprocessing

* URLs are removed from the text
* Text is lowercased
* Contractions are expanded
* Punctuations are removed from the text
* Digits are removed
* Extra white spaces are removed from the text
* Stop words are removed
* TFIDF Vectors are created (1grams)


## Approach 1

Models Trained on Dataset available on [Kaggle](https://www.kaggle.com/rmisra/news-category-dataset) as a multiclass problem using TFIDF vectors as features

This approach tags news articles as the following 8 tags:

Lifestyle, Politics, Global, Miscellaneous, Entertainment, Education, Business, Sports

___

## Approach 2

Models Trained on manually scraped data from [inshorts](https://inshorts.com/en/read) as a multi label problem using TFIDF vectors as features

This approach tags news articles as the following 7 tags:

National, Sports, World, Politics, Technology, Entertainment, Hatke
