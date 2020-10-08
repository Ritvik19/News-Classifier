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

### Approach

Models Trained on manually scraped data from
* Inshorts
* ANI
* India TV
* Janta Ka Reporter
* OpIndia
* PostCard News
* Swarajya
* TFIPost
* TheWeek
* TheWire 
  
and a dataset available on [Kaggle](https://www.kaggle.com/rmisra/news-category-dataset)

as a multi label problem using TFIDF vectors as features

This approach tags news articles as the following 9 tags:

    National, Sports, World, Politics, Technology, Entertainment, Business, Lifestyle, Hatke
