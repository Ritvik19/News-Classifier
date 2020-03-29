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


### Approach 1

Models Trained on Dataset available on [Kaggle](https://www.kaggle.com/rmisra/news-category-dataset) as a multiclass problem using TFIDF vectors as features

This approach tags news articles as the following 8 tags:

Lifestyle, Politics, Global, Miscellaneous, Entertainment, Education, Business, Sports

Data Dimensions:
* X shape: (200853, 77641)
* Y shape: (200853, 8)

Algorithms used:

Algorithm | Mean CV Score | Standard CV Score
:---:|:---:|:--:
Logistic Regression | 0.56 | 0.14
Multinomial NB | 0.43 | 0.19
Bernoulli NB | 0.55 | 0.14
SGD Classifier 'hinge' | 0.44 | 0.17
SGD Classifier 'log' | 0.38 | 0.18
SGD Classifier 'perceptron' | 0.52 | 0.10

The results of all models are used to give the final predictions
___

## Approach 2

Models Trained on manually scraped data from [inshorts](https://inshorts.com/en/read) as a multi label problem using TFIDF vectors as features

This approach tags news articles as the following 7 tags:

National, Sports, World, Politics, Technology, Entertainment, Hatke

Data Dimensions:
* X shape: (34539, 23577)
* Y shape: (34539, 7)

Algorithms used:

Algorithm | Mean CV Score | Standard CV Score
:---:|:---:|:--:
Logistic Regression | 0.56 | 0.03
Multinomial NB | 0.53 | 0.03
Bernoulli NB | 0.65 | 0.03
SGD Classifier 'hinge' | 0.60 | 0.03
SGD Classifier 'log' | 0.42 | 0.03
SGD Classifier 'perceptron' | 0.61 | 0.02

The results of all models are used to give the final predictions