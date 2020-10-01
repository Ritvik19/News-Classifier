import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

import re, nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer

class TextCleaner(BaseEstimator, TransformerMixin):
    
    def __init__(self, remove_stop=True, stem=False, lemm=True):
        self.remove_stop = remove_stop
        self.stem = stem
        self.lemm = lemm
        
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
        
        self.wnl = WordNetLemmatizer()
        self.ps = PorterStemmer()
        
    def expand_contractions(self, text):
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"what's", "what is ", text) 
        text = re.sub(r"'s", " ", text)
        text = re.sub(r"'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"'re", " are ", text)
        text = re.sub(r"'d", " would ", text)
        text = re.sub(r"'ll", " will ", text)  
        return text

    remove_url = lambda self, x: re.sub(r'''((http[s]?://)[^ <>'"{}|\^`[\]]*)''', r' ', x)
    remove_handles = lambda self, x: re.sub(r'@\S+', r' ', x)
    remove_incomplete_last_word = lambda self, x: re.sub(r'\S+…', r' ', x)
    remove_punc = lambda self, x : re.sub(r"\W", ' ', x)
    remove_num = lambda self, x : re.sub(r"\d", ' ', x)
    remove_css = lambda self, x: re.sub(r'<style.*>[\s\S]+</style>', ' ', x)
    remove_js = lambda self, x: re.sub(r'<script.*>[\s\S]*</script>', ' ', x)
    remove_html = lambda self, x: re.sub(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});", ' ', x)
    remove_extra_spaces = lambda self, x : re.sub(r"\s+", ' ', x)
    remove_shortwords = lambda self, x: ' '.join(word for word in x.split() if len(word) > 2)
    remove_unusualwords = lambda self, x: ' '.join(word for word in x.split() if len(word) < 16)
    lower_case = lambda self, x : x.lower()
    remove_stopwords = lambda self, x: ' '.join(word for word in x.split() if word not in self.stop_words)
    wnl_lemmatize = lambda self, x: ' '.join(self.wnl.lemmatize(word) for word in x.split())
    ps_stem = lambda self, x: ' '.join(self.ps.stem(word) for word in x.split())
    
    def clean(self, x):
        x = str(x)
        x = self.remove_url(x)
        x = self.remove_css(x)
        x = self.remove_js(x)
        x = self.remove_html(x)
        x = self.lower_case(x)
        x = self.expand_contractions(x)
        x = self.remove_punc(x)
        x = self.remove_num(x)
        x = self.remove_extra_spaces(x)
        x = self.remove_shortwords(x)
        x = self.remove_unusualwords(x)
        x = self.remove_incomplete_last_word(x)
        if self.remove_stop:
            x = self.remove_stopwords(x)
        if self.stem:
            x = self.ps_stem(x)
        if self.lemm:
            x = self.wnl_lemmatize(x)
        return x
    
    def fit(self, X):
        return self
    
    def transform(self, X):
        transformed = []
        for x in X:
            transformed.append(self.clean(x))
        return np.array(transformed)
    
    def fit_transform(self, X):
        return self.transform(X)