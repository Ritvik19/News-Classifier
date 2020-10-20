import sys
sys.path.append('../')

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from utils.feature_selection import SelectImportantFeatures


params_lr = {
        'estimator__C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100],
}


models = {
    'Baseline': [TfidfVectorizer(sublinear_tf=True), OneVsRestClassifier(DummyClassifier(strategy='most_frequent'))],
    'LR': [TfidfVectorizer(sublinear_tf=True), OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=100_000))],
    'LR-S': [TfidfVectorizer(sublinear_tf=True), SelectFromModel(DecisionTreeClassifier(class_weight='balanced')), OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=100_000))],
    'LR-H': [TfidfVectorizer(sublinear_tf=True),  GridSearchCV(OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=100_000)), params_lr, cv=5, verbose=3)],
    'LR-SH': [TfidfVectorizer(sublinear_tf=True), SelectFromModel(DecisionTreeClassifier(class_weight='balanced')), GridSearchCV(OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=100_000)), params_lr, cv=5, verbose=3)],
    'LGBM': [TfidfVectorizer(sublinear_tf=True), SelectFromModel(DecisionTreeClassifier(class_weight='balanced')), OneVsRestClassifier(LGBMClassifier(class_weight='balanced'))],
    'LGBM2': [TfidfVectorizer(sublinear_tf=True), SelectImportantFeatures(problem_type='classification'), OneVsRestClassifier(LGBMClassifier(class_weight='balanced'))],
    'LR-S2': [TfidfVectorizer(sublinear_tf=True), SelectImportantFeatures(problem_type='classification'), OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=100_000))],
    'LR-SH2': [TfidfVectorizer(sublinear_tf=True), SelectImportantFeatures(problem_type='classification'), GridSearchCV(OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=100_000)), params_lr, cv=5, verbose=3)],
}

