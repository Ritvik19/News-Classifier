from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier

models = {
    'LR': [TfidfVectorizer(sublinear_tf=True), OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=100_000))],
    'LR-S': [TfidfVectorizer(sublinear_tf=True), SelectFromModel(LogisticRegression(class_weight='balanced', max_iter=100_000)), OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=100_000))],
    'LR-S2': [TfidfVectorizer(sublinear_tf=True), SelectFromModel(DecisionTreeClassifier(class_weight='balanced')), OneVsRestClassifier(LogisticRegression(class_weight='balanced', max_iter=100_000))],
    'LR-S-A': [TfidfVectorizer(sublinear_tf=True), SelectFromModel(DecisionTreeClassifier(class_weight='balanced')), OneVsRestClassifier(AdaBoostClassifier(LogisticRegression(class_weight='balanced', max_iter=100_000)))],
    'LR-S-B': [TfidfVectorizer(sublinear_tf=True), SelectFromModel(DecisionTreeClassifier(class_weight='balanced')), OneVsRestClassifier(BaggingClassifier(LogisticRegression(class_weight='balanced', max_iter=100_000), bootstrap_features=True))],
    'RF': [TfidfVectorizer(sublinear_tf=True), SelectFromModel(DecisionTreeClassifier(class_weight='balanced')), OneVsRestClassifier(RandomForestClassifier(class_weight='balanced'))],
    'LGBM': [TfidfVectorizer(sublinear_tf=True), SelectFromModel(DecisionTreeClassifier(class_weight='balanced')), OneVsRestClassifier(LGBMClassifier(class_weight='balanced'))],
}

