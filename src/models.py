from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from xgboost import  XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from keras.models import Sequential
from keras.applications import MobileNetV2
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D

def cnn(image_size, output_classes):
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=image_size, activation='relu', padding='same'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    if output_classes == 2:
        classifier.add(Dense(units=1, activation='sigmoid'))
        classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    else:
        classifier.add(Dense(units=output_classes, activation='sigmoid'))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return classifier

def tl_cnn(image_size, output_classes):
    base_model = MobileNetV2(input_shape=image_size, include_top=False, weights='imagenet')
    base_model.trainable = True
    global_average_layer = GlobalAveragePooling2D()
    
    if output_classes == 2:
        prediction_layer = Dense(units=1, activation = 'sigmoid')
    else:
        prediction_layer = Dense(units=output_classes, activation = 'sigmoid')
    
    neuralnetwork = Sequential([
        base_model,
        global_average_layer,
        prediction_layer
        ])
    if output_classes == 2:
        neuralnetwork.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    else:
        neuralnetwork.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return neuralnetwork

xgboost_params = {
    'estimator__eta': [0.01, 0.015, 0.025, 0.050, 0.100,],
    'estimator__gamma': [0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    'estimator__max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
    'estimator__min_child_weight': [1, 3, 5, 7],
    'estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'estimator__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'estimator__lambda': [0.01, 0.1, 1.0],
    'estimator__alpha': [0.01, 0.1, 1.0]
}

lightgbm_params = {
    'estimator__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'estimator__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'estimator__reg_lambda': [0.01, 0.1, 1.0],
    'estimator__reg_alpha': [0.01, 0.1, 1.0]
}

lr_params = {
    'estimator__C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]
}

models = {
    'DC': OneVsRestClassifier(DummyClassifier(strategy='uniform', random_state=19)),
    'LR': OneVsRestClassifier(LogisticRegression(class_weight='balanced', random_state=19, max_iter=1_00_000)),
    'RF': OneVsRestClassifier(RandomForestClassifier(n_estimators=10, class_weight='balanced', random_state=19)),
    'BLR': OneVsRestClassifier(BaggingClassifier(base_estimator=LogisticRegression(class_weight='balanced', random_state=19), n_estimators=100, bootstrap_features=True, random_state=19)),
    'GB': OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100, random_state=19, max_depth=5)),
    'XGB': RandomizedSearchCV(OneVsRestClassifier(XGBClassifier(random_state=19)), xgboost_params, n_iter=50, verbose=10, scoring='roc_auc_ovr'),
    'LGBM': RandomizedSearchCV(OneVsRestClassifier(LGBMClassifier(boosting_type='gbdt', max_depth=-1, n_estimators=100, class_weight='balanced', random_state=19)), lightgbm_params, n_iter=50, verbose=10, scoring='roc_auc_ovr'),
    'LR-G': GridSearchCV(OneVsRestClassifier(LogisticRegression(class_weight='balanced', random_state=19, max_iter=1_00_000)), lr_params, cv=4, scoring='roc_auc_ovr', verbose=10)
}