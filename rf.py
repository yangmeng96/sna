import pandas as pd
import numpy as np
import os
import re
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# RF is a function to generate random forest on traning data and prediction on test data.
    # train: training data.
    # test: test data.
    # Y: the outcome. In our setting, it should be "recur".
    # X: all the covariates/predictors, they are comorbidities(all or the first K principal components) the and demographics.
    # For demographics, we only have four predicitors: age, stroke_subtype, sex and race.
    # seed: the random seed.
    # params: range of tuning parameters.


def RF(train,test,params):
    Y_train = train["recur"]
    X_train = train.drop(['recur_same','recur','stroke_date','patient_id'], axis=1)
    Y_test = test["recur"]
    X_test = test.drop(['recur_same','recur','stroke_date','patient_id'], axis=1)

    X_train = pd.get_dummies(X_train, columns=['stroke_subtype', 'sex', 'race'])
    X_test = pd.get_dummies(X_test, columns=['stroke_subtype', 'sex', 'race'])

    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    model = grid_search.best_estimator_

    Y_pred = model.predict(X_test)
    Y_scores = model.predict_proba(X_test)[:, 1] 
    FN = sum((Y_test == 1) & (Y_pred == 0))
    TP = sum((Y_test == 1) & (Y_pred == 1))
    recall = TP/(FN+TP)
    accuracy = accuracy_score(Y_test, Y_pred)

    print("Best params:", grid_search.best_params_)
    print("Accuracy:", accuracy_score(Y_test, Y_pred))
    print("Recall:", recall)
    return(Y_scores, accuracy, recall)


