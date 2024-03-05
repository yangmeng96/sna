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



# LR is a function to do logistic regression on traning data and prediction on test data.
    # train: training data.
    # test: test data.
    # Y: the outcome. In our setting, it should be "recur".
    # X: all the covariates/predictors, they are comorbidities(all or the first K principal components) the and demographics.
    # For demographics, we only have four predicitors: age, stroke_subtype, sex and race.
    # tolerance: the tolerance value when stopping iteration.
    # seed: the random seed.

def LR(train,test,tolerance,iter,seed):
    Y_train = train["recur"]
    X_train = train.drop(['recur_same','recur','stroke_date','patient_id'], axis=1)
    Y_test = test["recur"]
    X_test = test.drop(['recur_same','recur','stroke_date','patient_id'], axis=1)

    X_train = pd.get_dummies(X_train, columns=['stroke_subtype', 'sex', 'race'])
    X_test = pd.get_dummies(X_test, columns=['stroke_subtype', 'sex', 'race'])
    #print(X_train)
    #print(X_test)

    model = LogisticRegression(solver='saga', tol = tolerance, max_iter = iter, random_state = seed)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    print(model.coef_)
    print("Accuracy:", accuracy_score(Y_test, Y_pred))