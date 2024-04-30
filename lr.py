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
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import statsmodels.api as sm



# LR is a function to do logistic regression on traning data and prediction on test data.
    # train: training data.
    # test: test data.
    # Y: the outcome. In our setting, it should be "recur".
    # X: all the covariates/predictors, they are comorbidities(all or the first K principal components) the and demographics.
    # For demographics, we only have four predicitors: age, stroke_subtype, sex and race.
    # tolerance: the tolerance value when stopping iteration.
    # seed: the random seed.

def LR(train,test,tolerance,iter,seed):
    Y_train = train["recur_90"]
    X_train = train.drop(['recur_same','recur_30','recur_60','recur_90','stroke_date','patient_id','his_date'], axis=1)
    Y_test = test["recur_90"]
    X_test = test.drop(['recur_same','recur_30','recur_60','recur_90','stroke_date','patient_id',"his_date"], axis=1)

    X_train = pd.get_dummies(X_train, columns=['stroke_subtype', 'sex', 'race','patient_regional_location'])
    X_test = pd.get_dummies(X_test, columns=['stroke_subtype', 'sex', 'race','patient_regional_location'])
    #print(X_train)
    #print(X_test)

    model = LogisticRegression(solver='saga', tol = tolerance, max_iter = iter, random_state = seed)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    Y_scores = model.predict_proba(X_test)[:, 1] 
    FN = sum((Y_test == 1) & (Y_pred == 0))
    TP = sum((Y_test == 1) & (Y_pred == 1))
    recall = TP/(FN+TP)
    accuracy = accuracy_score(Y_test, Y_pred)

    print(model.coef_)
    print("Accuracy:", accuracy)
    print("Recall:", recall)
    return(Y_scores, accuracy, recall)




# def LR_P(train,test):
#     Y_train = train["recur"]
#     X_train = train.drop(['recur_same','recur','stroke_date','patient_id'], axis=1)
#     Y_test = test["recur"]
#     X_test = test.drop(['recur_same','recur','stroke_date','patient_id'], axis=1)

#     X_train = sm.add_constant(pd.get_dummies(X_train, columns=['stroke_subtype', 'sex', 'race'])).astype(float)
#     X_test = sm.add_constant(pd.get_dummies(X_test, columns=['stroke_subtype', 'sex', 'race'])).astype(float)
#     #print(X_train)
#     model = sm.Logit(Y_train, X_train)
#     result = model.fit(method='powell', maxiter=100000, epsilon=1e-4)
#     print(result.summary())
#     Y_pred = result.predict(X_test)

#     Y_pred_class = [1 if prob > 0.5 else 0 for prob in Y_pred]
#     accuracy = accuracy_score(Y_test,Y_pred_class)
#     print(f"Accuracy: {accuracy}")


# LR_P(train,test)



  

