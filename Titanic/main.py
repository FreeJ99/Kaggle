from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

import wandb

import pandas as pd
import numpy as np
from scipy.stats import loguniform
from scipy.stats import expon

from joblib import dump, load
import sys
import warnings
import os

import preprocessing as ppcs 

def run():
    dataset = pd.read_csv("data/train.csv")
    Y = dataset["Survived"]
    X = dataset.drop(["Survived"], axis=1)

    pipe = Pipeline([
        ('feature_removal', ppcs.get_feature_removal()),
        ('col_t', ppcs.get_col_transf()),
        ('model', ensemble.AdaBoostClassifier())
    ])

    param_dist = [
        {
            'model': [LogisticRegression()],
            'model__C': expon(scale=1),
            'col_t__num__poly': [PolynomialFeatures(degree=2)]
        },
        {
            'model': [LinearSVC()],
            'model__C': loguniform(0.000001, 1000000),
            'model__max_iter': [5000]
        },
        {
            'model': [SVC()],
            'model__C': loguniform(0.000001, 1e6),
            'model__kernel': ['rbf', 'poly'],
            'model__gamma': ['scale', 'auto']
        },
        {
            'model': [KNeighborsClassifier()],
            'model__n_neighbors': range(1, 10),
            'model__weights': ['uniform', 'distance']
        },
        {
            'model': [ensemble.RandomForestClassifier()],
            'model__n_estimators': [10,30,100,300,1000,3000],
            'model__criterion': ['gini', 'entropy'],
            'model__min_samples_split': range(2, 30),

        },
        {
            'model': [ensemble.AdaBoostClassifier()]
        },
        {
            'model': [ensemble.GradientBoostingClassifier()],
            'model__loss': ['deviance', 'exponential'],
            'model__n_estimators': [10,30,100,300,1000,3000],
            'model__min_samples_split': range(2, 30),
        }
    ]

    search = RandomizedSearchCV(pipe, param_dist, n_iter=100, cv=3, n_jobs=2,
                            verbose=1, random_state=42, return_train_score=True,
                            scoring = 'accuracy')

    search.fit(X, Y)

    dump(search.cv_results_, "models/results2.joblib")
    #dump(search.best_estimator_, "models/best_estimator.joblib")

if __name__ == "__main__":
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses
    
    wandb.init(project="Titanic")


    run()
   