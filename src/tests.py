import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import my_plots
import format_values

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df.sample(frac=1, random_state=44)

# feature importances 
def f1():
    boston = sklearn_to_df(load_boston())
    reg = DecisionTreeRegressor()
    
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    scores = cross_validate(reg, boston.drop('target', axis=1), boston['target'], cv=5,
        scoring=scoring, return_train_score=True, return_estimator=True)
    
    list_fi = [fi.feature_importances_ for fi in scores['estimator']]
    df = format_values.feat_imp(list_fi, boston.columns[:-1])

    my_plots.plot_feat_imp(df, 'Feature', 'Importance', n_max=10)

# regression line
def f2():
    boston = sklearn_to_df(load_boston())
    X_train, X_test, y_train, y_test = train_test_split(boston.drop('target', axis=1), boston['target'])
    reg = DecisionTreeRegressor()
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)

    df = format_values.ytrue_ypred(y_test, pred)
    my_plots.regplot(df, 'y_true', 'y_pred')

if __name__ == "__main__":
    # f1()
    f2()



def f(obj): pass
hasattr(f, '__get__')

class Class(object):
    @f
    @classmethod
    def cmethod(cls):
        pass 

Class.cmethod()     