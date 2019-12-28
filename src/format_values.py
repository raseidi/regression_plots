import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def feat_imp(scores, feature_names, save_path=None):
    '''
        scores: list of feature importances (if scores are retrieved
        from cross validation, pass scores like:
            [fi.feature_importances_ for fi in scores['estimator']]
        )
        feature_names: columns of original dataframe
        save_path: None is you do not wish to save it, the path otherwise
        returns feature importances (mean for cv) as dataframe
    '''
    scores = np.array(scores)
    df = pd.DataFrame(scores if scores.ndim == 1 else scores[0], index=feature_names, columns=['Importance'])
    if scores.ndim > 1:
        for estimator in scores[1:]:
            df = (df + pd.DataFrame(estimator,
                                    index=feature_names,
                                    columns=['Importance']))

    df = df/scores.ndim
    df = df.sort_values(by='Importance', ascending=False)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Feature'}, inplace=True)

    # ToDo: generic save_path param
    if save_path is None:
        df.to_csv('data/boston/feature_importance.csv', index=False)
    return df
    
    
def ytrue_ypred(y_true, y_pred, name_columns=['y_true', 'y_pred']):
    if len(name_columns) == 2:
        return pd.DataFrame(data=np.c_[y_true, y_pred], columns=name_columns, index=y_true.index)
    else:
        return 'Set correct name columns.'
