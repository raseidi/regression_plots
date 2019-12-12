import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from my_plots import single_plot

def format_feat_imp(scores, feature_names, save_path=None):
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

@single_plot
def plot_feat_imp(df, x, y, n_max=5):
    df = df.head(n=n_max)
    g = sns.barplot(x=x, y=y, data=df)
    g.set_xlabel('Importance')
    g.set_ylabel('Feature')
    g.set_title('Feature importance for Boston dataset')
    g.figure.savefig('figures/boston/{}.png'.format(y), bbox_inches="tight")
    plt.show()    