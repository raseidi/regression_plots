import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from my_plots import single_plot

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df.sample(frac=1, random_state=44)


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

def plot_metrics(df, metrics=['r2', 'test_neg_mean_squared_error']):
    metrics = list(metrics)
    for m in metrics:
        plt.plot(df[m])

if __name__ == "__main__":
    from sklearn.datasets import load_boston
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import train_test_split, cross_validate, KFold

    reg = DecisionTreeRegressor()
    boston = sklearn_to_df(load_boston())
    scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    scores = cross_validate(reg, boston.drop('target', axis=1), boston['target'], cv=10,
        scoring=scoring, return_train_score=True, return_estimator=True)
    
    # feature importances 
    list_fi = [fi.feature_importances_ for fi in scores['estimator']]
    df = format_feat_imp(list_fi, boston['feature_names'])
    # plot_feat_imp(df, 'Feature', 'Importance', n_max=10)

    # metrics (predictions)
    metrics = [key for key in scores.keys() if key.startswith('test')]
    for m in metrics:
            boston[m] = scores['test_r2']

    print('r2_mean: {:.2f}, r2_std: {:.2f}'.format(scores['test_r2'].mean(), scores['test_r2'].std()))
    scores['estimator'][0].predict(boston.drop('target', axis=1))
    format_metrics()

    