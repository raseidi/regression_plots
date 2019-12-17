import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from my_plots import single_plot

@single_plot
def plot_score(df, show_score=False):
    sns.lmplot(x=df.y_true, y=df.y_pred, data=df)
    # sns.scatterplot(df.y_true, df.y_pred)
    # linreg = sp.stats.linregress(df.y_true, df.y_pred)
    # plt.plot(df.y_true, linreg.intercept + linreg.slope*df.y_true, 'C3')
    # plt.text(30, 15, 'r-squared = {:.2f}'.format(linreg.rvalue))
    plt.show()

def format_score(y_true, y_pred, name_columns=['y_true', 'y_pred']):
    if len(name_columns) == 2:
        return pd.DataFrame(data=np.c_[y_true, y_pred], columns=name_columns)
    else:
        return 'Set correct name columns.'
