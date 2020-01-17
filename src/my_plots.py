import os
import functools
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

'''
    /usr/local/lib/python3.6/dist-packages/matplotlib/rcsetup.py
    from docummentation:
        fontsizes = ['xx-small', 'x-small', 'small', 'medium', 'large',
                    'x-large', 'xx-large', 'smaller', 'larger']
        
        ls_mapper = {'-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted'}
'''
GLOBAL_CFG = {
    'axes.labelsize': 'medium', # x and y labels
    'xtick.labelsize': 'small', # x and y values
    'ytick.labelsize': 'small',
    'grid.linestyle': '--',
    'figure.figsize': (12, 8),
    'savefig.dpi': 100,
    'axes.labelweight': 'bold'
}

def single_plot(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        n_max = kwargs['n_max'] if 'n_max' in kwargs else 5
        palette = sns.color_palette('Blues_d', n_colors=n_max)
        sns.set(style='darkgrid', palette=palette, color_codes=False, rc=GLOBAL_CFG)
        func(*args, **kwargs)
    return inner

@single_plot
def plot_feat_imp(df, x, y, n_max=5):
    df = df.head(n=n_max)
    g = sns.barplot(x=x, y=y, data=df)
    g.set_xlabel('Importance')
    g.set_ylabel('Feature')
    g.set_title('Feature importance for Boston dataset')
    # g.figure.savefig('figures/boston/{}.png'.format(y), bbox_inches="tight")
    plt.show()    

@single_plot
def regplot(df, x, y):
    '''
        df: dataframe
        x, y: column names
    '''
    g = sns.regplot(x=df[x], y=df[y], data=df)

    mse_std = np.std(np.power(df[x] - df[y], 2)) # std squared errors
    squared_errors = np.power(df[x] - df[y], 2)
    Q3 = np.percentile(squared_errors, 75)
    Q1 = np.percentile(squared_errors, 25)

    for line in df.index.values:
        # if(np.power(df[x][line] - df[y][line], 2) > mse_std):
        if not Q1 <= np.power(df[x][line] - df[y][line], 2) <= Q3:
            # print('ID: {}, SqE: {}'.format(line, np.power(df[x][line] - df[y][line], 2)))
            g.text(df[x][line]+0.1, df[y][line],
                line, horizontalalignment='left', rotation=45,
                size='small', color='black', weight='semibold')
    plt.show()