import os
import functools
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
def plot_test():
    g = sns.lineplot(x=[1,2,3], y=[4,5,6])
    g.set_title('Title')
    g.set_xlabel('X_label')
    g.set_ylabel('Y_label')
    plt.show()