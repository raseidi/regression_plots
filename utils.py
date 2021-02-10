import numpy as np
import pandas as pd

def read_data(path='data/kaggle_house_prices/train.csv', return_type_cols=False):
    df = pd.read_csv('data/kaggle_house_prices/train.csv', index_col=0)
    df = df.loc[:, df.isnull().sum() == 0]
    if return_type_cols:
        df_categorical = df.loc[:, df.dtypes == 'object'].columns.values
        df_numerical = df.loc[:, df.dtypes != 'object'].columns.values
        return df, df_categorical, df_numerical
    else:
        return df