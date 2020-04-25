from os import name, system

import pandas
from sklearn.linear_model import LinearRegression


def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')


def read_choice():
    while True:
        clear()
        choice = int(input(
            'Method of imputation:\n'
            '[1] Mean\n'
            '[2] Interpolation\n'
            '[3] Hot Deck\n'
            '[4] Regression\n'
            'Choice:\t'
        ))
        if 1 < choice or choice < 4:
            break
    clear()
    return choice


def na_fraction(df):
    return df.isna().sum().sum() / df.count().sum()


def get_stats(df):
    stats = df.describe().drop(['count', 'min', 'max']).T
    stats['%mv'] = df.isna().sum()
    return stats


def impute(df, choice, column_name, inplace=False):
    if choice == 1:
        return impute_mean(df, inplace=inplace)
    elif choice == 2:
        return impute_interpolation(df, inplace=inplace)
    elif choice == 3:
        return impute_hotdeck(df, inplace=inplace)
    elif choice == 4:
        return impute_regression(df, column_name=column_name)
    else:
        raise ValueError


def impute_mean(df, inplace=False):
    return df.fillna(df.mean(), inplace=inplace)


def impute_interpolation(df, inplace=False):
    return df.interpolate(inplace=inplace)


def impute_hotdeck(df, inplace=False):
    return df.fillna(method='ffill', inplace=inplace)


def impute_regression(df, column_name):
    mean_filled = impute_mean(df)
    regression_model = LinearRegression()
    regression_model.fit(
        mean_filled.index.values.reshape(-1, 1),
        mean_filled.loc[:, column_name].values.reshape(-1, 1)
    )
    regression_values = regression_model.predict(mean_filled.index.values.reshape(-1, 1))
    for i, j in enumerate((df[pandas.isnull(df[column_name])]).index):
        mean_filled.loc[j, column_name] = regression_values[i]
    return mean_filled
