from os import name, system

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
            '[0] No method\n'
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


def impute(df, choice, inplace=False):
    if choice == 0:
        return no_impute(df)
    elif choice == 1:
        return impute_mean(df, inplace=inplace)
    elif choice == 2:
        return impute_interpolation(df, inplace=inplace)
    elif choice == 3:
        return impute_hotdeck(df, inplace=inplace)
    elif choice == 4:
        return impute_regression(df)
    else:
        raise ValueError

def no_impute(df):
    return df.dropna()

def impute_mean(df, inplace=False):
    return df.fillna(df.mean(), inplace=inplace)


def impute_interpolation(df, inplace=False):
    return df.interpolate(inplace=inplace)


def impute_hotdeck(df, inplace=False):
    return df.fillna(method='ffill', inplace=inplace)


def impute_regression(df):
    mean_filled = impute_mean(df)
    for column in df.columns:
        regression_values = get_linear_regression_values(
            mean_filled.index.values.reshape(-1, 1),
            mean_filled.loc[:, column].values.reshape(-1, 1),
        )
        mean_filled.loc[df[column].isnull(), column] = regression_values[df[column].isnull()]
    return mean_filled


def get_linear_regression_values(x, y):
    regression_model = LinearRegression()
    regression_model.fit(x, y)
    print('\nRegressor coeficient:\t' + str(regression_model.coef_))
    print('Regressor intercept:\t' + str(regression_model.intercept_))
    return regression_model.predict(x)
