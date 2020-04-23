from os import name, system

import numpy
import pandas
import seaborn
from matplotlib import pyplot


def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')


def na_fraction(df):
    return df.isna().sum().sum() / df.count().sum()


def get_stats(df):
    stats = data.describe().drop(['count', 'min', 'max']).T
    stats['%mv'] = df.isna().sum()
    return stats


def impute_mean(df, inplace=False):
    return df.fillna(df.mean(), inplace=inplace)


def impute_interpolate(df, inplace=False):
    return df.interpolate(inplace=inplace)


def impute_hotdeck(df, inplace=False):
    return df.fillna(method='ffill', inplace=inplace)


def impute_regression(df, inplace=False):
    return df.interpolate(method='linear', limit_direction='both', inplace=inplace)


pandas.options.display.float_format = '{:.2f}'.format
data = pandas.read_csv('./datasets/horse.csv')

data = data.iloc[:, data.applymap(numpy.isreal).all().values]
data.drop(
    ['nasogastric_reflux_ph',
     'abdomo_protein'],
    axis=1,
    inplace=True
)

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

before = get_stats(data)
print(
    '- - - - - - - - Before imputation - - - - - - - -\n' +
    'Missing values:\t' + '{:.2%}'.format(na_fraction(data)) + '\n' +
    before.to_string()
)

if choice == 1:
    impute_mean(data, inplace=True)
elif choice == 2:
    impute_interpolate(data, inplace=True)
elif choice == 3:
    impute_hotdeck(data, inplace=True)
elif choice == 4:
    impute_regression(data, inplace=True)
else:
    raise ValueError

after = get_stats(data)
print(
    '- - - - - - - - After imputation - - - - - - - -\n' +
    'Missing values:\t' + '{:.2%}'.format(na_fraction(data)) + '\n' +
    after.to_string()
)

print(
    '- - - - - - - - Diff - - - - - - - -\n' +
    (after - before).to_string()
)

# seaborn.regplot(
#     x='lesion_1',
#     y='lesion_3',
#     data=data
# )
# pyplot.show()
