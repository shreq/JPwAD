import numpy
import pandas


def na_fraction(df):
    return df.isna().sum().sum() / df.count().sum()


def get_stats(df):
    stats = pandas.DataFrame()
    stats['Mean'] = df.mean()
    stats['Std deviation'] = df.std()
    stats['Q1 - 25%'] = df.quantile(0.25)
    stats['Q2 - 50%'] = df.quantile(0.5)
    stats['Q3 - 75%'] = df.quantile(0.75)
    stats['NA%'] = df.isna().sum()
    return stats


def impute_mean(df, inplace=False):
    return df.fillna(df.mean(), inplace=inplace)


pandas.options.display.float_format = '{:.2f}'.format
data = pandas.read_csv('./datasets/horse.csv')

data = data.iloc[:, data.applymap(numpy.isreal).all().values]
data.drop(
    ['nasogastric_reflux_ph',
     'abdomo_protein'],
    axis=1,
    inplace=True
)

previous = get_stats(data)
print(
    '- - - - - Before mean imputation - - - - -\n' +
    'Missing values:\t' + '{:.2%}'.format(na_fraction(data)) + '\n' +
    get_stats(data).T.to_string()
)

impute_mean(data, inplace=True)

print(
    '- - - - - After mean imputation - - - - -\n' +
    'Missing values:\t' + '{:.2%}'.format(na_fraction(data)) + '\n' +
    get_stats(data).T.to_string()
)

print(
    '- - - - - Diff - - - - -\n' +
    (get_stats(data) - previous).T.to_string()
)
