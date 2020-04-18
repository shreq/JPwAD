import pandas


def na_fraction(dataframe):
    return dataframe.isna().sum().sum() / dataframe.count().sum()


def get_stats(dataframe):
    stats = pandas.DataFrame()
    stats['Mean'] = dataframe.mean()
    stats['Standard deviation'] = dataframe.std()
    stats['Q1 - 25%'] = dataframe.quantile(0.25)
    stats['Q2 - 50%'] = dataframe.quantile(0.5)
    stats['Q3 - 75%'] = dataframe.quantile(0.75)
    return stats


pandas.options.display.float_format = '{:.2f}'.format
data = pandas.read_csv('./datasets/horse.csv')

data.drop(
    ['nasogastric_reflux_ph',
     'abdomo_protein',
     'abdomo_appearance',
     'abdomen',
     'nasogastric_reflux',
     'nasogastric_tube',
     'rectal_exam_feces'],
    axis=1,
    inplace=True
)

print(
    'Missing values:\t' + '{:.2%}'.format(na_fraction(data)) + '\n' +
    get_stats(data).T.to_string()
)
