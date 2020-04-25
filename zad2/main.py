import numpy
import pandas
from matplotlib import pyplot

from utils import get_stats, na_fraction, read_choice, impute, get_linear_regression_values


def plot_results():
    pyplot.plot(
        get_linear_regression_values(
            after.index.values.reshape(-1, 1),
            after.loc[:, column_name].values.reshape(-1, 1),
        ),
        color='r'
    )
    pyplot.scatter(
        x=after.index,
        y=after.loc[:, column_name],
    )
    pyplot.xlabel('index')
    pyplot.ylabel('value')
    pyplot.title(column_name)
    pyplot.savefig(
        column_name + '_' + {1: 'mean',
                             2: 'interpolation',
                             3: 'hotdeck',
                             4: 'regression'}[choice],
        dpi=200,
        bbox_inches='tight'
    )


def print_results():
    print('- - - - - - - - Before imputation - - - - - - - -\n' +
          'Missing values:\t' + '{:.2%}'.format(na_fraction(before)) + '\n' +
          get_stats(before).to_string())
    print('- - - - - - - - After imputation - - - - - - - -\n' +
          'Missing values:\t' + '{:.2%}'.format(na_fraction(after)) + '\n' +
          get_stats(after).to_string())
    print('- - - - - - - - Difference - - - - - - - -\n' +
          (get_stats(after) - get_stats(before)).to_string())


pandas.options.display.float_format = '{:.2f}'.format
before = pandas.read_csv('./datasets/horse.csv')
before = before.iloc[:, before.applymap(numpy.isreal).all().values]
before.drop(['nasogastric_reflux_ph',
             'abdomo_protein'],
            axis=1,
            inplace=True)
column_name = 'total_protein'

choice = read_choice()
after = impute(before, choice)

print_results()
plot_results()
