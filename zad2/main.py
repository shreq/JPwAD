import numpy
import pandas
from matplotlib import pyplot
from sklearn.linear_model import LinearRegression

from utils import get_stats, na_fraction, read_choice, impute


def plot_setup():
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
    # pyplot.show()


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
after = impute(before, choice, column_name)


regression_model = LinearRegression()
regression_model.fit(
    after.index.values.reshape(-1, 1),
    after.loc[:, column_name].values.reshape(-1, 1)
)
regression_values = regression_model.predict(after.index.values.reshape(-1, 1))


print_results()

pyplot.plot(
    regression_values,
    color='r'
)
pyplot.scatter(
    x=after.index,
    y=after.loc[:, column_name],
)
plot_setup()
