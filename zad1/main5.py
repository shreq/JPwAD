import numpy
import pandas
import scipy.stats
import seaborn
from matplotlib import pyplot


def factorize(column):
    if column.dtype in [numpy.float64, numpy.float32, numpy.int32, numpy.int64]:
        return column
    else:
        return pandas.factorize(column)[0]


numerical = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
datasets = {
    "footballers": ("./datasets/5/footballers.csv", 'Weight(pounds)',
                    'Position', ('Starting_Pitcher', 'Relief_Pitcher')),
    "cats": ("./datasets/5/cats-data.csv", 'Hwt',
             'Sex', ('F', 'M')),
}
choice = "cats"

data = pandas.read_csv(
    datasets[choice][0]
).iloc[:, 1:]
data.index += 1


data_n = (data.select_dtypes(numerical) - data.select_dtypes(numerical).mean()) / \
         data.select_dtypes(numerical).std()
# data_n = (data.select_dtypes(numerical) - data.select_dtypes(numerical).min()) / \
#          (data.select_dtypes(numerical).max() - data.select_dtypes(numerical).min())
data_n[datasets[choice][2]] = data[datasets[choice][2]]
# data_n = data

one = data_n[data_n[datasets[choice][2]] == datasets[choice][3][0]]
two = data_n[data_n[datasets[choice][2]] == datasets[choice][3][1]]

seaborn.distplot(
    one[datasets[choice][1]],
    label=datasets[choice][1] + ' when ' + datasets[choice][2] + '=' + datasets[choice][3][0],
    hist=False
)
seaborn.distplot(
    two[datasets[choice][1]],
    label=datasets[choice][1] + ' when ' + datasets[choice][2] + '=' + datasets[choice][3][1],
    hist=False
)
pyplot.xlabel("value")
pyplot.ylabel("frequency")
pyplot.legend()
pyplot.savefig("histogram5")
# pyplot.show()
