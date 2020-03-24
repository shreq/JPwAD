import pandas
import scipy
import seaborn
from matplotlib import pyplot

numerical = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
nonnumerical = ['object']
datasets = {
    "abalone": ("./datasets/3/abalone.data",
                ["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight",
                 "shell weight", "rings"]),
    "iris": ("./datasets/3/iris.data",
             ["sepal length", "sepal width", "petal length", "petal width", "species"])
}
choice = "iris"

data = pandas.read_csv(
    datasets[choice][0],
    names=datasets[choice][1]
)

print(
    "-> Median:\n" + data.select_dtypes(numerical).median().to_string() + '\n\n' +
    "-> Minimum:\n" + data.select_dtypes(numerical).min().to_string() + '\n\n' +
    "-> Maximum:\n" + data.select_dtypes(numerical).max().to_string() + '\n\n' +
    "-> Mode:\n" + data.select_dtypes(nonnumerical).mode().to_string(index=False) + '\n\n'
)

correlation = data.corr().stack()
correlation = correlation[
    correlation.index.get_level_values(0) != correlation.index.get_level_values(1)
    ].sort_values(ascending=False)

correlated_names = correlation.index.get_level_values(0)[0], correlation.index.get_level_values(0)[1]
correlated_data = data[[correlated_names[0],
                        correlated_names[1]]].to_numpy()

seaborn.distplot(correlated_data[:, 0], label=correlated_names[0], bins=15, kde=False)
seaborn.distplot(correlated_data[:, 1], label=correlated_names[1], bins=15, kde=False)
pyplot.xlabel("value")
pyplot.ylabel("frequency")
pyplot.legend()
pyplot.savefig("histogram3")
pyplot.show()
