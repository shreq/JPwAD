import pandas
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
choice = "abalone"

data = pandas.read_csv(
    datasets[choice][0],
    names=datasets[choice][1]
)

print(
    "-> Data sample:\n" + data.head().to_string() + '\n\n' +
    "-> Median:\n" + data.select_dtypes(numerical).median().to_string() + '\n\n' +
    "-> Minimum:\n" + data.select_dtypes(numerical).min().to_string() + '\n\n' +
    "-> Maximum:\n" + data.select_dtypes(numerical).max().to_string() + '\n\n' +
    "-> Dominant:\n" + data.select_dtypes(nonnumerical).mode().to_string(index=False) + '\n\n'
)

# seaborn.distplot(data.select_dtypes(numerical))
# seaborn.lmplot("length", "height", data.select_dtypes('float64'))
# pyplot.hist(data[datasets[choice][1][1]])
pyplot.show()
