import pandas
import scipy
import seaborn
from matplotlib import pyplot

numerical = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
nonnumerical = ['object']
datasets = {
    "births": ("./datasets/4/Births.csv", 'births'),
    "manaus": ("./datasets/4/manaus.csv", 'manaus'),
    "quakes": ("./datasets/4/quakes.csv", 'depth')
}
choice = "quakes"

data = pandas.read_csv(
    datasets[choice][0]
).iloc[:, 1:]
data.index += 1

print(
    "-> Data sample:\n" + data.head().to_string() + '\n\n' +
    "-> Mean " + datasets[choice][1] + ":\t " + str(data[datasets[choice][1]].mean()) + '\n'
)

data_n = (data - data.mean()) / data.std()
print(
    data_n.head().to_string() + '\n\n' +
    data_n.mean().to_string() + '\n\n' +
    data_n.std().to_string()
)
