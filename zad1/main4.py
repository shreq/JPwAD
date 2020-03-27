import pandas
import scipy.stats
from matplotlib import pyplot

datasets = {
    "births": ("./datasets/4/Births.csv", 'births', 10000),
    "manaus": ("./datasets/4/manaus.csv", 'manaus', 0),
    "quakes": ("./datasets/4/quakes.csv", 'depth', 300)
}
choice = "manaus"

data = pandas.read_csv(
    datasets[choice][0]
).iloc[:, 1:]
data.index += 1

statistic, p_value = scipy.stats.ttest_1samp(
    data[datasets[choice][1]],
    datasets[choice][2]
)

print(
    "T-Statistic:\t " + str(statistic) + '\n' +
    "P-Value    :\t " + str(p_value) + '\n' +
    "Hypothesis " + ("rejected" if p_value < 0.05 else "confirmed")
)

pyplot.hist(
    data[datasets[choice][1]],
    label=datasets[choice][1],
    bins=25,
    alpha=0.4,
    histtype='bar',
    density=True
)
pyplot.axvline(
    datasets[choice][2],
    label="hypothesis"
)
pyplot.xlabel("value")
pyplot.ylabel("frequency")
pyplot.legend()
pyplot.savefig("histogram4")
pyplot.show()
