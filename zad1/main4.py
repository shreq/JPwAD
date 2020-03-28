import pandas
import scipy.stats
from matplotlib import pyplot

datasets = {
    "births": ("./datasets/4/Births.csv", 'births', 10000),
    "manaus": ("./datasets/4/manaus.csv", 'manaus', 0),
    "quakes": ("./datasets/4/quakes.csv", 'depth', 300)
}
choice = "quakes"

data = pandas.read_csv(
    datasets[choice][0]
).iloc[:, 1:]
data.index += 1
sample = data[datasets[choice][1]]

normal, p_value_n = scipy.stats.normaltest(sample)

print(
    "-> Normality test\n" +
    "T-Statistic:\t" + str(normal) + '\n' +
    "P-Value:    \t" + str(p_value_n) + '\n'
)

if p_value_n < 0.05:
    print("Sample doesn't come from normal distribution\n" +
          "Quitting")
else:
    print("Sample comes from normal distribution")

    hypothesis, p_value_t = scipy.stats.ttest_1samp(
        sample,
        datasets[choice][2]
    )

    print(
        "\n-> Hypothesis test\n" +
        "T-Statistic:\t " + str(hypothesis) + '\n' +
        "P-Value    :\t " + str(p_value_t) + '\n' +
        "Hypothesis " + ("rejected" if p_value_t < 0.05 else "confirmed")
    )

pyplot.hist(
    sample,
    label=datasets[choice][1],
    bins=sample.nunique(),
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
