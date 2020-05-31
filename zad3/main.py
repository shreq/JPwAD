import pandas
from matplotlib import pyplot

from bayes import Bayes
from utils import generate_datasets

data = pandas.read_csv('./datasets/leaf.csv')
labels = data["species"]
data.drop(data.columns[-1], axis=1, inplace=True)

print(data.index)

for dataset in generate_datasets(data, labels):
    print('\n' + dataset.name)
    for training_percent in range(60, 91, 5):
        classifier = Bayes(dataset.data, labels, training_percent)
        classifier.train()
        classifier.test()
        dataset.result.append(classifier.get_accuracy())
        print('Training percent: ' + str(training_percent) + '%, accuracy: ' + str(classifier.get_accuracy()))
    pyplot.plot(range(60, 91, 5), dataset.result, label=dataset.name)

pyplot.xlabel('Training percent')
pyplot.ylabel('Accuracy')
pyplot.legend()
pyplot.savefig(
    'plot',
    dpi=200,
    bbox_inches='tight'
)
