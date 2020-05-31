import pandas
from matplotlib import pyplot
import numpy

from bayes import Bayes
from utils import pca, normalize, generateDatasets

data = pandas.read_csv('./datasets/leaf.csv')
labels = data["species"]
data.drop(data.columns[-1], axis=1, inplace=True)

for dataset in generateDatasets(data, labels):
    print(dataset)
    for training_percent in range(60, 91, 5): 
        classifier = Bayes(data, labels, training_percent, dataset.name)
        classifier.train()
        classifier.test()
        dataset.result.append(classifier.get_accuracy())
        #print('Training percent: ' + str(training_percent) + '%, accuracy: ' + str(classifier.get_accuracy()))
    print(dataset.result)
    pyplot.plot(range(60, 91, 5), dataset.result, label=classifier.name)

pyplot.xlabel('Training percent')
pyplot.ylabel('Accuracy')
pyplot.legend()
pyplot.savefig(
    'plot',
    dpi=200,
    bbox_inches='tight'
)
