import pandas
from matplotlib import pyplot

from bayes import Bayes
from utils import PCA, normalize
from sklearn.datasets import load_breast_cancer

data = pandas.read_csv('./datasets/leaf.csv')
labels = data["species"]
data.drop(data.columns[-1], axis=1, inplace=True)


for training_percent in range(60, 91, 5):
    classifier = Bayes(data, labels, training_percent)
    classifier.train()
    classifier.test()
    print(
        classifier.get_accuracy()
    )

# pca_data = PCA(normalize(data))

pyplot.savefig(
    'plot',
    dpi=200,
    bbox_inches='tight'
)
