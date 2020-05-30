from sklearn import naive_bayes as nb
from sklearn import metrics
from numpy import unique


class Bayes:

    def __init__(self, data, labels, training_percent, name):
        self.model = nb.GaussianNB()
        self.name = name
        self.unique_labels = unique(labels.values.ravel())
        partition_index = int(len(data.index) * training_percent / 100)
        self.test_data = data.iloc[partition_index:].values
        self.test_labels = labels.iloc[partition_index:].values.ravel()
        self.training_data = data.iloc[:partition_index].values
        self.training_labels = labels.iloc[:partition_index].values.ravel()

    def test(self):
        self.prediction = self.model.predict(self.test_data)

    def train(self):
        self.model.fit(self.training_data, self.training_labels)

    def get_accuracy(self, digits=3):
        return round(metrics.accuracy_score(
            self.test_labels,
            self.prediction), digits)
