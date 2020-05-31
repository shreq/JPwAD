from sklearn import metrics
from sklearn import naive_bayes as nb


class Bayes:
    model = None
    training_data = None
    training_labels = None
    test_data = None
    test_labels = None
    prediction = None

    def __init__(self, data, labels, training_percent):
        self.model = nb.GaussianNB()
        partition_index = int(len(data.index) * training_percent / 100)
        self.training_data = data.iloc[:partition_index].values
        self.training_labels = labels.iloc[:partition_index].values.ravel()
        self.test_data = data.iloc[partition_index:].values
        self.test_labels = labels.iloc[partition_index:].values.ravel()

    def test(self):
        self.prediction = self.model.predict(self.test_data)

    def train(self):
        self.model.fit(self.training_data, self.training_labels)

    def get_accuracy(self):
        return round(
            metrics.accuracy_score(self.test_labels, self.prediction),
            ndigits=5
        )
