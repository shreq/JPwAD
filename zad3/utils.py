from collections import namedtuple

import pandas
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import StandardScaler

Dataset = namedtuple('Dataset', ['name', 'data', 'result'])


def generate_datasets(data, labels):
    pca_data = pca(normalize(data))
    chi_data = chi_square(data, labels)
    greatest_variance_data = variance(data, False)
    smallest_variance_data = variance(data, True)

    datasets = [data, pca_data, chi_data, greatest_variance_data, smallest_variance_data, ]
    names = ['Every column', 'PCA', 'Chi Square', 'Greatest variance', 'Smallest variance', ]
    results = [[] for i in range(len(datasets))]

    return [Dataset(name, dataset, result) for name, dataset, result in zip(names, datasets, results)]


def normalize(dataset):
    return StandardScaler().fit_transform(dataset.values)


def pca(dataset):
    pca_ = PCA(n_components=2)
    data = pca_.fit_transform(dataset)
    print(f'Principal component analysis:\n\t{pca_.explained_variance_ratio_}')
    return pandas.DataFrame(data)


def variance(dataset, smallest):
    ds = dataset.var()
    ds = ds.sort_values(ascending=smallest)

    print(('Smallest' if smallest is True else 'Greatest'),
          f'variance:\n\t{ds.index[0]} ({ds[0]})\n\t{ds.index[1]} ({ds[1]})')
    return pandas.DataFrame(dataset[[ds.index[0], ds.index[1]]])


def chi_square(dataset, labels):
    selector = SelectKBest(score_func=chi2, k=2)
    selector.fit(dataset, labels)
    values = selector.scores_[selector.get_support()]
    columns = dataset.columns[selector.get_support()]
    print(f'Chi square:\n\t{columns[0]} ({values[0]})\n\t{columns[1]} ({values[1]})')

    return pandas.DataFrame(dataset[columns])
