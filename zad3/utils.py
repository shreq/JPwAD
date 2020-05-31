from os import system, name

import pandas
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.preprocessing import StandardScaler
from collections import namedtuple

def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')

Dataset = namedtuple('Dataset', ['name', 'data', 'result'])

def generateDatasets(data, labels):
    pca_data = pca(normalize(data))
    chi_data = chi_square(data, labels)
    gvar_data = variance(data, False)
    svar_data = variance(data, True)

    datasets = [data, pca_data, chi_data, gvar_data, svar_data, ]
    names = ['Every column', 'PCA', 'Chi Square', 'Greatest variance', 'Smallest variance', ]
    results = [[] for i in range(len(datasets))]

    return [Dataset(name, dataset, result) for name, dataset, result in zip(names, datasets, results)]

def normalize(dataset):
    values = dataset.values
    values = StandardScaler().fit_transform(values)
    return values

def pca(dataset):
    pca = PCA(n_components=2)
    data = pca.fit_transform(dataset)
    print('Principal component analysis: {}'.format(pca.explained_variance_ratio_))
    return pandas.DataFrame(data)

def variance(dataset, smallest):
    ds = dataset.var()
    ds = ds.sort_values(ascending=smallest)

    prefix = 'Smallest' if smallest is True else 'Greatest'
    
    print(prefix + ' variance: "{}" ({}), "{}" ({})'.format(ds.index[0], ds[0], ds.index[1], ds[1]))
    return pandas.DataFrame(dataset[[ds.index[0], ds.index[1]]])


def chi_square(dataset, labels):
    selector = SelectKBest(score_func=chi2, k=2)
    selector.fit(dataset, labels)
    values = selector.scores_[selector.get_support()]
    columns = dataset.columns[selector.get_support()]
    print('Chi square: "{}" ({}), "{}" ({})'
          .format(columns[0], values[0], columns[1], values[1]))

    return pandas.DataFrame(dataset[columns])
