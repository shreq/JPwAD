from os import system, name

import pandas
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def clear():
    if name == 'nt':
        _ = system('cls')
    else:
        _ = system('clear')


def normalize(dataset):
    values = dataset.values
    values = StandardScaler().fit_transform(values)
    return values


def PCA(dataset):
    pca = PCA(n_components=2)
    data = pca.fit_transform(dataset)
    print('Principal component analysis: {}'.format(pca.explained_variance_ratio_))
    return pandas.DataFrame(data)
