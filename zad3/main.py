import pandas
import matplotlib.pyplot as plt
from utils import PCA, Normalize
from bayes import Bayes

data = pandas.read_csv('./datasets/leaf.csv')
labels = data["species"]

pca_data = PCA(Normalize(data))
