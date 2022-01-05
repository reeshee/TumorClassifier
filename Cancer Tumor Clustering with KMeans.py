import numpy
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

data = pd.read_csv("data.csv")
data.pop('Unnamed: 0')
print(data.head())

X = []
for column in data.columns:
    li = list(data[column])
    X.append(li)
X = scale(np.asarray(X))

labels = pd.read_csv("labels.csv")
labels.pop('Unnamed: 0')

y = np.asarray(labels)
y = y.flatten()

k = len(np.unique(y)) ##dynamically get the number of centroids
## samples, features = data.shape

###scoring the model
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

clf = KMeans(n_clusters=k, init='random', n_init=10)
for _ in range(5):
    bench_k_means(clf, "1", data)

