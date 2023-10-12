#%%
'''
Test input for KMedoids.
According to the documentation: 'metric': What distance metric to use. See :func:metrics.pairwise_distances metric can be ‘precomputed’, the user must then feed the fit method with a precomputed kernel matrix and not the design matrix X.
It is not explained what a kernel matrix is.

Here, kmedoids is run by using a feature matrix as input, and then a second time by using a distance matrix calculated from the feature matrix as input.
The result should be the same.
'''

import matplotlib.pyplot as plt
import numpy as np

from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs


from sklearn.metrics.pairwise import (
    pairwise_distances,
    pairwise_distances_argmin,
)



print(__doc__)

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

# #############################################################################
# Compute Kmedoids clustering
cobj = KMedoids(n_clusters=3).fit(X)
labels = cobj.labels_

unique_labels = set(labels)
colors = [
    plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
]
for k, col in zip(unique_labels, colors):
    class_member_mask = labels == k

    xy = X[class_member_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.plot(
    cobj.cluster_centers_[:, 0],
    cobj.cluster_centers_[:, 1],
    "o",
    markerfacecolor="cyan",
    markeredgecolor="k",
    markersize=6,
)

plt.title("KMedoids clustering. Medoids are represented in cyan.")
# %%

centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

D = pairwise_distances(X, metric='euclidean')
cobj = KMedoids(n_clusters=3, metric='precomputed').fit(D)
labels = cobj.labels_

unique_labels = set(labels)
colors = [
    plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
]
for k, col in zip(unique_labels, colors):
    class_member_mask = labels == k

    xy = X[class_member_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

# plt.plot(
#     cobj.cluster_centers_[:, 0],
#     cobj.cluster_centers_[:, 1],
#     "o",
#     markerfacecolor="cyan",
#     markeredgecolor="k",
#     markersize=6,
# )

plt.title("KMedoids clustering. Medoids are represented in cyan.")

# %%
