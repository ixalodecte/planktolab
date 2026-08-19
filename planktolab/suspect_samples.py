import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def self_confidence(labels, probas):
    labels = np.asarray(labels)
    return probas[np.arange(len(labels)), labels]


def normalized_margin(labels, probas):
    labels = np.asarray(labels)

    p_true = probas[np.arange(len(labels)), labels]

    p_sorted = np.sort(probas, axis=1)
    p_second = p_sorted[:, -2]

    margin = p_true - p_second


    norm_margin = (margin + 1) / 2
    return norm_margin


def score_knn(features, labels, n_neighbors=11):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(features, labels)

    #pred_knn = neigh.predict(features)

    proba = neigh.predict_proba(features)
    idx = np.searchsorted(neigh.classes_, labels)
    proba_knn = proba[np.arange(len(labels)), idx]
    #proba_knn = neigh.predict_proba(features)[np.arange(len(labels)), labels]

    return proba_knn

