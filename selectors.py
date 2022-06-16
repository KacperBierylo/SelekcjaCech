import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import math
import random


def prepare_data(X, y):
    train_X = X.tolist()
    train_y = y.tolist()
    test_X = []
    test_y = []
    for i in range(int(len(train_X) / 3)):
        test_X.append(train_X.pop(i))
        test_y.append(train_y.pop(i))
    return train_X, train_y, test_X, test_y


def get_score(X, y):
    train_X, train_y, test_X, test_y = prepare_data(X, y)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_X, train_y)
    predicted = model.predict(test_X)
    score = 0
    for i in range(len(predicted)):
        if predicted[i] == test_y[i]:
            score += 1
    return (score * 100) / len(predicted)


def variance_threshold(X, threshold=0):
    new_X = np.copy(X)
    variances = np.var(new_X, axis=0)
    drop = []
    for i in range(len(variances)):
        if variances[i] <= threshold:
            drop.append(i)

    for i in range(len(drop)):
        new_X = np.delete(new_X, drop[i] - i, 1)

    return new_X


def forward_search(X, y, depth=-1):
    if depth == -1 or depth > len(X[0]):
        depth = np.size(X, axis=1)
    length = np.size(X, axis=0)
    best = []
    best_score = 0
    curr_best = []
    for i in range(depth):
        curr = curr_best[:]
        curr_best_score = 0

        numbers = range(depth)
        for n in curr:
            numbers.remove(n)
        for num in numbers:
            curr.append(num)
            curr.sort()
            XX = X[:, curr]
            score = get_score(XX, y)
            if score > curr_best_score:
                curr_best_score = score
                curr_best = curr[:]

            if score > best_score:
                best_score = score
                best = curr[:]
            curr.remove(num)
    new_X = X[:, best]
    return new_X


def correlation(feature1, feature2):
    f1avg = sum(feature1) / len(feature1)
    f2avg = sum(feature2) / len(feature2)
    numerator_values = []
    for i in range(len(feature1)):
        numerator_values.append((feature1[i] - f1avg) * (feature2[i] - f2avg))
    numerator = sum(numerator_values)

    m1temp = []
    for i in range(len(feature1)):
        m1temp.append((feature1[i] - f1avg) ** 2)
    m1 = math.sqrt(sum(m1temp))

    m2temp = []
    for i in range(len(feature2)):
        m2temp.append((feature2[i] - f2avg) ** 2)
    m2 = math.sqrt(sum(m2temp))

    return numerator / (m1 * m2)


def pearsons_correlation(X, features_number=-1):
    new_X = np.copy(X)
    if features_number == -1:
        features_number = np.size(new_X, axis=1) / 2
    to_delete = len(new_X[0]) - features_number
    correlations = np.zeros((len(new_X[0]), len(new_X[0]))).tolist()
    correlations_total = [0] * len(new_X[0])
    for i in range(len(new_X[0])):
        for j in range(i, len(new_X[0])):
            cij = correlation([row[i] for row in new_X], [row[j] for row in new_X])
            correlations_total[i] += abs(cij)
            correlations_total[j] += abs(cij)
            correlations[i][j] = abs(cij)
            correlations[j][i] = abs(cij)

    for i in range(to_delete):
        ind1 = 0
        ind2 = 0
        max_value = -1
        for j in range(len(correlations)):
            for k in range(j + 1, len(correlations[0])):
                if correlations[j][k] > max_value:
                    max_value = correlations[j][k]
                    ind1 = j
                    ind2 = k

        if correlations_total[ind1] > correlations_total[ind2]:
            new_X = np.delete(new_X, ind1, 1)
            correlations_total.pop(ind1)
            correlations.pop(ind1)
            for j in range(len(correlations)):
                correlations[j].pop(ind1)
        else:
            new_X = np.delete(new_X, ind2, 1)
            correlations_total.pop(ind2)
            correlations.pop(ind2)
            for j in range(len(correlations)):
                correlations[j].pop(ind2)

    return new_X
