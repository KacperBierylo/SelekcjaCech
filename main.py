from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import selectors
import plots
import pandas as pd


def compare_all():
    iris = load_iris()
    wine = load_wine()
    breast_cancer = load_breast_cancer()
    hill_valley_data = pd.read_csv('hill_valley.csv', header=None, usecols=range(0, 100)).values
    hill_valley_target = pd.read_csv('hill_valley.csv', header=None, usecols=[100]).values.ravel()
    data = ["Iris", "Wine", "Breast cancer", "Hill Valley"]
    X = [iris.data, wine.data, breast_cancer.data, hill_valley_data]
    y = [iris.target, wine.target, breast_cancer.target, hill_valley_target]

    for i in range(len(data)):
        print("\n  " + data[i] + ", " + str(len(X[i][0])) + " features:")
        fno = len(X[i][0]) / 2
        # Ours
        forward_search = selectors.forward_search(X[i], y[i])
        pearsons_correlation = selectors.pearsons_correlation(X[i])
        variance_threshold = selectors.variance_threshold(X[i])
        # From the library
        fvalue_best = SelectKBest(f_classif, k=fno)
        anova_fvalues = fvalue_best.fit_transform(X[i], y[i])
        mutual_info_best = SelectKBest(mutual_info_classif, k=fno)
        mutual_info = mutual_info_best.fit_transform(X[i], y[i])

        no_selectors = selectors.get_score(X[i], y[i])
        forward_selector = selectors.get_score(forward_search, y[i])
        pearsons_correlation_selector = selectors.get_score(pearsons_correlation, y[i])
        variance_selector = selectors.get_score(variance_threshold, y[i])
        anova_fvalues_selector = selectors.get_score(anova_fvalues, y[i])
        mutual_info_classif_selector = selectors.get_score(mutual_info, y[i])

        print("  No selectors: " + str(selectors.get_score(X[i], y[i])) + "%")
        print("  Forward search selector: " + str(forward_selector) + "%")
        print("  Variance threshold selector: " + str(variance_selector) + "%")
        print("  Pearson\'s correlation selector: " + str(pearsons_correlation_selector) + "%")
        print("  Best ANOVA f-values, " + str(fno) + " features: " + str(anova_fvalues_selector) + "%")
        print("  Mutual info classifier, " + str(fno) + " features: " + str(mutual_info_classif_selector) + "%")


def compare_pearsons(plot=False):
    wine = load_wine()
    breast_cancer = load_breast_cancer()
    hill_valley_data = pd.read_csv('hill_valley.csv', header=None, usecols=range(0, 100)).values
    hill_valley_target = pd.read_csv('hill_valley.csv', header=None, usecols=[100]).values.ravel()
    data = ["Wine", "Breast cancer", "Hill Valley"]
    X = [wine.data, breast_cancer.data, hill_valley_data]
    y = [wine.target, breast_cancer.target, hill_valley_target]
    print("Pearson\'s correlation and number of features:")

    for i in range(len(data)):
        num_of_features = len(X[i][0])
        print(data[i] + ", " + str(num_of_features) + " features:")
        features_no = range(1, num_of_features + 1)
        scores = []
        for j in features_no:
            pearsons_correlation = selectors.pearsons_correlation(X[i], j)
            pearsons_correlation_selector = selectors.get_score(pearsons_correlation, y[i])
            print(str(j) + " features: " + str(pearsons_correlation_selector) + "%")
            scores.append(pearsons_correlation_selector)
        if plot:
            plots.plot_scores("Pearson\'s correlation", data[i], features_no, scores)


def compare_anova(plot=False):
    wine = load_wine()
    breast_cancer = load_breast_cancer()
    hill_valley_data = pd.read_csv('hill_valley.csv', header=None, usecols=range(0, 100)).values
    hill_valley_target = pd.read_csv('hill_valley.csv', header=None, usecols=[100]).values.ravel()
    data = ["Wine", "Breast cancer", "Hill Valley"]
    X = [wine.data, breast_cancer.data, hill_valley_data]
    y = [wine.target, breast_cancer.target, hill_valley_target]
    print("Best ANOVA f-values and number of features:")

    for i in range(len(data)):
        num_of_features = len(X[i][0])
        print(data[i] + ", " + str(num_of_features) + " features:")
        features_no = range(1, num_of_features + 1)
        scores = []
        for j in features_no:
            fvalue_best = SelectKBest(f_classif, k=j)
            anova_fvalues = fvalue_best.fit_transform(X[i], y[i])
            anova_fvalues_selector = selectors.get_score(anova_fvalues, y[i])
            print(str(j) + " features: " + str(anova_fvalues_selector) + "%")
            scores.append(anova_fvalues_selector)
        if plot:
            plots.plot_scores("Anova f-values", data[i], features_no, scores)


def compare_mutual(plot=False):
    wine = load_wine()
    breast_cancer = load_breast_cancer()
    hill_valley_data = pd.read_csv('hill_valley.csv', header=None, usecols=range(0, 100)).values
    hill_valley_target = pd.read_csv('hill_valley.csv', header=None, usecols=[100]).values.ravel()
    data = ["Wine", "Breast cancer", "Hill Valley"]
    X = [wine.data, breast_cancer.data, hill_valley_data]
    y = [wine.target, breast_cancer.target, hill_valley_target]
    print("Mutual information classifier and number of features:")

    for i in range(len(data)):
        num_of_features = len(X[i][0])
        print(data[i] + ", " + str(num_of_features) + " features:")
        features_no = range(1, num_of_features + 1)
        scores = []
        for j in features_no:
            mutual_info_best = SelectKBest(mutual_info_classif, k=j)
            mutual_info = mutual_info_best.fit_transform(X[i], y[i])
            mutual_info_classif_selector = selectors.get_score(mutual_info, y[i])
            print(str(j) + " features: " + str(mutual_info_classif_selector) + "%")
            scores.append(mutual_info_classif_selector)
        if plot:
            plots.plot_scores("Mutual information classifier", data[i], features_no, scores)


def compare_forward(plot=False):
    wine = load_wine()
    breast_cancer = load_breast_cancer()
    hill_valley_data = pd.read_csv('hill_valley.csv', header=None, usecols=range(0, 100)).values
    hill_valley_target = pd.read_csv('hill_valley.csv', header=None, usecols=[100]).values.ravel()
    data = ["Wine", "Breast cancer", "Hill Valley"]
    X = [wine.data, breast_cancer.data, hill_valley_data]
    y = [wine.target, breast_cancer.target, hill_valley_target]
    print("Forward search and depth:")

    for i in range(len(data)):
        num_of_features = len(X[i][0])
        print(data[i] + ", " + str(num_of_features) + " features:")
        depth = range(1, num_of_features + 1)
        scores = []
        for j in depth:
            forward_search = selectors.forward_search(X[i], y[i], j)
            forward_selector = selectors.get_score(forward_search, y[i])
            print("Depth " + str(j) + ": " + str(forward_selector) + "%")
            scores.append(forward_selector)
        if plot:
            plots.plot_scores("Forward search", data[i], depth, scores)


if __name__ == '__main__':
    compare_all()
