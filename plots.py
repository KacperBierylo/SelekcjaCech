from matplotlib import pyplot as plt


def plot_scores(selector, dataset, fno, scores):
    path = "plots/" + str(selector) + " " + str(dataset) + ".png"
    plt.clf()
    plt.title(str(selector) + ": " + str(dataset))
    plt.bar(fno, scores, align="center")
    if selector == "Forward search":
        plt.xlabel("Depth")
    else:
        plt.xlabel("Number of features")
    plt.ylabel("Score %")
    plt.savefig(path, bbox_inches="tight")
