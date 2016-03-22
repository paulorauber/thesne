import numpy as np

from sklearn.utils import check_random_state

from thesne.model.dynamic_tsne import dynamic_tsne
from thesne.examples import plot


def create_blobs(classes=10, dims=100, class_size=100, variance=0.1, steps=4,
                 advection_ratio=0.5, random_state=None):
    random_state = check_random_state(random_state)
    X = []

    indices = random_state.permutation(dims)[0:classes]
    means = []
    for c in range(classes):
        mean = np.zeros(dims)
        mean[indices[c]] = 1.0
        means.append(mean)

        X.append(random_state.multivariate_normal(mean, np.eye(dims)*variance,
                                                  class_size))
    X = np.concatenate(X)
    y = np.concatenate([[i]*class_size for i in range(classes)])

    Xs = [np.array(X)]
    for step in range(steps - 1):
        Xnext = np.array(Xs[step])
        for c in range(classes):
            stard, end = class_size*c, class_size*(c + 1)
            Xnext[stard: end] += advection_ratio*(means[c] - Xnext[stard: end])

        Xs.append(Xnext)

    return Xs, y


def main():
    seed = 0

    Xs, y = create_blobs(class_size=200, advection_ratio=0.1, steps=10,
                         random_state=seed)

    Ys = dynamic_tsne(Xs, perplexity=70, lmbda=0.1, verbose=1, sigma_iters=50,
                      random_state=seed)

    for Y in Ys:
        plot.plot(Y, y)


if __name__ == "__main__":
    main()
