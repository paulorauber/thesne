import gzip
import pickle

from sklearn.utils import check_random_state

from thesne.model.tsne import tsne
from thesne.examples import plot


def subsample(X, y, size, random_state=None):
    random_state = check_random_state(random_state)

    shuffle = random_state.permutation(X.shape[0])

    X, y = X[shuffle], y[shuffle]
    X, y = X[0:size], y[0:size]

    return X, y


def main():
    seed = 0

    # Available at http://deeplearning.net/tutorial/gettingstarted.html
    datapath = 'data/mnist.pkl.gz'

    f = gzip.open(datapath, 'rb')
    train_Xy, _, _ = pickle.load(f)
    f.close()

    X, y = subsample(train_Xy[0], train_Xy[1], size=2000, random_state=seed)

    Y = tsne(X, perplexity=30, n_epochs=1000, sigma_iters=50,
             random_state=seed, verbose=1)

    plot.plot(Y, y)


if __name__ == "__main__":
    main()
