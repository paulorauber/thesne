import theano.tensor as T
import theano
import numpy as np

epsilon = 1e-16
floath = np.float32


def sqeuclidean_var(X):
    N = X.shape[0]
    ss = (X ** 2).sum(axis=1)
    return ss.reshape((N, 1)) + ss.reshape((1, N)) - 2*X.dot(X.T)


def p_Xp_given_X_var(X, sigma, metric):
    N = X.shape[0]

    if metric == 'euclidean':
        sqdistance = sqeuclidean_var(X)
    elif metric == 'precomputed':
        sqdistance = X**2
    else:
        raise Exception('Invalid metric')

    esqdistance = T.exp(-sqdistance / ((2 * (sigma**2)).reshape((N, 1))))
    esqdistance_zd = T.fill_diagonal(esqdistance, 0)

    row_sum = T.sum(esqdistance_zd, axis=1).reshape((N, 1))

    return esqdistance_zd/row_sum  # Possibly dangerous


def p_Xp_X_var(p_Xp_given_X):
    return (p_Xp_given_X + p_Xp_given_X.T) / (2 * p_Xp_given_X.shape[0])


def p_Yp_Y_var(Y):
    sqdistance = sqeuclidean_var(Y)
    one_over = T.fill_diagonal(1/(sqdistance + 1), 0)
    return one_over/one_over.sum()  # Possibly dangerous

    
def cost_var(X, Y, sigma, metric):
    p_Xp_given_X = p_Xp_given_X_var(X, sigma, metric)
    PX = p_Xp_X_var(p_Xp_given_X)
    PY = p_Yp_Y_var(Y)
    
    PXc = T.maximum(PX, epsilon)
    PYc = T.maximum(PY, epsilon)
    return T.sum(PX * T.log(PXc / PYc))  # Possibly dangerous (clipped)


def find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters,
               metric, verbose=0):
    """Binary search on sigma for a given perplexity."""
    X = T.fmatrix('X')
    sigma = T.fvector('sigma')

    target = np.log(perplexity)

    P = T.maximum(p_Xp_given_X_var(X, sigma, metric), epsilon)

    entropy = -T.sum(P*T.log(P), axis=1)

    # Setting update for binary search interval
    sigmin_shared = theano.shared(np.full(N, np.sqrt(epsilon), dtype=floath))
    sigmax_shared = theano.shared(np.full(N, np.inf, dtype=floath))

    sigmin = T.fvector('sigmin')
    sigmax = T.fvector('sigmax')

    upmin = T.switch(T.lt(entropy, target), sigma, sigmin)
    upmax = T.switch(T.gt(entropy, target), sigma, sigmax)

    givens = {X: X_shared, sigma: sigma_shared, sigmin: sigmin_shared,
              sigmax: sigmax_shared}
    updates = [(sigmin_shared, upmin), (sigmax_shared, upmax)]

    update_intervals = theano.function([], entropy, givens=givens,
                                       updates=updates)

    # Setting update for sigma according to search interval
    upsigma = T.switch(T.isinf(sigmax), sigma*2, (sigmin + sigmax)/2.)

    givens = {sigma: sigma_shared, sigmin: sigmin_shared,
              sigmax: sigmax_shared}
    updates = [(sigma_shared, upsigma)]

    update_sigma = theano.function([], sigma, givens=givens, updates=updates)

    for i in range(sigma_iters):
        e = update_intervals()
        update_sigma()
        if verbose:
            print('Iteration: {0}.'.format(i+1))
            print('Perplexities in [{0:.4f}, {1:.4f}].'.format(np.exp(e.min()),
                  np.exp(e.max())))

    if np.any(np.isnan(np.exp(e))):
        raise Exception('Invalid sigmas. The perplexity is probably too low.')
