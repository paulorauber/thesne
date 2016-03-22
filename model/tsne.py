import numpy as np
import theano
import theano.tensor as T

from sklearn.utils import check_random_state

from core import floath
from core import cost_var
from core import find_sigma


def find_Y(X_shared, Y_shared, sigma_shared, N, output_dims, n_epochs,
           initial_lr, final_lr, lr_switch, init_stdev, initial_momentum,
           final_momentum, momentum_switch, metric, verbose=0):
    """Optimize cost wrt Y"""
    # Optimization hyperparameters
    initial_lr = np.array(initial_lr, dtype=floath)
    final_lr = np.array(final_lr, dtype=floath)
    initial_momentum = np.array(initial_momentum, dtype=floath)
    final_momentum = np.array(final_momentum, dtype=floath)

    lr = T.fscalar('lr')
    lr_shared = theano.shared(initial_lr)

    momentum = T.fscalar('momentum')
    momentum_shared = theano.shared(initial_momentum)

    # Y velocities
    Yv = T.fmatrix('Yv')
    Yv_shared = theano.shared(np.zeros((N, output_dims), dtype=floath))

    # Cost
    X = T.fmatrix('X')
    sigma = T.fvector('sigma')
    Y = T.fmatrix('Y')

    cost = cost_var(X, Y, sigma, metric)

    # Setting update for Y velocities
    grad_Y = T.grad(cost, Y)

    updates = [(Yv_shared, momentum*Yv - lr*grad_Y)]
    givens = {X: X_shared, sigma: sigma_shared, Y: Y_shared, Yv: Yv_shared,
              lr: lr_shared, momentum: momentum_shared}

    update_Yv = theano.function([], cost, givens=givens, updates=updates)

    # Setting update for Y
    givens = {Y: Y_shared, Yv: Yv_shared}
    updates = [(Y_shared, Y + Yv)]

    update_Y = theano.function([], [], givens=givens, updates=updates)

    # Momentum-based gradient descent
    for epoch in range(n_epochs):
        if epoch == lr_switch:
            lr_shared.set_value(final_lr)
        if epoch == momentum_switch:
            momentum_shared.set_value(final_momentum)

        c = update_Yv()
        update_Y()
        if verbose:
            print('Epoch: {0}. Cost: {1:.6f}.'.format(epoch + 1, float(c)))

    return np.array(Y_shared.get_value())


def tsne(X, perplexity=30, Y=None, output_dims=2, n_epochs=1000,
         initial_lr=2400, final_lr=200, lr_switch=250, init_stdev=1e-4,
         sigma_iters=50, initial_momentum=0.5, final_momentum=0.8,
         momentum_switch=250, metric='euclidean', random_state=None,
         verbose=1):
    """Compute projection from a matrix of observations (or distances) using 
    t-SNE.
    
    Parameters
    ----------
    X : array-like, shape (n_observations, n_features), \
            or (n_observations, n_observations) if `metric` == 'precomputed'.
        Matrix containing the observations (one per row). If `metric` is 
        'precomputed', pairwise dissimilarity (distance) matrix.
    
    perplexity : float, optional (default = 30)
        Target perplexity for binary search for sigmas.
        
    Y : array-like, shape (n_observations, output_dims), optional \
            (default = None)
        Matrix containing the starting position for each point.
    
    output_dims : int, optional (default = 2)
        Target dimension.
        
    n_epochs : int, optional (default = 1000)
        Number of gradient descent iterations.
        
    initial_lr : float, optional (default = 2400)
        The initial learning rate for gradient descent.
        
    final_lr : float, optional (default = 200)
        The final learning rate for gradient descent.
        
    lr_switch : int, optional (default = 250)
        Iteration in which the learning rate changes from initial to final.
        This option effectively subsumes early exaggeration.
        
    init_stdev : float, optional (default = 1e-4)
        Standard deviation for a Gaussian distribution with zero mean from
        which the initial coordinates are sampled.
        
    sigma_iters : int, optional (default = 50)
        Number of binary search iterations for target perplexity.
        
    initial_momentum : float, optional (default = 0.5)
        The initial momentum for gradient descent.
        
    final_momentum : float, optional (default = 0.8)
        The final momentum for gradient descent.
        
    momentum_switch : int, optional (default = 250)
        Iteration in which the momentum changes from initial to final.
        
    metric : 'euclidean' or 'precomputed', optional (default = 'euclidean')
        Indicates whether `X` is composed of observations ('euclidean') 
        or distances ('precomputed').
    
    random_state : int or np.RandomState, optional (default = None)
        Integer seed or np.RandomState object used to initialize the
        position of each point. Defaults to a random seed.

    verbose : bool (default = 1)
        Indicates whether progress information should be sent to standard 
        output.
        
    Returns
    -------
    Y : array-like, shape (n_observations, output_dims)
        Matrix representing the projection. Each row (point) corresponds to a
        row (observation or distance to other observations) in the input matrix.
    """
    random_state = check_random_state(random_state)

    N = X.shape[0]

    X_shared = theano.shared(np.asarray(X, dtype=floath))
    sigma_shared = theano.shared(np.ones(N, dtype=floath))

    if Y is None:
        Y = random_state.normal(0, init_stdev, size=(N, output_dims))
    Y_shared = theano.shared(np.asarray(Y, dtype=floath))

    find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters, metric,
               verbose)

    Y = find_Y(X_shared, Y_shared, sigma_shared, N, output_dims, n_epochs,
               initial_lr, final_lr, lr_switch, init_stdev, initial_momentum,
               final_momentum, momentum_switch, metric, verbose)

    return Y
