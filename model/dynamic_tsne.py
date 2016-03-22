import numpy as np
import theano
import theano.tensor as T

from sklearn.utils import check_random_state

from core import floath
from core import cost_var
from core import find_sigma


def movement_penalty(Ys, N):
    penalties = []
    for t in range(len(Ys) - 1):
        penalties.append(T.sum((Ys[t] - Ys[t + 1])**2))

    return T.sum(penalties)/(2*N)


def find_Ys(Xs_shared, Ys_shared, sigmas_shared, N, steps, output_dims,
            n_epochs, initial_lr, final_lr, lr_switch, init_stdev,
            initial_momentum, final_momentum, momentum_switch, lmbda, metric,
            verbose=0):
    """Optimize cost wrt Ys[t], simultaneously for all t"""
    # Optimization hyperparameters
    initial_lr = np.array(initial_lr, dtype=floath)
    final_lr = np.array(final_lr, dtype=floath)
    initial_momentum = np.array(initial_momentum, dtype=floath)
    final_momentum = np.array(final_momentum, dtype=floath)

    lr = T.fscalar('lr')
    lr_shared = theano.shared(initial_lr)

    momentum = T.fscalar('momentum')
    momentum_shared = theano.shared(initial_momentum)

    # Penalty hyperparameter
    lmbda_var = T.fscalar('lmbda')
    lmbda_shared = theano.shared(np.array(lmbda, dtype=floath))

    # Yv velocities
    Yvs_shared = []
    zero_velocities = np.zeros((N, output_dims), dtype=floath)
    for t in range(steps):
        Yvs_shared.append(theano.shared(np.array(zero_velocities)))

    # Cost
    Xvars = T.fmatrices(steps)
    Yvars = T.fmatrices(steps)
    Yv_vars = T.fmatrices(steps)
    sigmas_vars = T.fvectors(steps)

    c_vars = []
    for t in range(steps):
        c_vars.append(cost_var(Xvars[t], Yvars[t], sigmas_vars[t], metric))

    cost = T.sum(c_vars) + lmbda_var*movement_penalty(Yvars, N)

    # Setting update for Ys velocities
    grad_Y = T.grad(cost, Yvars)

    givens = {lr: lr_shared, momentum: momentum_shared,
              lmbda_var: lmbda_shared}
    updates = []
    for t in range(steps):
        updates.append((Yvs_shared[t], momentum*Yv_vars[t] - lr*grad_Y[t]))

        givens[Xvars[t]] = Xs_shared[t]
        givens[Yvars[t]] = Ys_shared[t]
        givens[Yv_vars[t]] = Yvs_shared[t]
        givens[sigmas_vars[t]] = sigmas_shared[t]

    update_Yvs = theano.function([], cost, givens=givens, updates=updates)

    # Setting update for Ys positions
    updates = []
    givens = dict()
    for t in range(steps):
        updates.append((Ys_shared[t], Yvars[t] + Yv_vars[t]))
        givens[Yvars[t]] = Ys_shared[t]
        givens[Yv_vars[t]] = Yvs_shared[t]

    update_Ys = theano.function([], [], givens=givens, updates=updates)

    # Momentum-based gradient descent
    for epoch in range(n_epochs):
        if epoch == lr_switch:
            lr_shared.set_value(final_lr)
        if epoch == momentum_switch:
            momentum_shared.set_value(final_momentum)

        c = update_Yvs()
        update_Ys()
        if verbose:
            print('Epoch: {0}. Cost: {1:.6f}.'.format(epoch + 1, float(c)))

    Ys = []
    for t in range(steps):
        Ys.append(np.array(Ys_shared[t].get_value(), dtype=floath))

    return Ys


def dynamic_tsne(Xs, perplexity=30, Ys=None, output_dims=2, n_epochs=1000,
                 initial_lr=2400, final_lr=200, lr_switch=250, init_stdev=1e-4,
                 sigma_iters=50, initial_momentum=0.5, final_momentum=0.8,
                 momentum_switch=250, lmbda=0.0, metric='euclidean',
                 random_state=None, verbose=1):
    """Compute sequence of projections from a sequence of matrices of
    observations (or distances) using dynamic t-SNE.
    
    Parameters
    ----------
    Xs : list of array-likes, each with shape (n_observations, n_features), \
            or (n_observations, n_observations) if `metric` == 'precomputed'.
        List of matrices containing the observations (one per row). If `metric` 
        is 'precomputed', list of pairwise dissimilarity (distance) matrices. 
        Each row in `Xs[t + 1]` should correspond to the same row in `Xs[t]`, 
        for every time step t > 1.
    
    perplexity : float, optional (default = 30)
        Target perplexity for binary search for sigmas.
        
    Ys : list of array-likes, each with shape (n_observations, output_dims), \
            optional (default = None)
        List of matrices containing the starting positions for each point at
        each time step.
    
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
        
    lmbda : float, optional (default = 0.0)
        Movement penalty hyperparameter. Controls how much each point is
        penalized for moving across time steps.
        
    metric : 'euclidean' or 'precomputed', optional (default = 'euclidean')
        Indicates whether `X[t]` is composed of observations ('euclidean') 
        or distances ('precomputed'), for all t.
    
    random_state : int or np.RandomState, optional (default = None)
        Integer seed or np.RandomState object used to initialize the
        position of each point. Defaults to a random seed.

    verbose : bool (default = 1)
        Indicates whether progress information should be sent to standard 
        output.
        
    Returns
    -------
    Ys : list of array-likes, each with shape (n_observations, output_dims)
        List of matrices representing the sequence of projections. 
        Each row (point) in `Ys[t]` corresponds to a row (observation or 
        distance to other observations) in the input matrix `Xs[t]`, for all t.
    """
    random_state = check_random_state(random_state)

    steps = len(Xs)
    N = Xs[0].shape[0]

    if Ys is None:
        Y = random_state.normal(0, init_stdev, size=(N, output_dims))
        Ys = [Y]*steps

    for t in range(steps):
        if Xs[t].shape[0] != N or Ys[t].shape[0] != N:
            raise Exception('Invalid datasets.')

        Xs[t] = np.array(Xs[t], dtype=floath)

    Xs_shared, Ys_shared, sigmas_shared = [], [], []
    for t in range(steps):
        X_shared = theano.shared(Xs[t])
        sigma_shared = theano.shared(np.ones(N, dtype=floath))

        find_sigma(X_shared, sigma_shared, N, perplexity, sigma_iters,
                   metric=metric, verbose=verbose)

        Xs_shared.append(X_shared)
        Ys_shared.append(theano.shared(np.array(Ys[t], dtype=floath)))
        sigmas_shared.append(sigma_shared)

    Ys = find_Ys(Xs_shared, Ys_shared, sigmas_shared, N, steps, output_dims,
                 n_epochs, initial_lr, final_lr, lr_switch, init_stdev,
                 initial_momentum, final_momentum, momentum_switch, lmbda,
                 metric, verbose)

    return Ys
