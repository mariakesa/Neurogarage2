"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    def log_multiv_normal(k, sample_index):
        non_zero_ents=np.where(X[sample_index]!=0)[0]
        d = len(non_zero_ents)
        mu=mixture.mu[k][non_zero_ents]
        var=mixture.var[k]
        squared_norm = np.sum((X[sample_index][non_zero_ents] - mu)**2)
        likelihood=-0.5 * (d * np.log(2 * np.pi * var) + squared_norm / var)
        return likelihood
    
    K=mixture.mu.shape[0]
    prior=mixture.p
    
    n=X.shape[0]

    log_unnormalized_resps=np.zeros((n, K))

    for sample_index in range(n):
        for cluster in range(K):
            log_unnormalized_resps[sample_index, cluster]=np.log(prior[cluster])+log_multiv_normal(cluster, sample_index)

    normalizers=logsumexp(log_unnormalized_resps,axis=1, keepdims=True)
    responsibilities=np.exp(log_unnormalized_resps-normalizers)

    loglikelihood=np.sum(normalizers)
    return responsibilities, loglikelihood



    '''
    K=mixture.mu.shape[0]
    prior=mixture.p
    
    n=X.shape[0]

    log_unnormalized_resps=np.zeros((n, K))

    for k in range(K):
        log_unnormalized_resps[:,k]=np.log(prior[k])+log_multiv_normal(k)

    normalizers=logsumexp(log_unnormalized_resps,axis=1, keepdims=True)
    responsibilities=np.exp(log_unnormalized_resps-normalizers)

    loglikelihood=np.sum(normalizers)
    return responsibilities, loglikelihood
    '''


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    mu = np.zeros((K, d))
    normalizer = np.zeros((K, d))

    # Step 1: Compute weighted sums and normalizers
    for n_i in range(n):
        u = X[n_i]
        nonzero = np.where(u != 0)[0]
        for k in range(K):
            for dim in nonzero:
                mu[k, dim] += post[n_i, k] * u[dim]
                normalizer[k, dim] += post[n_i, k]

    # Step 2: Normalize only if total weight >= 1
    for k in range(K):
        for dim in range(d):
            if normalizer[k, dim] >= 1:
                mu[k, dim] /= normalizer[k, dim]
            else:
                mu[k, dim] = mixture.mu[k, dim]  # keep previous mean

    # Step 3: Compute isotropic variances
    var = np.zeros(K)

    for k in range(K):
        weighted_squared_diff = 0.0
        total_weighted_observed_dims = 0.0

        for n_i in range(n):
            u = X[n_i]
            nonzero = np.where(u != 0)[0]
            if len(nonzero) == 0:
                continue

            diff = u[nonzero] - mu[k, nonzero]
            squared_diff = np.sum(diff ** 2)
            weighted_squared_diff += post[n_i, k] * squared_diff
            total_weighted_observed_dims += post[n_i, k] * len(nonzero)

        if total_weighted_observed_dims > 0:
            var[k] = max(weighted_squared_diff / total_weighted_observed_dims, min_variance)
        else:
            var[k] = min_variance  # fallback

    # Step 4: Update mixture probabilities
    p = np.sum(post, axis=0) / n

    return GaussianMixture(mu=mu, var=var, p=p)

def fill_matrix(X, mixture):
    """
    Fills an incomplete matrix according to a Gaussian mixture model using log-sum-exp for stability.
    
    Args:
        X: (n, d) array of incomplete data (missing entries are 0).
        mixture: tuple (mu, var, pi) representing the GMM:
                 mu is array of shape (K, d) of component means,
                 var is array of shape (K,) of isotropic variances for each component,
                 pi is array of shape (K,) of mixture weights (summing to 1).
    
    Returns:
        X_filled: (n, d) array with missing entries imputed.
    """
    X_filled = X.copy().astype(float)
    n, d = X.shape
    mu, var, pi = mixture
    K = mu.shape[0]

    log_pi = np.log(pi)
    responsibilities = np.zeros((n, K))

    for i in range(n):
        observed = X[i] != 0
        obs_count = np.sum(observed)

        if obs_count > 0:
            log_probs = np.zeros(K)
            for k in range(K):
                diff = X[i, observed] - mu[k, observed]
                exponent = -0.5 * np.sum((diff ** 2) / var[k])
                log_norm = -0.5 * obs_count * np.log(2 * np.pi * var[k])
                log_probs[k] = log_pi[k] + log_norm + exponent
            # Normalize with logsumexp
            log_denom = logsumexp(log_probs)
            responsibilities[i, :] = np.exp(log_probs - log_denom)
        else:
            responsibilities[i, :] = pi  # fallback to prior

    # Compute expected value for missing entries
    weighted_means = responsibilities @ mu
    X_filled[X == 0] = weighted_means[X == 0]
    return X_filled



def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    ll_old = None
    while True:
        responsibilities, ll_new = estep(X, mixture)
        mixture = mstep(X, responsibilities, mixture)

        if ll_old is not None and (ll_new - ll_old) <= 1e-6 * abs(ll_new):
            break
        ll_old = ll_new

    X_filled=fill_matrix(X, mixture)
    return X_filled, ll_old

