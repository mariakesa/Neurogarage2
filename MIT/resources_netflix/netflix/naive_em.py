"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
from scipy.special import logsumexp

'''
Naive implementation without logsumexp or vectorization

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    def normal_density(x,k):
        mu=mixture.mu[k]
        var=mixture.var[k]
        d=mu.shape[0]
        p=1./np.sqrt((2*np.pi*var)**d)*np.exp(-0.5*var**-1*(x-mu)@(x-mu).T)
        return p
    def responsibilities(X):
        responsibilities=[]
        for n in range(X.shape[0]):
            X_ps=np.array([mixture.p[i]*normal_density(X[n],i) for i in range(mixture.mu.shape[0])])
            resps=X_ps/np.sum(X_ps)
            responsibilities.append(resps)
        return np.array(responsibilities)
    def log_likelihood(X):
        probs=[]
        for n in range(X.shape[0]):
            p_under_k=[mixture.p[i]*normal_density(X[n],i) for i in range(mixture.mu.shape[0])]
            probs.append(p_under_k)
        probs=np.array(probs)
        return np.sum(np.log(np.sum(probs,axis=1)))
    resp=responsibilities(X)
    return resp, log_likelihood(X)
'''

import numpy as np
from scipy.special import logsumexp
from typing import Tuple

def estep(X: np.ndarray, mixture) -> Tuple[np.ndarray, float]:
    """E-step with logsumexp for numerical stability."""
    n, d = X.shape
    K = mixture.mu.shape[0]

    # Precompute log of priors
    log_pi = np.log(mixture.p)
    log_resp = np.zeros((n, K))  # to store log p(x_n, z_n=k)

    for k in range(K):
        mu_k = mixture.mu[k]
        var_k = mixture.var[k]

        # Log probability of data under kth Gaussian
        diff = X - mu_k  # shape (n, d)
        log_N = -0.5 * (d * np.log(2 * np.pi * var_k) + np.sum(diff**2, axis=1) / var_k)  # shape (n,)
        log_resp[:, k] = log_pi[k] + log_N

    # Normalize responsibilities using logsumexp
    logsumexp_vals = logsumexp(log_resp, axis=1, keepdims=True)  # shape (n, 1)
    log_gamma = log_resp - logsumexp_vals
    gamma = np.exp(log_gamma)  # shape (n, K)

    # Total log-likelihood
    total_log_likelihood = np.sum(logsumexp_vals)

    return gamma, total_log_likelihood

def estep(X: np.ndarray, mixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    def log_multiv_normal(k):
        d=X.shape[1]
        mu=mixture.mu[k]
        var=mixture.var[k]
        squared_norms = np.sum((X - mu)**2, axis=1)
        likelihood=-0.5*(d*np.log(2*np.pi)+np.log(var))-0.5*(var**-1)*squared_norms
        return likelihood
    
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



def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    s=X.T@post
    normalizer=np.sum(post,axis=0)
    mu=(s/normalizer).T

    d=X.shape[1]
    K=post.shape[1]
    
    sigma_sq_k = []
    for k in range(K):
        diff_sq = np.sum((X - mu[k])**2, axis=1)        # shape (n,)
        weighted_sq_dist = post[:, k] * diff_sq         # shape (n,)
        S_k = np.sum(weighted_sq_dist)                  # scalar
        weighted_count = np.sum(post[:, k])             # scalar
        sigma_sq_k.append(S_k / (d * weighted_count))   # scalar

    sigma_sq_k=np.array(sigma_sq_k)

    N=X.shape[0]

    p=np.sum(post, axis=0)/N

    return GaussianMixture(mu, sigma_sq_k, p)




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
    raise NotImplementedError
