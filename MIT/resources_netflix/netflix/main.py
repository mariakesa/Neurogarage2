import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

#X=np.loadtxt("netflix_incomplete.txt")
#X_gold=np.loadtxt("netflix_complete.txt")


K=[1,2,3,4]

def run_K(X, k):

    seeds=[4]

    costs=[]
    for s in seeds:
        mixture, post = common.init(X, k, s)
        mixture, post, cost = kmeans.run(X, mixture, post)
        costs.append(cost)
        common.plot(X, mixture, post,title=f'Cluster size {k} and seed {s}')
    return np.argmin(costs)
'''
K=[4]
costs=[]
for k in K:
    print(k)
    cost=run_K(X,k)
    costs.append(cost)

print(costs)
'''
import concurrent.futures
import numpy as np

def em_task(k, seed, X, X_gold):
    mixture, post = common.init(X, k, seed)
    X_filled, ll = em.run(X, mixture, post)
    rmse_val = common.rmse(X_gold, X_filled)
    return k, seed, ll, X_filled, rmse_val

import concurrent.futures
from itertools import product

from multiprocessing import Pool

def run_em_(X, X_gold):
    K = [1, 12]
    seeds = [0, 1, 2, 3, 4]
    tasks = [(k, seed, X, X_gold) for k in K for seed in seeds]

    with Pool(processes=8) as pool:  # adjust number of workers as needed
        results = pool.starmap(em_task, tasks)

    for k in K:
        group = [(ll, x, seed, rmse_val) for (kk, seed, ll, x, rmse_val) in results if kk == k]
        best = max(group, key=lambda tup: tup[0])
        best_ll, best_x, best_seed, best_rmse = best

        print(f"Best result for k={k}:")
        print(f"  Seed: {best_seed}")
        print(f"  Log-likelihood: {best_ll}")
        print(f"  RMSE: {best_rmse}")



#run_em(X, X_gold)

from common import bic

from multiprocessing import Pool
from common import bic

def em_task(k, seed, X):
    mixture, post = common.init(X, k, seed)
    _, ll = em.run(X, mixture, post)
    return k, seed, ll, mixture

def run_em_bic(X):
    K = [1, 2, 3, 4]
    seeds = [0]
    n, _ = X.shape

    # Prepare all jobs
    tasks = [(k, seed, X) for k in K for seed in seeds]

    # Run them in parallel
    with Pool(processes=8) as pool:
        results = pool.starmap(em_task, tasks)

    best_models = {}
    for k in K:
        # Find best model for each k (based on LL)
        k_results = [(ll, mixture, seed) for (kk, seed, ll, mixture) in results if kk == k]
        best_ll, best_mixture, best_seed = max(k_results, key=lambda tup: tup[0])
        best_models[k] = (best_ll, best_mixture)

        print(f"Best model for k={k} (seed={best_seed}) -> log-likelihood: {best_ll:.2f}")

    # Compute BICs
    bics = {}
    for k, (ll, mixture) in best_models.items():
        bics[k] = bic(X, mixture, ll)
        print(f"BIC for k={k}: {bics[k]:.2f}")

    # Find k with the lowest BIC
    best_k = min(bics, key=bics.get)
    print(f"\n✅ Best number of clusters: k={best_k} (BIC={bics[best_k]:.2f})")
    return best_k, bics

from common import init, bic
from typing import Tuple
import numpy as np

def compute_bic_for_k(X: np.ndarray, k: int) -> float:
    """
    Runs EM with a single initialization and computes the BIC for a given k.

    Args:
        X: (n, d) array with incomplete entries (0 for missing)
        k: number of Gaussian components

    Returns:
        float: the BIC score of the trained mixture model
    """
    # Fixed, deterministic initialization (e.g., seed=0)
    mixture, post = init(X, k, seed=0)
    
    # Run EM
    _, log_likelihood = em.run(X, mixture, post)

    # Compute BIC
    return bic(X, mixture, log_likelihood)

for k in range(1, 10):
    score = compute_bic_for_k(X, k)
    print(f"BIC for k={k}: {score:.2f}")



