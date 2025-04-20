import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")


K=[1,2,3,4]

def run_K(X, k):

    seeds=[0,1,2,3,4]

    costs=[]
    for s in seeds:
        mixture, post = common.init(X, k, s)
        mixture, post, cost = kmeans.run(X, mixture, post)
        costs.append(cost)
        common.plot(X, mixture, post, 
            title=f'Cluster size {k} and seed {s}')
    return min(costs)
'''
costs=[]
for k in K:
    cost=run_K(X,k)
    costs.append(cost)

print(costs)
'''

def run_em(X):
    mixture, ____=common.init(X, 2, 0)
    _, __ = naive_em.estep(X,mixture)
    print(_,__)

    naive_em.mstep(X,_)

run_em(X)




