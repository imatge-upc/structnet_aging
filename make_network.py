import numpy as np
import pandas as pd
import torch

def make_graphical_lasso(x, alpha=1e-3, max_iter=1000):
    from sklearn.covariance import GraphicalLasso

    # fit the model
    gl = GraphicalLasso(alpha=alpha, max_iter=max_iter)
    gl.fit(x)

    adj = gl.precision_
    np.fill_diagonal(adj, 0)

    src, dst = np.where(adj)
    src, dst = src[ src != dst], dst[ src != dst] # remove self loops
    values   = adj[src, dst]

    edge_index  = torch.stack([torch.LongTensor(src), torch.LongTensor(dst)], dim=0)
    edge_weight = torch.FloatTensor(values) 

    return edge_index, edge_weight
  
def make_regression_lasso(x, alpha=1e-3, max_iter=1000):
    from sklearn import linear_model

    num_nodes = x.shape[1]

    clf = linear_model.Lasso(alpha=alpha, positive=True, max_iter=max_iter)
    betas = list()
    for i in range(num_nodes):
        mx_idx = [j for j in range(num_nodes) if j!=i]
        Y_i = x[:, mx_idx]
        y_i = x[:, i]

        clf.fit(Y_i, y_i)
        betas.append( np.concatenate([ clf.coef_[:i], np.array([0]), clf.coef_[i:] ]) )

    betas = np.stack(betas, axis=0)
    adj = np.sqrt( np.multiply(betas, betas.T) )

    src, dst = np.where(adj)
    src, dst = src[ src != dst], dst[ src != dst] # remove self loops
    values   = adj[src, dst]

    edge_index  = torch.stack([torch.LongTensor(src), torch.LongTensor(dst)], dim=0)
    edge_weight = torch.FloatTensor(values) 

    return edge_index, edge_weight
