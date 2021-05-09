import numpy as np
from scipy import sparse


def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()

    # set zeros to inf to avoid dividing by zero
    degree[degree == 0.] = np.inf

    degree_inv = 1. / degree
    degree_inv_mat = sparse.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)

    if feat_norm.nnz == 0:
        print('ERROR: normalized adjacency matrix has only zero entries!!!!!')
        exit()

    return feat_norm


def preprocess_user_item_features(u_features, v_features):
    """
    Creates one big feature matrix out of user features and item features.
    Stacks item features under the user features.
    """

    zero_csr_u = sparse.csr_matrix((u_features.shape[0], v_features.shape[1]), dtype=u_features.dtype)
    zero_csr_v = sparse.csr_matrix((v_features.shape[0], u_features.shape[1]), dtype=v_features.dtype)

    u_features = sparse.hstack([u_features, zero_csr_u], format='csr')
    v_features = sparse.hstack([zero_csr_v, v_features], format='csr')

    return u_features, v_features
