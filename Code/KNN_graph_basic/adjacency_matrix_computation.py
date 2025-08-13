import torch
from sklearn.neighbors import NearestNeighbors


def build_knn_graph(X, k=5, output='sparse'):

    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    # construct knn
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)

    knn_graph = nbrs.kneighbors_graph(X, mode='connectivity')  # sparse adj matrix
    knn_graph.setdiag(0)
    knn_graph.eliminate_zeros()

    A_sparse = knn_graph.maximum(knn_graph.T)

    # return corresponding format
    if output == 'sparse':
        return A_sparse  # scipy.sparse
    elif output == 'numpy':
        return A_sparse.toarray()  # dense numpy array
    elif output == 'tensor':
        return torch.tensor(A_sparse.toarray(), dtype=torch.float32)  # dense torch tensor
    else:
        raise ValueError("output must be one of: 'sparse', 'numpy', 'tensor'")
