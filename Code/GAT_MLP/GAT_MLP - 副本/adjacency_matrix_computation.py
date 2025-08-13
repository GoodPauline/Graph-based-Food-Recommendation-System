import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix


def build_knn_graph(X, k=5, output='sparse'):
    """
    Build a symmetric k-nearest neighbors graph from input features.
    
    Args:
        X: Input features (torch.Tensor or numpy array)
        k: Number of nearest neighbors
        output: Output format ('sparse', 'numpy' or 'tensor')
    
    Returns:
        Adjacency matrix in specified format
    """
    # Convert to numpy if input is torch tensor
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    # Initialize nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)
    
    # Get nearest neighbors - returns in CSR format by default
    knn_graph = nbrs.kneighbors_graph(X, mode='connectivity')
    
    # Remove self-connections
    knn_graph = knn_graph.tolil()  # Convert to LIL for efficient modification
    knn_graph.setdiag(0)
    
    # Make symmetric by taking maximum with transpose
    # Create new LIL matrix for symmetric result
    n_nodes = knn_graph.shape[0]
    sym_graph = lil_matrix((n_nodes, n_nodes))
    
    for i in range(n_nodes):
        for j in knn_graph.rows[i]:
            sym_graph[i,j] = max(knn_graph[i,j], knn_graph[j,i])
            sym_graph[j,i] = sym_graph[i,j]  # Ensure symmetry
    
    # Convert to CSR for efficient storage and computation
    sym_graph = sym_graph.tocsr()
    sym_graph.eliminate_zeros()
    
    # Return in requested format
    if output == 'sparse':
        return sym_graph
    elif output == 'numpy':
        return sym_graph.toarray()
    elif output == 'tensor':
        # Convert directly from CSR to torch tensor
        coo = sym_graph.tocoo()

        indices_np = np.vstack([coo.row, coo.col])
        indices = torch.as_tensor(indices_np, dtype=torch.long)
        
        values = torch.as_tensor(coo.data, dtype=torch.float32)

        sparse_tensor = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=coo.shape
        )
        
        return sparse_tensor.to_dense()
    else:
        raise ValueError("output must be one of: 'sparse', 'numpy', 'tensor'")
    

def build_knn_graph_v2(X, k=5, metric='euclidean', output='sparse'):
    """
    Build a symmetric k-nearest neighbors graph from input features.
    
    Args:
        X: Input features (torch.Tensor or numpy array)
        k: Number of nearest neighbors
        metric: Distance metric ('euclidean' or 'cosine')
        output: Output format ('sparse', 'numpy' or 'tensor')
    
    Returns:
        Adjacency matrix in specified format
    """
    # Convert to numpy if input is torch tensor
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()

    # For cosine similarity, we need to compute similarities directly
    if metric == 'cosine':
        # Compute full cosine similarity matrix
        sim_matrix = cosine_similarity(X)
        n_nodes = sim_matrix.shape[0]
        
        # Initialize adjacency matrix
        adj = lil_matrix((n_nodes, n_nodes))
        
        # For each node, select top k+1 neighbors (including self)
        for i in range(n_nodes):
            top_k_indices = np.argpartition(-sim_matrix[i], k+1)[:k+1]
            for j in top_k_indices:
                if i != j:  # Exclude self-connections
                    adj[i,j] = sim_matrix[i,j]
        
        # Make symmetric
        adj = adj.maximum(adj.T)
    else:
        # Use KNN for Euclidean distance
        nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
        knn_graph = nbrs.kneighbors_graph(X, mode='connectivity')
        knn_graph = knn_graph.tolil()
        knn_graph.setdiag(0)
        
        # Make symmetric
        adj = knn_graph.maximum(knn_graph.T)
    
    # Convert to CSR for efficiency
    adj = adj.tocsr()
    adj.eliminate_zeros()
    
    # Return in requested format
    if output == 'sparse':
        return adj
    elif output == 'numpy':
        return adj.toarray()
    elif output == 'tensor':
        coo = adj.tocoo()
        indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        values = torch.as_tensor(coo.data, dtype=torch.float32)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, coo.shape)
        return sparse_tensor.to_dense()
    else:
        raise ValueError("output must be one of: 'sparse', 'numpy', 'tensor'")
    

def build_gat_adjacency(X, k=5, metric='euclidean'):
    """
    Build adjacency matrix optimized for GAT
    
    Args:
        X: Input features
        k: Number of neighbors
        metric: Distance metric
    
    Returns:
        Sparse tensor in COO format ready for GAT
    """
    adj = build_knn_graph(X, k=k, metric=metric, output='sparse')
    coo = adj.tocoo()
    
    # Convert to PyTorch sparse tensor
    indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.as_tensor(coo.data, dtype=torch.float32)
    
    # For GAT, we can use binary adjacency (values=1) or keep original weights
    # Here we keep original weights for cosine similarity, binary for Euclidean
    if metric == 'cosine':
        pass  # Keep cosine similarity values
    else:
        values = torch.ones_like(values)  # Binary adjacency
    
    return torch.sparse_coo_tensor(
        indices=indices,
        values=values,
        size=coo.shape
    )