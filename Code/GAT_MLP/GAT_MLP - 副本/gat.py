import torch
import torch.nn as nn
import torch.nn.functional as F
from adjacency_matrix_computation import build_knn_graph

class GATLayer(nn.Module):
    """
    Graph Attention Network layer
    
    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        dropout (float): Dropout rate
        alpha (float): Negative slope for LeakyReLU
        concat (bool): Whether to concatenate multi-head outputs
    """
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # Linear transformation for node features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, X, A):
        """
        Forward pass
        
        Args:
            X (Tensor): Node feature matrix [num_nodes, in_features]
            A (Tensor): Adjacency matrix [num_nodes, num_nodes]
            
        Returns:
            output (Tensor): Output features [num_nodes, out_features]
        """
        # Linear transformation
        h = torch.mm(X, self.W)  # [num_nodes, out_features]
        N = h.size()[0]  # Number of nodes
        
        # Prepare attention inputs
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), 
                            h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        
        # Compute attention coefficients
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Apply mask (only attend to neighbors)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(A > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout_layer(attention)
        
        # Aggregate neighbor features
        h_prime = torch.matmul(attention, h)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    """
    Two-layer Graph Attention Network
    
    Args:
        in_features (int): Input feature dimension
        hidden_dim (int): Hidden layer dimension
        out_features (int): Output feature dimension
        dropout (float): Dropout rate
        alpha (float): Negative slope for LeakyReLU
        n_heads (int): Number of attention heads
    """
    def __init__(self, in_features, hidden_dim, out_features, dropout=0.6, alpha=0.2, n_heads=4):
        super(GAT, self).__init__()
        
        # First layer with multiple attention heads
        self.attentions = [GATLayer(in_features, hidden_dim, dropout, alpha, concat=True) 
                          for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
            
        # Second layer (single attention head)
        self.out_att = GATLayer(hidden_dim * n_heads, out_features, dropout, alpha, concat=False)
        
    def forward(self, X, A):
        # First layer - multi-head attention
        X = torch.cat([att(X, A) for att in self.attentions], dim=1)
        X = F.dropout(X, 0.6, training=self.training)
        
        # Second layer
        X = self.out_att(X, A)
        return X
    

def embed_nodes_one_type_gat(nodes:torch.Tensor, hidden_dim:int, output_dim:int) -> torch.Tensor:
    """
    Embed one type of nodes using GAT
    
    Args:
        nodes (torch.Tensor): Nodes that need embedding
        hidden_dim (int): Hidden layer's dimension in GAT
        output_dim (int): Output's dimension
    Returns:
        embedded_nodes (torch.Tensor): Embedded nodes that aggregates neighbors' information
    """
    A_matrix = build_knn_graph(nodes, output="tensor")   # adjacency matrix

    gat = GAT(
        in_features=nodes.shape[1], 
        hidden_dim=hidden_dim,
        out_features=output_dim,
        dropout=0.6,
        alpha=0.2,
        n_heads=4
    )
    
    embedded_nodes = gat(nodes, A_matrix)
    return embedded_nodes