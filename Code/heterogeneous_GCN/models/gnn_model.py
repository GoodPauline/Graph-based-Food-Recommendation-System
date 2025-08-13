import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv

class SimpleGCNRec(nn.Module):
    def __init__(self, u_dim, f_dim, hidden):
        super().__init__()
        self.u_proj = nn.Linear(u_dim, hidden)
        self.f_proj = nn.Linear(f_dim, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(0.3)
        self.exist_head = nn.Linear(2 * hidden, 1)
        self.rate_head = nn.Linear(2 * hidden, 1)

    def forward(self, u_feat, f_feat, edge_index, edge_u, edge_f):
        x = torch.cat([self.u_proj(u_feat), self.f_proj(f_feat)], 0)
        x = F.leaky_relu(self.norm1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        x = F.leaky_relu(self.norm2(self.conv2(x, edge_index)))
        x = self.dropout(x)
        emb = torch.cat([x[edge_u], x[edge_f]], -1)
        exist = torch.sigmoid(self.exist_head(emb)).squeeze(-1)
        rating = 1 + 4 * torch.sigmoid(self.rate_head(emb)).squeeze(-1)
        return exist, rating

class SimpleWeightedRec(nn.Module):
    def __init__(self, u_dim, f_dim, hidden, max_rating=5.0):
        super().__init__()
        self.max_rating = max_rating
        self.u_proj = nn.Linear(u_dim, hidden)
        self.f_proj = nn.Linear(f_dim, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(0.3)
        self.reg_head = nn.Linear(2 * hidden, 1)
        # self.reg_head = nn.Sequential(
        #     nn.Linear(2 * hidden, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )


    def forward(self, u_feat, f_feat, edge_index, edge_w, edge_u, edge_f):
        x = torch.cat([self.u_proj(u_feat), self.f_proj(f_feat)], 0)
        x = F.leaky_relu(self.norm1(self.conv1(x, edge_index, edge_weight=edge_w)))
        x = self.dropout(x)
        x = F.leaky_relu(self.norm2(self.conv2(x, edge_index, edge_weight=edge_w)))
        x = self.dropout(x)
        emb = torch.cat([x[edge_u], x[edge_f]], -1)
        score = torch.sigmoid(self.reg_head(emb)).squeeze(-1) * self.max_rating
        # score = (torch.tanh(self.reg_head(emb)).squeeze(-1) + 1) / 2 * self.max_rating

        return score
