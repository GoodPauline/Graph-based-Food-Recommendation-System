import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import GCNConv, GATConv
from adjacency_matrix_computation import build_knn_graph
from data_preparation import extract_features, read_scores, generate_train_test_mask
import numpy as np

#set hyper parameters
PATH = "data/user_data.csv"
K = 6
HIDDEN_DIM = 64
EPOCHS = 350
LR = 0.005
HEADS = 4

X = extract_features(PATH, col_start=1, col_end=19, n_row=None)
ratings = read_scores(X.shape[0])
train_mask, test_mask = generate_train_test_mask(ratings.numpy())
train_mask = torch.tensor(train_mask)
test_mask = torch.tensor(test_mask)
A = build_knn_graph(X, k=K, output='sparse')
edge_index, _ = from_scipy_sparse_matrix(A)
num_users, num_items = ratings.shape

class GNNRecommender(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=1):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x, edge_index)) 
        x = F.relu(self.gat2(x, edge_index))
        return self.out(x)

# class GNNRecommender(nn.Module):
#     def __init__(self, in_dim, hidden_dim, out_dim):
#         super().__init__()
#         self.conv1 = GCNConv(in_dim, hidden_dim)
#         self.conv2 = GCNConv(hidden_dim, hidden_dim)
#         self.out = nn.Linear(hidden_dim, out_dim)

#     def forward(self, x, edge_index):
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.relu(self.conv2(x, edge_index))
#         return self.out(x)

def evaluate_regression(pred, truth, mask):
    pred_masked = pred[mask]
    truth_masked = truth[mask]
    mse = F.mse_loss(pred_masked, truth_masked).item()
    rmse = torch.sqrt(F.mse_loss(pred_masked, truth_masked)).item()
    mae = F.l1_loss(pred_masked, truth_masked).item()
    return mse, rmse, mae

def get_top_k_recommendations(pred, k=10):
    top_k_indices = torch.topk(pred, k, dim=1).indices
    return top_k_indices

def evaluate_top_k_recommendations(top_k_indices, ratings, k=10, threshold=3):
    num_users = ratings.shape[0]
    precision_at_k = 0
    recall_at_k = 0
    
    for user_idx in range(num_users):
        user_rated_items = ratings[user_idx] > threshold  
        recommended_items = top_k_indices[user_idx]
        user_rated_items = user_rated_items.to(torch.bool) 
        relevant_items = torch.sum(user_rated_items[recommended_items.cpu()])
        precision_at_k += relevant_items.item() / k
        rated_items_count = torch.sum(user_rated_items).item()
        if rated_items_count > 0:
            recall_at_k += relevant_items.item() / rated_items_count
        else:
            recall_at_k += 0  
    
    precision_at_k /= num_users
    recall_at_k /= num_users
    
    return precision_at_k, recall_at_k

model = GNNRecommender(X.shape[1], HIDDEN_DIM, num_items, heads = HEADS)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()
best_test_rmse = float('inf')
best_test_mae = float('inf')
best_model_path = "knn_graph_best_model.pth"
best_epoch = -1
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    out = model(X, edge_index)  # [num_users, num_items]
    loss = loss_fn(out[train_mask], ratings[train_mask])
    loss.backward()
    optimizer.step()
    model.eval()

    with torch.no_grad():
        pred = model(X, edge_index) #here edge_index is just equivalent to the adj matrix A
        pred = torch.round(torch.clamp(pred, 1, 5))
        train_mse, train_rmse, train_mae = evaluate_regression(pred, ratings, train_mask)
        test_mse, test_rmse, test_mae = evaluate_regression(pred, ratings, test_mask)
        top_k_recommendations = get_top_k_recommendations(pred, k=K)
        precision_at_k, recall_at_k = evaluate_top_k_recommendations(top_k_recommendations, ratings, k=K)
        if epoch == EPOCHS:
            print(pred)
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_test_mae = test_mae
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)
    if epoch % 50 == 49:
        print(f"Epoch {epoch+1:03d} | "
            f"Train Loss: {loss.item():.4f} | "
            f"Train MSE: {train_mse:.4f} | Train RMSE: {train_rmse:.4f} | Train MAE: {train_mae:.4f} | "
            f"Precision@k: {precision_at_k:.4f} | Recall@k: {recall_at_k:.4f} | "
            f"Test MSE: {test_mse:.4f} | Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}")
print("Best model saved! Best model at epoch ", best_epoch, " Best test rmse: ", best_test_rmse, " Best test mae: ", best_test_mae)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = GNNRecommender(in_dim=X.shape[1], hidden_dim=HIDDEN_DIM, out_dim=num_items, heads=HEADS).to(device)
# model.load_state_dict(torch.load("knn_graph_best_model.pth", map_location=device))
# model.eval()