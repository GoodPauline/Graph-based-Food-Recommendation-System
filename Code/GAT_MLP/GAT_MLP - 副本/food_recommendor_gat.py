import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from gat import GAT
from mlp import MLP
from data_preparation import read_user_food_info, read_scores
# from adjacency_matrix_computation import build_gat_adjacency


DIR_BEST_MODEL = "best_models"
FILE_NAME_BEST_MODEL = "best_recommendor_gat.pth"
LOWER_S, UPPER_S = 1, 5   # Lower and upper bound of a valid score


class PairwiseDataset(Dataset):
    """
    Pair-wise data set
    
    Args:
        ratings (Tensor): Users' ratings on foods
    """

    def __init__(self, scores:torch.Tensor):
        self.samples = []

        n_user, n_food = scores.shape

        for u in range(n_user):
            for f in range(n_food):
                score = scores[u, f].item()   # score by user u on food f
                if score > 0:   # Only use the known scores
                    self.samples.append((u, f, score))

        return
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[int, int, torch.Tensor]:
        u, f, s = self.samples[idx]   # user index, food index, and score
        return u, f, torch.tensor([s], dtype=torch.float32)


class GraphRecSys(nn.Module):
    """
    Unified model combining GCN for node embedding and MLP for rating prediction

    Args:
        n_user_features (int): Number of user features
        n_food_features (int): Number of food features
        gcn_hidden_dim (int): Dimension of GCN's hidden layer
        gcn_output_dim (int): Dimension of GCN's output layer
        mlp_hidden_dims (list): Dimensions of MLP's hidden layers
    """

    def __init__(self, n_user_features:int, n_food_features:int, 
                 gat_hidden_dim:int, gat_output_dim:int, mlp_hidden_dims:list,
                 gat_dropout:float=0.6, gat_alpha:float=0.2, gat_n_heads:int=4) -> None:
        
        super(GraphRecSys, self).__init__()
        
        self.user_gat = GAT(
            in_features=n_user_features,
            hidden_dim=gat_hidden_dim,
            out_features=gat_output_dim,
            dropout=gat_dropout,
            alpha=gat_alpha,
            n_heads=gat_n_heads
        )
        
        self.food_gat = GAT(
            in_features=n_food_features,
            hidden_dim=gat_hidden_dim,
            out_features=gat_output_dim,
            dropout=gat_dropout,
            alpha=gat_alpha,
            n_heads=gat_n_heads
        )
        
        # MLP for rating prediction
        self.mlp = MLP(
            input_dim=gat_output_dim * 2,  # concatenated user and food embeddings
            hidden_dims=mlp_hidden_dims
        )

        # KNN parameters (can be made configurable)
        self.k = 5
        self.user_metric = 'euclidean'  # for demographic features
        self.food_metric = 'cosine'    # for food feature vectors

        return
    
    def forward(self, user_nodes:torch.Tensor, food_nodes:torch.Tensor):
        """
        Modified forward pass that handles adjacency computation internally
        """
        # Compute adjacency matrices on-the-fly
        A_user = self._build_adjacency(user_nodes, metric=self.user_metric)
        A_food = self._build_adjacency(food_nodes, metric=self.food_metric)
        
        # Generate embeddings
        user_emb = self.user_gat(user_nodes, A_user)
        food_emb = self.food_gat(food_nodes, A_food)
        
        # Concatenate and predict
        pair_emb = torch.cat([user_emb, food_emb], dim=1)
        return self.mlp(pair_emb)
    
    def _build_adjacency(self, X:torch.Tensor, metric:str='euclidean'):
        """
        Internal method to build adjacency matrix for a batch
        """
        device = X.device
        
        # Convert to numpy for sklearn (consider implementing pure PyTorch version for GPU)
        X_np = X.detach().cpu().numpy()
        
        if metric == 'cosine':
            sim_matrix = cosine_similarity(X_np)
            n_nodes = sim_matrix.shape[0]
            adj = np.zeros_like(sim_matrix)
            
            # Select top-k neighbors
            for i in range(n_nodes):
                top_k = np.argpartition(-sim_matrix[i], self.k+1)[:self.k+1]
                top_k = top_k[top_k != i]  # remove self
                adj[i, top_k] = sim_matrix[i, top_k]
            
            adj = np.maximum(adj, adj.T)  # make symmetric
        else:
            nbrs = NearestNeighbors(n_neighbors=self.k+1, metric=metric).fit(X_np)
            adj = nbrs.kneighbors_graph(X_np, mode='connectivity').toarray()
            np.fill_diagonal(adj, 0)  # remove self-connections
            adj = np.maximum(adj, adj.T)  # make symmetric
        
        # Convert back to tensor
        adj = torch.FloatTensor(adj).to(device)
        
        # For GAT, we can optionally normalize the adjacency matrix here
        return adj
    
    def train_model(
        self,
        n_epoch: int,
        train_loader: DataLoader,
        test_loader: DataLoader,
        user_features: torch.Tensor,
        food_features: torch.Tensor,
        n_patience_epoch: int,
        optimizer: optim.Adam,
        criterion: nn.MSELoss,
        scheduler: optim.lr_scheduler.ReduceLROnPlateau
    ) -> None:
        """
        Train the unified model

        Method (in each epoch):
        1. Load training data in batches and for each batch:
            1. Extract the used data (as nodes) according to the training data batch
            2. Calculate adjacency matrices for user nodes and food nodes
            3. Forward propagation using nodes and adjacency matrices
            4. Calculate loss and conduct backward propagation to update parameters
        2. Calculate and print average loss of each data batch
        3. Schedule the learning rate according to the average loss
        4. If the model is improved (epoch loss reduced), then update and save the best model
           Else, count this epoch, and when the patience number is reached, conduct early stopping

        Args:
            n_epoch (int): Number of epochs
            train_loader (DataLoader): Loader that provides training data in batches
            user_features (torch.Tensor): Matrix of user features
            food_features (torch.Tensor): Matrix of food features
            n_patience_epoch (int): Number of patience epochs before early stopping
            optimizer (optim.Adam): Optimizer of the loss function
            criterion (nn.MSELoss): MSE loss function
            scheduler (optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler
        """
        global DIR_BEST_MODEL, FILE_NAME_BEST_MODEL

        self.train()

        best_loss = float('inf')
        n_epochs_no_improve = 0   # Number of epochs that did not improve the best model

        for epoch in range(n_epoch):
            epoch_loss = 0.0   # Total loss in this epoch

            for user_indices, food_indices, scores in train_loader:
                optimizer.zero_grad()

                user_nodes, food_nodes = get_nodes(
                    user_features=user_features,
                    food_features=food_features,
                    user_indices=user_indices,
                    food_indices=food_indices
                )
                
                prediction = self(
                    user_nodes=user_nodes,
                    food_nodes=food_nodes
                )

                loss = criterion(prediction, scores)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()   # Loss from the current epoch

            avg_loss_batch = epoch_loss / len(train_loader)   # average loss of each batch of training data in this epoch
            mse_loss, rmse, mae, _ = self.evaluate_model(
                test_loader=test_loader,
                user_features=user_features,
                food_features=food_features,
                criterion=criterion
            )
            print(f'Epoch {epoch+1}/{n_epoch}, train loss={avg_loss_batch:.4f}, test loss={mse_loss:.4f}, rmse={rmse:.4f}, mae={mae:.4f}')

            if scheduler is not None:
                scheduler.step(avg_loss_batch)

            if epoch == 0 or epoch_loss < best_loss:   # the best model is improved
                # Refresh the best loss and the number of epochs that did not improve the best models
                best_loss = epoch_loss
                n_epochs_no_improve = 0

                # Save the best model
                path_best_model = os.path.join(DIR_BEST_MODEL, FILE_NAME_BEST_MODEL)
                torch.save(self.state_dict(), path_best_model)
            
            else:   # the best model is not improved
                n_epochs_no_improve += 1

                if n_epochs_no_improve >= n_patience_epoch:   # Number of patience epochs for early stopping is reached
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        return
    
    def evaluate_model(
        self,
        test_loader: DataLoader,
        user_features: torch.Tensor,
        food_features: torch.Tensor,
        criterion: nn.MSELoss
    ) -> tuple[float, float, float, torch.Tensor]:
        """
        Evaluate the unified model

        Args:
            test_loader (DataLoader): Loader that provides test data in batches
            user_features (torch.Tensor): Matrix of user features
            food_features (torch.Tensor): Matrix of food features
            criterion (nn.MSELoss): MSE loss function
        
        Returns:
            results (tuple[float, float, float, torch.Tensor]): `(mse_loss, rmse, mae, predictions)`
                - `mse_loss`: Mean Squared Error loss
                - `rmse`: Root Mean Square Error
                - `mae`: Mean Absolute Error
                - `predictions`: Predicted scores that are intergers in [1, 5]
        """
        global LOWER_S, UPPER_S

        self.eval()

        total_loss, total_squared_error, total_absolute_error = 0.0, 0.0, 0.0
        count = 0

        all_predictions = []

        with torch.no_grad():
            for user_indices, food_indices, scores in test_loader:
                user_nodes, food_nodes = get_nodes(
                    user_features=user_features,
                    food_features=food_features,
                    user_indices=user_indices,
                    food_indices=food_indices
                )
                
                prediction = self(
                    user_nodes=user_nodes,
                    food_nodes=food_nodes
                )

                valid_pred = torch.round(torch.clamp(prediction, LOWER_S, UPPER_S))
                all_predictions.append(valid_pred)

                total_squared_error += ((valid_pred - scores)**2).sum().item()
                total_absolute_error += torch.abs(valid_pred - scores).sum().item()
                total_loss += criterion(prediction, scores).item() * len(scores)

                count += len(scores)
        
        mse_loss = total_loss / count
        rmse = torch.sqrt(torch.tensor(total_squared_error / count)).item()
        mae = total_absolute_error / count

        predictions = torch.cat(all_predictions, dim=0)   # combine different batches of predictions into a whole matrix

        return mse_loss, rmse, mae, predictions


def construct_dataloaders(
    scores: torch.Tensor,
    test_size: float,
    batch_size: int
) -> tuple[DataLoader, DataLoader]:
    """
    Construct two dataloaders for training and testing, respectively

    Method:
    1. Split user node embeddings and scores into training and testing data
    2. Create training and testing datasets instance

    Args:
        scores (torch.Tensor): Scores by users on foods
        test_size (float): Proportion of test data
        batch_size (int): Size of each training data batch
    
    Returns:
        data_loaders (tuple[DataLoader, DataLoader]): `(train_loader, test_loader)`
            - `train_loader`: Data loader for training
            - `test_loader`: Data loader for testing
    """
    dataset = PairwiseDataset(scores=scores)

    train_size = int((1 - test_size) * len(dataset))
    test_size = len(dataset) - train_size

    # Random split and get: training dataset and testing dataset
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader


def get_nodes(
    user_features: torch.Tensor,
    food_features: torch.Tensor,
    user_indices: torch.Tensor,
    food_indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified version that only extracts nodes - adjacency is now handled internally
    """
    user_nodes = user_features[user_indices]
    food_nodes = food_features[food_indices]

    return user_nodes, food_nodes


def evaluate_best_model(
    n_user_features: int,
    n_food_features: int,
    gat_hidden_dim: int,
    gat_output_dim: int,
    mlp_hidden_dims: list,
    test_loader: DataLoader,
    user_features: torch.Tensor,
    food_features: torch.Tensor,
    criterion: nn.MSELoss
) -> tuple[float, float, float, torch.Tensor]:
    """
    Evaluate the best unified model

    Args:
        n_user_features (int): Number of user features
        n_food_features (int): Number of food features
        gcn_hidden_dim (int): Dimension of GCN's hidden layer
        gcn_output_dim (int): Dimension of GCN's output layer
        mlp_hidden_dims (list): Dimensions of MLP's hidden layers
        test_loader (DataLoader): Loader that provides test data in batches
        user_features (torch.Tensor): Matrix of user features
        food_features (torch.Tensor): Matrix of food features
        criterion (nn.MSELoss): MSE loss function
    
    Returns:
        results (tuple[float, float, float, torch.Tensor]): `(mse_loss, rmse, mae, predictions)`
            - `mse_loss`: Mean Squared Error loss
            - `rmse`: Root Mean Square Error
            - `mae`: Mean Absolute Error
            - `predictions`: Predicted scores that are intergers in [1, 5]
    """
    global DIR_BEST_MODEL, FILE_NAME_BEST_MODEL

    best_model = GraphRecSys(
        n_user_features=n_user_features,
        n_food_features=n_food_features,
        gat_hidden_dim=gat_hidden_dim,
        gat_output_dim=gat_output_dim,
        mlp_hidden_dims=mlp_hidden_dims
    )

    # Load the best model
    path_best_model = os.path.join(DIR_BEST_MODEL, FILE_NAME_BEST_MODEL)
    best_model.load_state_dict(torch.load(path_best_model, weights_only=True))

    mse_loss, rmse, mae, predictions = best_model.evaluate_model(
        test_loader=test_loader,
        user_features=user_features,
        food_features=food_features,
        criterion=criterion
    )

    print(f"\nBest Model Evaluation: MSE Loss={mse_loss:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    return mse_loss, rmse, mae, predictions


def main(
    n_user: int,
    test_size: float,
    gat_hidden_dim: int,
    gat_output_dim: int,
    mlp_hidden_dims: list,
    n_epoch:int,
    lr_init: float,
    lr_min: float,
    lr_decay_factor: float,
    lr_schd_patience: int,
    lr_schd_cooldown: int
):
    # Read user data, food data, and scores
    user_features, food_features = read_user_food_info(n_user=n_user)
    scores = read_scores(n_user=n_user)

    # Extract necessary dimension information
    _, n_user_features = user_features.shape   # number of users has been given so no need to extract it
    n_food, n_food_features = food_features.shape

    # Prepare data for training and test
    train_loader, test_loader = construct_dataloaders(
        scores=scores,
        test_size=test_size,
        batch_size=n_food
    )

    # Initialize model
    model = GraphRecSys(
        n_user_features=n_user_features,
        n_food_features=n_food_features,
        gat_hidden_dim=gat_hidden_dim,
        gat_output_dim=gat_output_dim,
        mlp_hidden_dims=mlp_hidden_dims
    )

    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=lr_init)
    criterion = nn.MSELoss()

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_decay_factor,
        patience=lr_schd_patience,
        cooldown=lr_schd_cooldown,
        min_lr=lr_min
    )

    model.train_model(
        n_epoch=n_epoch,
        train_loader=train_loader,
        test_loader=test_loader,
        user_features=user_features,
        food_features=food_features,
        n_patience_epoch=lr_schd_patience + lr_schd_cooldown + 1,
        criterion = criterion,
        optimizer = optimizer,
        scheduler = scheduler
    )

    mse_loss, rmse, mae, predictions = evaluate_best_model(
        n_user_features=n_user_features,
        n_food_features=n_food_features,
        gat_hidden_dim=gat_hidden_dim,
        gat_output_dim=gat_output_dim,
        mlp_hidden_dims=mlp_hidden_dims,
        test_loader=test_loader,
        user_features=user_features,
        food_features=food_features,
        criterion=criterion
    )

    return mse_loss, rmse, mae, predictions


if __name__ == "__main__":
    mse_loss, rmse, mae, predictions = main(
        n_user=300,
        test_size=0.2,
        gat_hidden_dim=24,
        gat_output_dim=16,
        mlp_hidden_dims=[32],
        n_epoch=500,
        lr_init=0.01,
        lr_min=1e-6,
        lr_decay_factor=0.5,
        lr_schd_patience=10,
        lr_schd_cooldown=8
    )