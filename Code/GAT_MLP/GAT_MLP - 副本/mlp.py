"""
Multi-layer perceptron to combine node embeddings of users and foods to predict scores.

Main function: `implement_mlp`

Main input: user node embedding and food node embedding
    1. For dimension consistent in MLP's operations, the number of users can be divided by the number of foods
    2. Format of the two tensors:
        - row: a user/food's information
        - column: one feature

Modifications:
1. Pairwise input and prediction
"""
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from data_preparation import read_scores

import os


path_user_data = "data/user_data.csv"
col_start_food_scores = 19    # Index of the starting column of food scores by users in user data

lower_bound_score, upper_bound_score = 1, 5   # Lower and upper bound of a valid score


class MLP(nn.Module):
    """
    Multi-layer perceptron for predicting a user's rating for a food item.

    Note:
    1. Input: Concatenated embedding of a user and a food
    2. Output: Predicted score, so the output dimension should be 1
    
    Args:
        input_dim (int): Dimension of input
        hidden_dims (list): List of dimensions of hidden layers
    """

    def __init__(self, input_dim:int, hidden_dims:list) -> None:
        super(MLP, self).__init__()

        layers = []   # List to store layers: Input layer + hidden layer + output layer
        prev_dim = input_dim   # The first layer, input layer's dimension

        # Create and store input layer + hidden layer
        '''
        Create and store input layer + hidden layer
            - The input layer will be included because `prev_dim` is initialized as the input dimension
        '''
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))   # layer
            layers.append(nn.ReLU())   # activation function
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))   # output layer: output dimension should be 1

        # Form layers into `nn.Sequential` for the convenience of forward propagation
        self.model = nn.Sequential(*layers)
        return
    
    def forward(self, x):
        """
        Forward propagation

        Method: `self.model(x)`
            - Reason: `self.model` forms all layers with activation functions into `nn.Sequential`
        """
        return self.model(x)
    
    def train_model(
        self, 
        train_loader: DataLoader, 
        num_epochs: int, 
        n_patience_epoch: int,
        criterion,
        optimizer,
        scheduler
    ) -> None:
        """
        Train the MLP

        Args:
            train_loader (DataLoader): Training data able to be loaded in batches. Data include
                - Data include: user node embedding, scores
            num_epochs (int): Number of epochs in training
            n_patience_epoch (int): Number of patience epochs for early stopping
            criterion: Loss function of MLP
            optimizer: Optimizer for the loss function
            scheduler: Learning rate scheduler for optimizer
        """

        self.train()   # train mode
        
        best_loss = float('inf')
        n_epochs_no_improve = 0   # Number of epochs that did not improve the best model

        for epoch in range(num_epochs):
            epoch_loss = 0.0   # Total loss in this epoch

            for input, target in train_loader:   # Batches of training data
                prediction = self(input)
                loss = criterion(prediction, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()   # Loss from the current epoch
            
            '''
            Print the current epoch's training result

            Note:
                1. `epoch+1`: Due to the property of `range`, `epoch` starts from 0
                2. `epoch_loss/len(train_data)`: Average loss from each batch of data
            '''
            avg_loss_batch = epoch_loss / len(train_loader)   # average loss of each batch of training data in this epoch
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss_batch:.3f}')

            if scheduler is not None:
                scheduler.step(avg_loss_batch)

            if epoch == 0 or epoch_loss < best_loss:   # the best model is improved
                # Refresh the best loss and the number of epochs that did not improve the best models
                best_loss = epoch_loss
                n_epochs_no_improve = 0
                # Save the best model
                filename = "pairwise_mlp.pth"
                torch.save(self.state_dict(), os.path.join("best_models", filename))
            else:   # the best model is not improved
                n_epochs_no_improve += 1
                if n_epochs_no_improve >= n_patience_epoch:   # Number of patience epochs for early stopping is reached
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        return
        
    def evaluate_model(
        self,
        test_loader:DataLoader,
        criterion
    ) -> tuple[float, float]:
        """
        Evaluate the MLP's performance

        Args:
            test_loader (DataLoader): Test dataset
            criterion: Loss function of MLP
            
        Returns:
            test_result (tuple[float, float]): `(rmse, mae)`
                - `rmse`: RMSE
                - `mae`: MAE
        """
        global lower_bound_score, upper_bound_score

        self.eval()

        total_squared_error = 0
        total_absolute_error = 0
        total_loss = 0
        count = 0

        criterion = nn.MSELoss()

        with torch.no_grad():
            for input, target in test_loader:
                prediction = self(input)
                valid_preds = torch.round(torch.clamp(prediction, lower_bound_score, upper_bound_score))

                total_squared_error += ((valid_preds - target)**2).sum().item()
                total_absolute_error += torch.abs(valid_preds - target).sum().item()
                total_loss += criterion(prediction, target).item() * len(target)   # total squared error of this batch

                count += len(target)

        rmse = np.sqrt(total_squared_error / count)
        mae = total_absolute_error / count
        mse = total_loss / count

        return mse, rmse, mae, valid_preds


if __name__ == "__main__":
    pass