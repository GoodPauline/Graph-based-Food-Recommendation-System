import torch
import torch.nn as nn
from food_recommendor_gat import construct_dataloaders, evaluate_best_model
from data_preparation import read_user_food_info, read_scores


def main(
    n_user: int,
    test_size: float,
    gat_hidden_dim: int,
    gat_output_dim: int,
    mlp_hidden_dims: list,
):
    # Read user data, food data, and scores
    user_features, food_features = read_user_food_info(n_user=n_user)
    scores = read_scores(n_user=n_user)

    # Extract necessary dimension information
    _, n_user_features = user_features.shape   # number of users has been given so no need to extract it
    n_food, n_food_features = food_features.shape

    # Prepare data for training and test
    _, test_loader = construct_dataloaders(
        scores=scores,
        test_size=test_size,
        batch_size=n_food
    )

    criterion = nn.MSELoss()

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


def show_predicted_scores(predictions: torch.Tensor) -> None:
    n_int_on_each_row = 30

    print("\nPredicted scores:")

    flattened = predictions.flatten().to(torch.int)
    for i in range(0, len(flattened), n_int_on_each_row):
        print(flattened[i : i+n_int_on_each_row].tolist())

    # Count the number of integers 0-5
    counts = torch.bincount(flattened, minlength=6)
    # Extract the number of integers 1-5
    result = counts[1:6]

    print("Number of each score (1-5):", list(result.tolist()))
    return


if __name__ == "__main__":
    mse_loss, rmse, mae, predictions = main(
        n_user=300,
        test_size=0.2,
        gat_hidden_dim=24,
        gat_output_dim=16,
        mlp_hidden_dims=[32]
    )

    show_predicted_scores(predictions=predictions)