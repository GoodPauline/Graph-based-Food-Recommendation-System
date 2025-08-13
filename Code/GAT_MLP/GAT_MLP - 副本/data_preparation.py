import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
from sklearn.preprocessing import OneHotEncoder


n_user_read = 100    # The number of users whose information will be read
col_start_food_labels = 1    # # Index of the starting column of food labels in food data
col_start_food_scores = 19    # Index of the starting column of food scores by users in user data

path_user_data = "data/user_data.csv"
path_food_data = "data/food_labels.csv"


def extract_features(
    path: str,
    col_start: int | None,
    col_end: int | None,
    n_row: int | None,
    age_col: str = "age", 
    return_type: str = "torch"
) -> np.ndarray | torch.Tensor:
    """
    Extract one type of data for Graph Neural Network (GNN) from  the corresponding .csv file
    
    Method:
    1. Extracting specified columns and the given number of rows
    2. Performing one-hot encoding on age column (if present)
    3. Converting to numpy array or PyTorch tensor
    
    Args:
        path (str): File path to the CSV data file.
        col_start (int | None): The start of target columns' index range (`None` means access from the first column)
        col_end (int | None): The end of target columns' index range (`None` means access until the last column)
        n_row (int | None): Number of rows to extract (`None` for all rows). 
        age_col (str): Column name for age to be one-hot encoded. 
            - Defaults to 'age'.
        return_type (str): Return type of the features. Options are 
            - `numpy` or `torch`. Defaults to `torch`.
    
    Returns:
        X (numpy.ndarray | torch.Tensor): Extracted data in the required type
    """

    # Read teh given number of rows of CSV file into pandas DataFrame with UTF-8-SIG encoding
    df = pd.read_csv(path, encoding="utf-8-sig", nrows=n_row)

    # Extract specified columns as raw features (None for all columns)
    if (col_start is None) and (col_end is None):
        X_raw = df.copy()
    elif (col_start is not None) and (col_end is None):
        X_raw = df.iloc[:, col_start:].copy()
    elif (col_start is None) and (col_end is not None):
        X_raw = df.iloc[:, :col_end].copy()
    else:
        X_raw = df.iloc[:, col_start:col_end].copy()

    # Perform one-hot encoding if age column exists
    if age_col in X_raw.columns:
        encoder = OneHotEncoder(sparse_output=False)   # initialize one-hot encoder (not sparse matrix)
        age_onehot = encoder.fit_transform(X_raw[[age_col]])   # fit and transform age column to one-hot encoding

        age_cols = [f"{age_col}_{cat}" for cat in encoder.categories_[0]]   # column names for encoded features
        age_onehot = pd.DataFrame(age_onehot, columns=age_cols)   # converted to DataFrame with column names

        '''
        Combine features and replace the original age column with them
        1. Drop original age column
        2. Add one-hot encoded columns, i.e. concatenate the encoded columns 
        with the raw matrix that has dropped the original age columns
        '''
        X_processed = pd.concat([X_raw.drop(columns=[age_col]), age_onehot], axis=1)
    else:
        X_processed = X_raw  # no age column: use raw features
    
    # Convert processed DataFrame to numpy array with float32 dtype
    X_numpy = X_processed.astype(np.float32).values

    # Return based on required type
    return torch.tensor(X_numpy, dtype=torch.float32) if return_type.lower() == "torch" else X_numpy


def read_user_food_info(n_user:int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read user features and food features respectively.

    Args:
        n_user (int) The number of users

    Returns:
        user_food_info (tuple[torch.Tensor, torch.Tensor]): `user_features, food_features`
            - `user_features`: Users' features
            - `food_features`: Foods' features
    """
    global path_user_data, path_food_data
    global col_start_food_labels, col_start_food_scores

    user_features = extract_features(
        path=path_user_data, 
        col_start=0,
        col_end=col_start_food_scores,
        n_row=n_user
    )

    food_features = extract_features(
        path=path_food_data, 
        col_start=col_start_food_labels,
        col_end=None,
        n_row=None
    )

    return user_features, food_features


def read_scores(n_user:int) -> torch.Tensor:
    """
    Read the same number of rows of scores as the number of users

    Args:
        n_user (int) The number of users

    Returns:
        scores (torch.Tensor): Scores read from the user data file
    """
    global path_user_data, col_start_food_scores

    # Return the torch type of scores
    return extract_features(
        path=path_user_data,
        col_start=col_start_food_scores,
        col_end=None,
        n_row=n_user,
        return_type="torch"
    )


def generate_train_test_mask(ratings:torch.Tensor, train_ratio:float=0.8, seed=42):
    np.random.seed(seed)

    num_users, num_items = ratings.shape

    known_indices = np.argwhere(ratings > 0)   # indices where scores > 0
    num_known = len(known_indices)   # number of indices where scores > 0

    # shuffle and split
    indices = np.arange(num_known)
    np.random.shuffle(indices)

    split = int(train_ratio * num_known)

    train_idx = known_indices[indices[:split]]
    test_idx  = known_indices[indices[split:]]
    
    # init mask
    train_mask = np.zeros((num_users, num_items), dtype=bool)
    test_mask  = np.zeros((num_users, num_items), dtype=bool)

    for u, i in train_idx:
        train_mask[u, i] = True
    
    for u, i in test_idx:
        test_mask[u, i] = True
    
    return train_mask, test_mask


def split_tensor_for_train_and_test(
    X_data: torch.Tensor, 
    train_size: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split a data set into a training set and a testing set
    
    Args:
        X (torch.Tensor): The data set for splitting
        train_size (float): The proportion of training set in the whole dataset
            - It should be a float in (0, 1)
    
    Returns:
        two_sets (tuple[torch.Tensor, torch.Tensor]): `(X_train, X_test)`
            - `X_train`: training data set
            - `X_test`: testing data set
    """

    # Transform X into a dataset type for the convenience of latter operations
    X_dataset = TensorDataset(X_data)

    # Calculate the sizes of training dataset and testing dataset
    train_size = int(train_size * len(X_dataset))
    test_size = len(X_dataset) - train_size

    # Randomly split the whole dataset
    train_dataset, test_dataset = random_split(X_dataset, [train_size, test_size])

    # Extract data from the two returns
    X_train = train_dataset[:][0]
    X_test = test_dataset[:][0]

    return X_train, X_test


if __name__ == "__main__":
    pass