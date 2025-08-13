import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

def prepare_user_features(path, first_n_columns=19, age_column='age'):
    df = pd.read_csv(path, encoding='utf-8-sig')
    X_raw = df.iloc[:, :first_n_columns].copy()
    if age_column in X_raw.columns:
        encoder = OneHotEncoder(sparse_output=False)
        age_onehot = encoder.fit_transform(X_raw[[age_column]])
        age_cols = [f'{age_column}_{cat}' for cat in encoder.categories_[0]]
        age_onehot = pd.DataFrame(age_onehot, columns=age_cols)
        X_processed = pd.concat([X_raw.drop(columns=[age_column]), age_onehot], axis=1)
    else:
        X_processed = X_raw
    return torch.tensor(X_processed.astype(np.float32).values)

def prepare_food_features(path):
    df = pd.read_csv(path, encoding='utf-8-sig')
    X_processed = df.iloc[:, 1:]  # 第一列是食物名字
    return torch.tensor(X_processed.astype(np.float32).values)

def split_users(user_features, test_size=0.2, random_state=42):
    indices = np.arange(user_features.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    return train_idx, test_idx

def sample_negative_edges(train_users, rating_mat, n_samples):
    u = np.random.choice(train_users, size=n_samples * 2)
    f = np.random.choice(rating_mat.shape[1], size=n_samples * 2)
    mask = rating_mat[u, f] == 0
    neg_edges = list(zip(u[mask], f[mask]))
    return neg_edges[:n_samples]

def evaluate_model(model, user_feat, food_feat, edge_index, edge_weight, rating_mat, train_users, test_users, n_user, n_food, use_weighted):
    model.eval()
    with torch.no_grad():
        if use_weighted:
            u_idx = torch.arange(n_user).repeat_interleave(n_food)
            f_idx = torch.arange(n_food).repeat(n_user) + n_user
            preds = model(user_feat, food_feat, edge_index, edge_weight, u_idx, f_idx).cpu().numpy()
            if isinstance(preds, np.ndarray):
                preds = np.round(preds) 
            else:
                preds = torch.round(preds)  

            labels = rating_mat.flatten()
            mask = labels != 0  # 只在有评分的样本上算 MSE
            val_mse = mean_squared_error(labels[mask], preds[mask])
            val_mae = mean_absolute_error(labels[mask], preds[mask])
            return val_mse, val_mae
        else:
            mses = []
            maes = []
            for uid in test_users:
                u_idx = torch.tensor([uid] * n_food)
                f_idx = torch.tensor(np.arange(n_food) + n_user)
                exist_p, rate_p = model(user_feat, food_feat, edge_index, u_idx, f_idx)
                rate_p = torch.round(rate_p)
                labels = (rating_mat[uid] != 0).astype(float)
                if labels.sum() == 0:
                    continue
                
                
                mse = mean_squared_error(rating_mat[uid][labels == 1], rate_p.cpu().numpy()[labels == 1])
                mae = mean_absolute_error(rating_mat[uid][labels == 1], rate_p.cpu().numpy()[labels == 1])
                mses.append(mse)
                maes.append(mae)
                
            val_mse = np.mean(mses) if len(mses) > 0 else np.nan
            val_mae = np.mean(maes) if len(maes) > 0 else np.nan
            return val_mse, val_mae
