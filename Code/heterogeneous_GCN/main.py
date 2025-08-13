import csv
import random
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.preprocess import (
    prepare_user_features,
    prepare_food_features,
    split_users,
    sample_negative_edges,
    evaluate_model,
)
from models.gnn_model import SimpleGCNRec, SimpleWeightedRec

USE_WEIGHTED = True  # True 用 SimpleWeightedRec；False 用 SimpleGCNRec
GRID_LR = [0.001, 0.005]
GRID_HIDDEN = [32, 64, 128]
EPOCHS = 100
NEG_SAMPLE_RATIO = 3
PATIENCE = 30
SEED = 4210
BEST_PATH = Path("best_model.pth")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

user_feat = prepare_user_features("data/processed_questionnaire_data.csv").to(DEVICE)
food_feat = prepare_food_features("data/food_labels.csv").to(DEVICE)
rating_mat = np.genfromtxt("data/processed_questionnaire_data.csv", delimiter=",", skip_header=1, usecols=range(19, 49), dtype=int)
N_USER, N_FOOD = rating_mat.shape
TRAIN_USERS, TEST_USERS = split_users(user_feat)

pos_u, pos_f = np.where(rating_mat != 0)
edge_index = torch.tensor(np.vstack([pos_u, pos_f + N_USER]), dtype=torch.long, device=DEVICE)
edge_index = torch.cat([edge_index, edge_index[[1, 0]]], 1)
edge_weight = torch.tensor(rating_mat[pos_u, pos_f] / 5.0, dtype=torch.float32, device=DEVICE)
edge_weight = torch.cat([edge_weight, edge_weight])

logfile = open("train_log.csv", "w")
logfile.write("lr,hidden,epoch,train_loss,val_mse,lr_current\n")

best_metric = float("-inf")  # metric = -val_mse，越大越好
best_cfg = {}

# 调参
for lr, hidden in itertools.product(GRID_LR, GRID_HIDDEN):
    print(f"\n=== Trying config: lr={lr}, hidden={hidden} ===")
    if USE_WEIGHTED:
        model = SimpleWeightedRec(user_feat.size(1), food_feat.size(1), hidden).to(DEVICE)
        criterion = nn.MSELoss()
    else:
        model = SimpleGCNRec(user_feat.size(1), food_feat.size(1), hidden).to(DEVICE)
        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    best_epoch, patience_counter = 0, 0

    for ep in range(EPOCHS):
        model.train()
        zipped = list(zip(pos_u, pos_f))
        random.shuffle(zipped)
        pos_u, pos_f = zip(*zipped)
        batch_pos = list(zip(pos_u, pos_f))

        if USE_WEIGHTED:
            u_idx = torch.tensor([u for u, _ in batch_pos], device=DEVICE)
            f_idx = torch.tensor([f + N_USER for _, f in batch_pos], device=DEVICE)
            rate_lbl = torch.tensor([rating_mat[u, f] for u, f in batch_pos], dtype=torch.float32, device=DEVICE)

            opt.zero_grad()
            pred = model(user_feat, food_feat, edge_index, edge_weight, u_idx, f_idx)
            mask = rate_lbl != 0  # 只在评分不为0处计算 loss
            loss = criterion(pred[mask], rate_lbl[mask])
            loss.backward()
            opt.step()
        else:
            batch_neg = sample_negative_edges(TRAIN_USERS, rating_mat, len(batch_pos) * NEG_SAMPLE_RATIO)
            all_u = [u for u, _ in batch_pos] + [u for u, _ in batch_neg]
            all_f = [f for _, f in batch_pos] + [f for _, f in batch_neg]
            exist_lbl = [1] * len(batch_pos) + [0] * len(batch_neg)
            rate_lbl = [rating_mat[u, f] for u, f in batch_pos] + [0] * len(batch_neg)

            u_idx = torch.tensor(all_u, device=DEVICE)
            f_idx = torch.tensor(np.array(all_f) + N_USER, device=DEVICE)
            exist_lbl = torch.tensor(exist_lbl, dtype=torch.float32, device=DEVICE)
            rate_lbl = torch.tensor(rate_lbl, dtype=torch.float32, device=DEVICE)

            opt.zero_grad()
            exist_pred, rate_pred = model(user_feat, food_feat, edge_index, u_idx, f_idx)
            loss_exist = bce_loss(exist_pred, exist_lbl)
            loss_rating = mse_loss(rate_pred[exist_lbl == 1], rate_lbl[exist_lbl == 1])
            loss = loss_exist + 0.5 * loss_rating
            loss.backward()
            opt.step()


        val_mse, val_mae = evaluate_model(model, user_feat, food_feat, edge_index, edge_weight,
                                 rating_mat, TRAIN_USERS, TEST_USERS, N_USER, N_FOOD, USE_WEIGHTED)
        metric = -val_mse  
        current_lr = opt.param_groups[0]['lr']

        logfile.write(f"{lr},{hidden},{ep},{loss.item():.6f},{val_mse:.6f},{current_lr}\n")
        logfile.flush()

        print(f"Epoch {ep:03d} | Loss: {loss.item():.4f} | Val MSE: {val_mse:.4f} | LR: {current_lr:.6f} | Val MAE: {val_mae:.4f}")

        scheduler.step(val_mse)
        if metric > best_metric:
            best_metric = metric
            best_cfg = {"lr": lr, "hidden": hidden}
            torch.save(model.state_dict(), BEST_PATH)
            best_epoch = ep
            patience_counter = 0
            print("New best model saved")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping")
            break

print("\nBest config:", best_cfg, "Metric:", best_metric)
print("Best model saved to:", BEST_PATH)
logfile.close()
