# ==============================
# 玉米产量预测：AW-MIL + 辅助特征融合
# 基于 Wang et al. (2025) "Learning county from pixels"
# ==============================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# 路径配置
# ----------------------------
DATA_DIR = "./data"
IMG_TRAIN_DIR = os.path.join(DATA_DIR, "Image_arrays_train")
IMG_TEST_DIR = os.path.join(DATA_DIR, "Image_arrays_test")


# ----------------------------
# 数据加载与预处理
# ----------------------------

def load_image(field_id, is_train=True, normalize=True):
    folder = IMG_TRAIN_DIR if is_train else IMG_TEST_DIR
    path = os.path.join(folder, f"{field_id}.npy")
    if not os.path.exists(path):
        return np.zeros((360, 41, 41), dtype=np.float32)
    img = np.load(path).astype(np.float32)
    if normalize:
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)
    return img


def get_aux_features(field_id, soil_climate_df):
    if field_id in soil_climate_df.index:
        return soil_climate_df.loc[field_id].values.astype(np.float32)
    return np.zeros(soil_climate_df.shape[1], dtype=np.float32)


# ----------------------------
# Dataset
# ----------------------------

class CropYieldDataset(Dataset):
    def __init__(self, df, soil_climate, is_train=True, aux_scaler=None):
        self.df = df.reset_index(drop=True)
        self.soil_climate = soil_climate
        self.is_train = is_train

        self.images = []
        self.aux_feats = []
        self.targets = []

        for _, row in self.df.iterrows():
            fid = row["Field_ID"]
            self.images.append(load_image(fid, is_train))
            self.aux_feats.append(get_aux_features(fid, soil_climate))
            if is_train:
                self.targets.append(row["Yield"])

        self.images = np.stack(self.images)  # (N, 360, 41, 41)
        self.aux_feats = np.stack(self.aux_feats)  # (N, D_aux)

        if aux_scaler is None:
            self.aux_scaler = StandardScaler()
            self.aux_feats = self.aux_scaler.fit_transform(self.aux_feats)
        else:
            self.aux_feats = aux_scaler.transform(self.aux_feats)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx])  # (360, 41, 41)
        aux = torch.tensor(self.aux_feats[idx])  # (D_aux,)
        if self.is_train:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return img, aux, y
        return img, aux


# ----------------------------
# AW-MIL 模型（简化但保留核心）
# ----------------------------

class AW_MIL_CropYield(nn.Module):
    def __init__(self, aux_dim, img_channels=1, embed_dim=128, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim

        # 图像编码器：1D CNN over time
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels=img_channels * 41 * 41, out_channels=64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=embed_dim, kernel_size=5, padding=2),
            nn.AdaptiveAvgPool1d(1)  # (B, E, 1)
        )

        # 辅助特征投影
        self.aux_proj = nn.Linear(aux_dim, embed_dim)

        # Attention weights for instances (here each field is one instance)
        self.attention = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # 回归头
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, img, aux):
        B = img.shape[0]
        # Reshape image: (B, T, H, W) -> (B, C*H*W, T)
        img = img.unsqueeze(1)  # (B, 1, 360, 41, 41)
        img = img.view(B, 1 * 41 * 41, 360)  # (B, 1681, 360)

        # Temporal encoding
        img_emb = self.temporal_conv(img).squeeze(-1)  # (B, E)

        # Aux encoding
        aux_emb = self.aux_proj(aux)  # (B, E)

        # Concatenate
        combined = torch.cat([img_emb, aux_emb], dim=1)  # (B, 2E)

        # Attention (for MIL; here trivial since 1 instance per bag)
        att_weights = self.attention(combined)  # (B, 1)
        att_weights = torch.softmax(att_weights, dim=0)  # across instances (if multiple)

        # Weighted sum (still valid for single instance)
        weighted_emb = att_weights * combined  # (B, 2E)

        # Regression
        out = self.regressor(weighted_emb).squeeze(-1)  # (B,)
        return out


# ----------------------------
# 主流程
# ----------------------------

def main():
    # Load data
    train_df = pd.read_csv(os.path.join("Train.csv"), header=None)
    train_df.columns = ["Field_ID", "Year", "Quality", "Yield"]
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_field_ids_with_year.csv"))
    soil_climate = pd.read_excel(os.path.join(DATA_DIR, "samply.xlsx"), engine='openpyxl')
    soil_climate.set_index("Field_ID", inplace=True)

    # Train/Val split
    train_meta, val_meta = train_test_split(train_df, test_size=0.2, random_state=42)

    # Datasets
    train_dataset = CropYieldDataset(train_meta, soil_climate, is_train=True)
    val_dataset = CropYieldDataset(val_meta, soil_climate, is_train=True, aux_scaler=train_dataset.aux_scaler)
    test_dataset = CropYieldDataset(test_df, soil_climate, is_train=False, aux_scaler=train_dataset.aux_scaler)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AW_MIL_CropYield(aux_dim=train_dataset.aux_feats.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse = float('inf')
    for epoch in range(30):
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        for img, aux, y in train_loader:
            img, aux, y = img.to(device), aux.to(device), y.to(device)
            pred = model(img, aux)
            loss = criterion(pred, y)  # MSE loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_mse += loss.item()  # since criterion is MSE

        # Validation
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for img, aux, y in val_loader:
                img, aux, y = img.to(device), aux.to(device), y.to(device)
                pred = model(img, aux)
                val_mse += criterion(pred, y).item()

        # Compute averages
        train_mse /= len(train_loader)
        val_mse /= len(val_loader)
        train_rmse = np.sqrt(train_mse)
        val_rmse = np.sqrt(val_mse)

        print(f"Epoch {epoch + 1:2d} | "
              f"Train MSE: {train_mse:.4f} | Train RMSE: {train_rmse:.4f} | "
              f"Val MSE: {val_mse:.4f} | Val RMSE: {val_rmse:.4f}")

        # Save best model based on RMSE
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_model.pth"))
            print(f"New best RMSE: {best_val_rmse:.4f} — model saved.")

    # Test prediction
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "best_model.pth")))
    model.eval()
    preds = []
    with torch.no_grad():
        for img, aux in test_loader:
            img, aux = img.to(device), aux.to(device)
            pred = model(img, aux)
            preds.extend(pred.cpu().numpy())

    # Save submission
    submission = pd.DataFrame({
        "Field_ID": test_df["Field_ID"],
        "Yield": np.clip(preds, 0, None)  # 防止负产量
    })
    submission.to_csv(os.path.join(DATA_DIR, "submission_AW_MIL.csv"), index=False)
    print("Submission saved to submission_AW_MIL.csv")


if __name__ == "__main__":

    main()
