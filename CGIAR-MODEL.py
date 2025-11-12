# ==============================
# 玉米产量预测：AW-MIL + 辅助特征融合（升级版）
# 基于 Wang et al. (2025) "Learning county from pixels"
# 结合本体与遥感模态的神经-符号预测器
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
# Dataset（保持不变）
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
# 模型定义（新结构：双流 + 融合）
# ----------------------------

class OntologyEncoder(nn.Module):
    """本体模态编码器：处理辅助特征（土壤、气候）"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return x


class SatelliteEncoder(nn.Module):
    """卫星模态编码器：处理时间序列图像（3D CNN）"""
    def __init__(self, img_channels=1, embed_dim=128):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(in_channels=img_channels, out_channels=32, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))  # Global average pooling
        )
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        B, T, H, W = x.shape
        x = x.unsqueeze(1)  # (B, 1, T, H, W)
        x = self.temporal_conv(x)  # (B, 64, 1, 1, 1)
        x = x.view(B, 64)  # (B, 64)
        x = self.fc(x)  # (B, embed_dim)
        return x


class NeuralSymbolicPredictor(nn.Module):
    """神经-符号预测器：加权融合两个模态"""
    def __init__(self, embed_dim=128):
        super().__init__()
        self.weight_gen = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出权重 ∈ [0,1]
        )

    def forward(self, ontology_emb, satellite_emb):
        # 融合前先拼接
        combined = torch.cat([ontology_emb, satellite_emb], dim=1)  # (B, 2*E)

        # 生成权重
        weight = self.weight_gen(combined)  # (B, 1)

        # 加权融合
        fused = weight * satellite_emb + (1 - weight) * ontology_emb  # (B, E)

        return fused, weight  # 返回融合结果和权重（用于解释性）


class CropYieldModel(nn.Module):
    """主模型：双流 + 融合 + 回归"""
    def __init__(self, aux_dim, img_channels=1, embed_dim=128):
        super().__init__()
        self.ontology_encoder = OntologyEncoder(aux_dim, embed_dim)
        self.satellite_encoder = SatelliteEncoder(img_channels, embed_dim)
        self.fusion_module = NeuralSymbolicPredictor(embed_dim)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, img, aux):
        # 卫星模态编码
        sat_emb = self.satellite_encoder(img)  # (B, E)

        # 本体模态编码
        ont_emb = self.ontology_encoder(aux)  # (B, E)

        # 融合
        fused_emb, weights = self.fusion_module(ont_emb, sat_emb)  # (B, E), (B, 1)

        # 回归
        pred = self.regressor(fused_emb).squeeze(-1)  # (B,)

        return pred, weights  # 返回预测值和权重（可解释性）


# ----------------------------
# 主流程（更新）
# ----------------------------

def main():
    # Load data
    train_df = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))

    # Clean and convert Yield to numeric
    train_df['Yield'] = pd.to_numeric(train_df['Yield'], errors='coerce')
    train_df = train_df.dropna(subset=['Yield']).reset_index(drop=True)

    # Extract as numpy arrays (ensure numeric)
    image_ids = train_df['Field_ID'].values
    targets = train_df['Yield'].values.astype(np.float32)

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
    model = CropYieldModel(aux_dim=train_dataset.aux_feats.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse = float('inf')
    for epoch in range(30):
        model.train()
        train_loss = 0.0
        for img, aux, y in train_loader:
            img, aux, y = img.to(device), aux.to(device), y.to(device)
            pred, weights = model(img, aux)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_mse = 0.0
        with torch.no_grad():
            for img, aux, y in val_loader:
                img, aux, y = img.to(device), aux.to(device), y.to(device)
                pred, _ = model(img, aux)
                val_mse += criterion(pred, y).item()

        val_mse /= len(val_loader)
        val_rmse = np.sqrt(val_mse)

        print(f"Epoch {epoch + 1:2d} | Val RMSE: {val_rmse:.4f}")

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
            pred, weights = model(img, aux)
            preds.extend(pred.cpu().numpy())

    # Save submission
    submission = pd.DataFrame({
        "Field_ID": test_df["Field_ID"],
        "Yield": np.clip(preds, 0, None)
    })
    submission.to_csv(os.path.join(DATA_DIR, "submission_AW_MIL.csv"), index=False)
    print("Submission saved to submission_AW_MIL.csv")


if __name__ == "__main__":
    main()