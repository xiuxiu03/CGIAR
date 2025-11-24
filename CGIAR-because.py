import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# 路径配置
# ----------------------------
DATA_DIR = "./data"

train_file = os.path.join(DATA_DIR, "Train.csv")
test_file = os.path.join(DATA_DIR, "test_field_ids_with_year.csv")
aux_file = os.path.join(DATA_DIR, "fields_w_additional_info.csv")

# ----------------------------
# 加载数据
# ----------------------------
train_df = pd.read_csv(train_file, header=None)
train_df.columns = ["Field_ID", "Year", "Quality", "Yield"]
train_df['Yield'] = pd.to_numeric(train_df['Yield'], errors='coerce')
train_df = train_df.dropna(subset=['Yield']).reset_index(drop=True)

test_df = pd.read_csv(test_file)
aux_df = pd.read_csv(aux_file)
aux_df.set_index("Field_ID", inplace=True)

# ----------------------------
# 特征构建函数（支持训练/测试）
# ----------------------------
def build_and_filter_features(df, aux_df, is_train=True):
    soil_cols = [col for col in aux_df.columns if col.startswith("soil_")]
    var_names = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
    
    valid_indices = []
    climate_list = []
    soil_list = []
    y_list = []

    for idx, row in df.iterrows():
        fid = row["Field_ID"]
        year = int(row["Year"])
        
        if fid not in aux_df.index:
            continue
        
        aux_row = aux_df.loc[fid]
        
        # 土壤特征
        soil_vals = aux_row[soil_cols].values.astype(np.float32)
        if np.all(~np.isfinite(soil_vals)):
            continue
        
        # 气候序列
        climate_seq = np.full((12, len(var_names)), np.nan, dtype=np.float32)
        for month in range(12):
            base = f"climate_{year}_{month+1}_"
            for j, var in enumerate(var_names):
                col = f"{base}{var}"
                if col in aux_row.index:
                    val = aux_row[col]
                    if isinstance(val, (int, float)) and np.isfinite(val):
                        climate_seq[month, j] = float(val)
        
        if np.all(~np.isfinite(climate_seq)):
            continue
        
        valid_indices.append(idx)
        climate_list.append(climate_seq)
        soil_list.append(soil_vals)
        if is_train:
            y_list.append(row["Yield"])
    
    if not climate_list:
        raise ValueError("No valid samples after filtering!")
    
    climate_arr = np.stack(climate_list)
    soil_arr = np.stack(soil_list)
    y_arr = np.array(y_list, dtype=np.float32) if is_train else None
    
    # 清洗 inf → nan
    climate_arr = np.where(np.isinf(climate_arr), np.nan, climate_arr)
    soil_arr = np.where(np.isinf(soil_arr), np.nan, soil_arr)
    
    return climate_arr, soil_arr, y_arr, valid_indices

# ----------------------------
# 构建特征
# ----------------------------
climate_train, soil_train, y_train, train_valid_idx = build_and_filter_features(train_df, aux_df, is_train=True)
climate_test, soil_test, _, _ = build_and_filter_features(test_df, aux_df, is_train=False)

print(f"✅ Train samples: {len(climate_train)} | Test samples: {len(climate_test)}")

train_df_filtered = train_df.iloc[train_valid_idx].reset_index(drop=True)

# ----------------------------
# 安全标准化：拼接后统一处理（关键修复！）
# ----------------------------
N, T, C_climate = climate_train.shape
S_soil = soil_train.shape[1]

# 拼接原始特征（未标准化）
soil_train_tiled_raw = np.tile(soil_train[:, None, :], (1, T, 1))
soil_test_tiled_raw = np.tile(soil_test[:, None, :], (1, T, 1))

X_train_raw = np.concatenate([climate_train, soil_train_tiled_raw], axis=-1)  # (N, 12, D)
X_test_raw = np.concatenate([climate_test, soil_test_tiled_raw], axis=-1)

D = X_train_raw.shape[-1]

# 展平
X_train_flat = X_train_raw.reshape(-1, D)
X_test_flat = X_test_raw.reshape(-1, D)

# 再次处理 inf → nan
X_train_flat = np.where(np.isinf(X_train_flat), np.nan, X_train_flat)
X_test_flat = np.where(np.isinf(X_test_flat), np.nan, X_test_flat)

# 处理 NaN：用训练集列均值填充
if np.isnan(X_train_flat).any():
    col_means = np.nanmean(X_train_flat, axis=0)
    col_means[np.isnan(col_means)] = 0.0
    X_train_flat = np.where(np.isnan(X_train_flat), col_means, X_train_flat)
    X_test_flat = np.where(np.isnan(X_test_flat), col_means, X_test_flat)

# 移除零方差列（防止 StandardScaler 崩溃）
variances = np.var(X_train_flat, axis=0)
valid_cols = variances > 1e-8
if not np.all(valid_cols):
    print(f"🗑️ Removing {np.sum(~valid_cols)} zero-variance columns")
    X_train_flat = X_train_flat[:, valid_cols]
    X_test_flat = X_test_flat[:, valid_cols]

# 标准化
scaler = StandardScaler()
X_train_scaled_flat = scaler.fit_transform(X_train_flat)
X_test_scaled_flat = scaler.transform(X_test_flat)

# 重塑回 (N, 12, D_new)
D_new = X_train_scaled_flat.shape[1]
X_train_full = X_train_scaled_flat.reshape(N, T, D_new)
X_test_full = X_test_scaled_flat.reshape(-1, T, D_new)

# 最终裁剪到安全 float32 范围
X_train_full = np.clip(X_train_full, -1e5, 1e5).astype(np.float32)
X_test_full = np.clip(X_test_full, -1e5, 1e5).astype(np.float32)

y_train = np.array(y_train, dtype=np.float32)
y_train = np.clip(y_train, 0.0, 200.0)

# ----------------------------
# 按 Field_ID 划分（防数据泄露）
# ----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X_train_full, y_train, groups=train_df_filtered["Field_ID"]))

X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
y_tr, y_val = y_train[train_idx], y_train[val_idx]

print(f"📊 Final splits → Train: {len(X_tr)}, Val: {len(X_val)} | Feature dim: {D_new}")

# ----------------------------
# Dataset
# ----------------------------
class YieldDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]

# ----------------------------
# Model
# ----------------------------
class TimeShiftedTransformerYieldPredictor(nn.Module):
    def __init__(self, seq_len=12, input_dim=34, embed_dim=128, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.month_embedding = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lag_logits = nn.Parameter(torch.randn(seq_len))
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def get_lag_weights(self):
        return torch.softmax(self.lag_logits, dim=0)

    def forward(self, x):
        B, L, D = x.shape
        x_feat = self.embedding(x)
        month_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x_month = self.month_embedding(month_ids)
        x = x_feat + x_month
        out = self.transformer(x, mask=self.causal_mask)
        weights = self.get_lag_weights()
        weighted_repr = (out * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return self.regressor(weighted_repr).squeeze(-1)

# ----------------------------
# 农业先验（4–9月为生长季）
# ----------------------------
def get_agricultural_prior(seq_len=12, growing_season=(3, 9)):
    prior = np.zeros(seq_len)
    prior[growing_season[0]:growing_season[1]] = 1.0
    prior = prior / prior.sum()
    return torch.tensor(prior, dtype=torch.float32)

# ----------------------------
# 训练设置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_tr.shape[-1]

model = TimeShiftedTransformerYieldPredictor(
    seq_len=12,
    input_dim=input_dim,
    embed_dim=128,
    nhead=8,
    num_layers=2,
    dropout=0.1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion_mse = nn.MSELoss()

agri_prior = get_agricultural_prior().to(device)
lambda_kl = 0.1

train_dataset = YieldDataset(X_tr, y_tr)
val_dataset = YieldDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----------------------------
# 训练
# ----------------------------
best_val_rmse = float('inf')
for epoch in range(50):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        mse_loss = criterion_mse(pred, y)
        learned_probs = model.get_lag_weights()
        kl_loss = torch.sum(agri_prior * torch.log(agri_prior / (learned_probs + 1e-8)))
        total_loss = mse_loss + lambda_kl * kl_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(y.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_model_kl_regularized.pth")

    with torch.no_grad():
        current_weights = model.get_lag_weights().cpu().numpy()
        current_kl = np.sum(agri_prior.cpu().numpy() * np.log(agri_prior.cpu().numpy() / (current_weights + 1e-8)))
    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f} | KL Loss: {current_kl:.4f}")

# ----------------------------
# 提交
# ----------------------------
model.load_state_dict(torch.load("best_model_kl_regularized.pth", map_location=device))
model.eval()

with torch.no_grad():
    val_preds = []
    for x, _ in val_loader:
        x = x.to(device)
        pred = model(x)
        val_preds.append(pred.cpu().numpy())
    final_rmse = np.sqrt(mean_squared_error(y_val, np.concatenate(val_preds)))
    print(f"\n🎯 Final Validation RMSE: {final_rmse:.4f}")

    lag_weights = model.get_lag_weights().detach().cpu().numpy()
    print("\n📅 Learned lag weights (Month 1 to 12):")
    for i, w in enumerate(lag_weights, 1):
        print(f"  Month {i:2d}: {w:.4f}")

    test_dataset = YieldDataset(X_test_full)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_preds = []
    for x in test_loader:
        x = x.to(device)
        pred = model(x)
        test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds)

submission = pd.DataFrame({
    "Field_ID": test_df["Field_ID"],
    "Yield": np.clip(test_preds, 0, None)
})
submission.to_csv("submission_kl_regularized.csv", index=False)
print(f"\n✅ Submission saved to 'submission_kl_regularized.csv'")
