import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit  # ← 关键：按Field_ID分组
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
# 加载主数据
# ----------------------------
train_df = pd.read_csv(train_file, header=None)
train_df.columns = ["Field_ID", "Year", "Quality", "Yield"]
train_df['Yield'] = pd.to_numeric(train_df['Yield'], errors='coerce')
train_df = train_df.dropna(subset=['Yield']).reset_index(drop=True)

test_df = pd.read_csv(test_file)
aux_df = pd.read_csv(aux_file)
aux_df.set_index("Field_ID", inplace=True)

# ----------------------------
# 农业变量索引（固定顺序）
# ----------------------------
VAR_NAMES = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
PR_IDX = VAR_NAMES.index("pr")
TMMN_IDX = VAR_NAMES.index("tmmn")
TMMX_IDX = VAR_NAMES.index("tmmx")

# ----------------------------
# 辅助函数：计算生长季农业指标
# ----------------------------
def compute_agro_features(climate_seq):
    """
    climate_seq: (12, 14)
    返回: (5,) 特征向量
    """
    pr = climate_seq[:, PR_IDX]      # 降水
    tmmn = climate_seq[:, TMMN_IDX]  # 最低温
    tmmx = climate_seq[:, TMMX_IDX]  # 最高温
    
    # 生长季：4-9月（索引 3 到 8）
    gs = slice(3, 9)
    
    total_pr = np.nansum(pr[gs]) if not np.all(np.isnan(pr[gs])) else 0.0
    avg_temp = np.nanmean((tmmx[gs] + tmmn[gs]) / 2) if not np.all(np.isnan(tmmx[gs])) else 0.0
    # 积温 GDD (>10°C)
    daily_mean = (tmmx[gs] + tmmn[gs]) / 2
    gdd = np.nansum(np.clip(daily_mean - 10.0, 0, None)) if not np.all(np.isnan(daily_mean)) else 0.0
    # 灌浆期高温（7-9月，索引 6-8）
    max_tmmx_grainfill = np.nanmax(tmmx[6:9]) if not np.all(np.isnan(tmmx[6:9])) else 0.0
    # 播种期低温（4-6月，索引 3-5）
    min_tmmn_sowing = np.nanmin(tmmn[3:6]) if not np.all(np.isnan(tmmn[3:6])) else 0.0

    return np.array([total_pr, avg_temp, gdd, max_tmmx_grainfill, min_tmmn_sowing], dtype=np.float32)

# ----------------------------
# 构建结构化特征（含农业特征）
# ----------------------------
def build_features_structured(df, aux_df):
    soil_cols = [col for col in aux_df.columns if col.startswith("soil_")]
    climate_seq_list = []
    soil_feat_list = []
    agro_feat_list = []  # ← 新增农业特征
    y_list = []

    for _, row in df.iterrows():
        fid = row["Field_ID"]
        year = int(row["Year"])
        
        if fid in aux_df.index:
            aux_row = aux_df.loc[fid]
            soil_feat = aux_row[soil_cols].values.astype(np.float32)
            
            climate_seq = np.full((12, len(VAR_NAMES)), np.nan, dtype=np.float32)
            for month in range(12):
                base = f"climate_{year}_{month+1}_"
                for j, var in enumerate(VAR_NAMES):
                    col = f"{base}{var}"
                    if col in aux_row.index:
                        val = aux_row[col]
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            climate_seq[month, j] = float(val)
            agro_feat = compute_agro_features(climate_seq)
        else:
            climate_seq = np.full((12, len(VAR_NAMES)), np.nan, dtype=np.float32)
            soil_feat = np.full(len(soil_cols), np.nan, dtype=np.float32)
            agro_feat = np.zeros(5, dtype=np.float32)  # 默认0
        
        climate_seq_list.append(climate_seq)
        soil_feat_list.append(soil_feat)
        agro_feat_list.append(agro_feat)
        
        if "Yield" in row:
            y_list.append(row["Yield"])
    
    climate_seqs = np.stack(climate_seq_list)  # (N, 12, 14)
    soil_feats = np.stack(soil_feat_list)      # (N, S)
    agro_feats = np.stack(agro_feat_list)      # (N, 5)
    y = np.array(y_list, dtype=np.float32) if y_list else None
    
    return climate_seqs, soil_feats, agro_feats, y

# ----------------------------
# 清洗函数
# ----------------------------
def clean_array_3d(X):
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    return X.astype(np.float32)

def clean_array_2d(X):
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e6, 1e6)
    return X.astype(np.float32)

# ----------------------------
# 构建特征
# ----------------------------
climate_train, soil_train, agro_train, y_train = build_features_structured(train_df, aux_df)
climate_test, soil_test, agro_test, _ = build_features_structured(test_df, aux_df)

climate_train = clean_array_3d(climate_train)
climate_test = clean_array_3d(climate_test)
soil_train = clean_array_2d(soil_train)
soil_test = clean_array_2d(soil_test)
agro_train = clean_array_2d(agro_train)
agro_test = clean_array_2d(agro_test)

y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y_train = np.clip(y_train, 0.0, 200.0)

# ----------------------------
# 标准化
# ----------------------------
N, T, C = climate_train.shape
S = soil_train.shape[1]
A = agro_train.shape[1]  # A = 5

# 气候标准化
climate_train_flat = climate_train.reshape(-1, C)
climate_scaler = StandardScaler()
climate_train_scaled_flat = climate_scaler.fit_transform(climate_train_flat)
climate_train_scaled = climate_train_scaled_flat.reshape(N, T, C)
climate_test_scaled = climate_scaler.transform(climate_test.reshape(-1, C)).reshape(-1, T, C)

# 土壤标准化
soil_scaler = StandardScaler()
soil_train_scaled = soil_scaler.fit_transform(soil_train)
soil_test_scaled = soil_scaler.transform(soil_test)

# 农业特征标准化
agro_scaler = StandardScaler()
agro_train_scaled = agro_scaler.fit_transform(agro_train)
agro_test_scaled = agro_scaler.transform(agro_test)

# 拼接气候 + 土壤（每月）
soil_train_tiled = np.tile(soil_train_scaled[:, None, :], (1, T, 1))
soil_test_tiled = np.tile(soil_test_scaled[:, None, :], (1, T, 1))
X_climate_soil_train = np.concatenate([climate_train_scaled, soil_train_tiled], axis=-1)  # (N, 12, 14+S)
X_climate_soil_test = np.concatenate([climate_test_scaled, soil_test_tiled], axis=-1)

# ----------------------------
# 按 Field_ID 分组划分（关键改进！）
# ----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X_climate_soil_train, y_train, groups=train_df["Field_ID"]))

X_tr_seq = X_climate_soil_train[train_idx]      # (N_tr, 12, D)
X_val_seq = X_climate_soil_train[val_idx]       # (N_val, 12, D)
X_tr_agro = agro_train_scaled[train_idx]        # (N_tr, 5)
X_val_agro = agro_train_scaled[val_idx]         # (N_val, 5)
y_tr = y_train[train_idx]
y_val = y_train[val_idx]

# ----------------------------
# Dataset（支持序列 + 全局农业特征）
# ----------------------------
class YieldDataset(Dataset):
    def __init__(self, X_seq, X_agro, y=None):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_agro = torch.tensor(X_agro, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        if self.y is not None:
            return (self.X_seq[idx], self.X_agro[idx]), self.y[idx]
        return (self.X_seq[idx], self.X_agro[idx])

# ----------------------------
# 改进的 Time-Shifted Transformer（融合农业特征）
# ----------------------------
class TimeShiftedTransformerYieldPredictor(nn.Module):
    def __init__(self, seq_len=12, seq_input_dim=14+20, agro_dim=5, embed_dim=128, nhead=8, num_layers=2, dropout=0.2):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(seq_input_dim, embed_dim)
        self.month_embed = nn.Embedding(seq_len, embed_dim)  # ← 显式月份嵌入
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True,
            layer_norm_eps=1e-6
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.lag_weights = nn.Parameter(torch.randn(seq_len))
        
        # 融合农业特征
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim + agro_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x_seq, x_agro):
        B = x_seq.size(0)
        # 序列编码
        x = self.embedding(x_seq)  # (B, L, E)
        month_ids = torch.arange(self.seq_len, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.month_embed(month_ids)  # 加月份嵌入
        
        out = self.transformer(x, mask=self.causal_mask)  # (B, L, E)
        weights = torch.softmax(self.lag_weights, dim=0)
        weighted_repr = (out * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (B, E)
        
        # 拼接农业特征
        final_repr = torch.cat([weighted_repr, x_agro], dim=1)  # (B, E + A)
        return self.regressor(final_repr).squeeze(-1)

# ----------------------------
# 训练设置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_input_dim = X_tr_seq.shape[-1]  # 14 + S
agro_dim = X_tr_agro.shape[1]       # 5

model = TimeShiftedTransformerYieldPredictor(
    seq_len=12,
    seq_input_dim=seq_input_dim,
    agro_dim=agro_dim,
    embed_dim=128,
    nhead=8,
    num_layers=3,      # ← 增加一层
    dropout=0.2        # ← 稍微增加防过拟合
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # ← 增大 weight_decay
criterion = nn.HuberLoss(delta=1.0)  # ← 关键：Huber Loss！

train_dataset = YieldDataset(X_tr_seq, X_tr_agro, y_tr)
val_dataset = YieldDataset(X_val_seq, X_val_agro, y_val)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)   # ← 增大 batch size
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ----------------------------
# 训练循环（带早停）
# ----------------------------
best_val_rmse = float('inf')
patience = 10
no_improve = 0

for epoch in range(100):  # ← 增加最大 epoch
    model.train()
    for (x_seq, x_agro), y in train_loader:
        x_seq, x_agro, y = x_seq.to(device), x_agro.to(device), y.to(device)
        pred = model(x_seq, x_agro)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ← 梯度裁剪
        optimizer.step()

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for (x_seq, x_agro), y in val_loader:
            x_seq, x_agro, y = x_seq.to(device), x_agro.to(device), y.to(device)
            pred = model(x_seq, x_agro)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(y.cpu().numpy())
    
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        no_improve = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        no_improve += 1

    print(f"Epoch {epoch+1:3d} | Val RMSE: {val_rmse:.4f}")

    if no_improve >= patience:
        print("Early stopping!")
        break

# ----------------------------
# 最终评估
# ----------------------------
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

with torch.no_grad():
    val_preds = []
    for (x_seq, x_agro), _ in val_loader:
        x_seq, x_agro = x_seq.to(device), x_agro.to(device)
        pred = model(x_seq, x_agro)
        val_preds.append(pred.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    final_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print(f"\nFinal Validation RMSE: {final_rmse:.4f}")

# ----------------------------
# 测试预测 & 提交
# ----------------------------
# 构建测试集全局农业特征
X_test_agro = agro_test_scaled
test_dataset = YieldDataset(X_climate_soil_test, X_test_agro)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

with torch.no_grad():
    test_preds = []
    for x_seq, x_agro in test_loader:
        x_seq, x_agro = x_seq.to(device), x_agro.to(device)
        pred = model(x_seq, x_agro)
        test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds)

submission = pd.DataFrame({
    "Field_ID": test_df["Field_ID"],
    "Yield": np.clip(test_preds, 0, None)
})
submission.to_csv("submission_improved.csv", index=False)
print("\nSubmission saved to submission_improved.csv")
