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
# 农业变量名
# ----------------------------
VAR_NAMES = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
PR_IDX = VAR_NAMES.index("pr")
TMMN_IDX = VAR_NAMES.index("tmmn")
TMMX_IDX = VAR_NAMES.index("tmmx")

# ----------------------------
# ⭐ 新增：计算生长季农业指标
# ----------------------------
def compute_agro_features(climate_seq):
    """
    climate_seq: (12, C)
    返回: (agro_dim,) 特征向量
    """
    pr = climate_seq[:, PR_IDX]
    tmmn = climate_seq[:, TMMN_IDX]
    tmmx = climate_seq[:, TMMX_IDX]
    tavg = (tmmn + tmmx) / 2.0

    # 生长季：4–9月 (index 3 to 8)
    gs = slice(3, 9)
    # 灌浆期：7–9月 (index 6 to 8)
    grain_fill = slice(6, 9)

    features = []
    features.append(np.nansum(pr[gs]))                     # 总降水
    features.append(np.nanmean(tavg[gs]))                  # 平均温度
    features.append(np.nansum(np.maximum(tavg[gs] - 10, 0)))  # GDD >10°C
    features.append(np.nanmax(tmmx[grain_fill]))           # 灌浆期最高温
    features.append(np.nanmin(tmmn[gs]))                   # 生长季最低温

    return np.array(features, dtype=np.float32)

# ----------------------------
# 构建特征：气候序列 + 土壤 + 农业指标
# ----------------------------
def build_features_full(df, aux_df):
    soil_cols = [col for col in aux_df.columns if col.startswith("soil_")]
    climate_seq_list = []
    soil_feat_list = []
    agro_feat_list = []  # ← 新增
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
            
            # 计算农业特征
            agro_feat = compute_agro_features(climate_seq)
            
            climate_seq_list.append(climate_seq)
            soil_feat_list.append(soil_feat)
            agro_feat_list.append(agro_feat)
        else:
            climate_seq_list.append(np.full((12, len(VAR_NAMES)), np.nan, dtype=np.float32))
            soil_feat_list.append(np.full(len(soil_cols), np.nan, dtype=np.float32))
            agro_feat_list.append(np.full(5, np.nan, dtype=np.float32))  # 5个农业特征
        
        if "Yield" in row:
            y_list.append(row["Yield"])
    
    climate_seqs = np.stack(climate_seq_list)
    soil_feats = np.stack(soil_feat_list)
    agro_feats = np.stack(agro_feat_list)  # (N, 5)
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
# 构建完整特征
# ----------------------------
climate_train, soil_train, agro_train, y_train = build_features_full(train_df, aux_df)
climate_test, soil_test, agro_test, _ = build_features_full(test_df, aux_df)

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
S_soil = soil_train.shape[1]
S_agro = agro_train.shape[1]

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

# 拼接土壤到每个时间步
soil_train_tiled = np.tile(soil_train_scaled[:, None, :], (1, T, 1))
soil_test_tiled = np.tile(soil_test_scaled[:, None, :], (1, T, 1))

# 最终输入：气候 + 土壤（时序），农业特征（全局）
X_train_seq = np.concatenate([climate_train_scaled, soil_train_tiled], axis=-1)  # (N, 12, C+S_soil)
X_test_seq = np.concatenate([climate_test_scaled, soil_test_tiled], axis=-1)

agro_train_final = agro_train_scaled  # (N, 5)
agro_test_final = agro_test_scaled

# ----------------------------
# 按 Field_ID 分组划分
# ----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X_train_seq, y_train, groups=train_df["Field_ID"]))

X_tr_seq, X_val_seq = X_train_seq[train_idx], X_train_seq[val_idx]
agro_tr, agro_val = agro_train_final[train_idx], agro_train_final[val_idx]
y_tr, y_val = y_train[train_idx], y_train[val_idx]

print(f"Train samples: {len(X_tr_seq)} | Val samples: {len(X_val_seq)}")

# ----------------------------
# Dataset（支持序列 + 全局特征）
# ----------------------------
class YieldDataset(Dataset):
    def __init__(self, X_seq, X_global, y=None):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_global = torch.tensor(X_global, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X_seq[idx], self.X_global[idx], self.y[idx]
        return self.X_seq[idx], self.X_global[idx]

# ----------------------------
# ⭐ 改进版模型：融合序列 + 农业特征
# ----------------------------
class AgroAwareTransformerYieldPredictor(nn.Module):
    def __init__(self, seq_len=12, seq_input_dim=14+20, global_input_dim=5, 
                 embed_dim=128, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(seq_input_dim, embed_dim)
        self.month_embedding = nn.Embedding(seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ⭐ 初始化 lag_weights 集中在生长季 (4–9月 → index 3–8)
        init_lag = torch.zeros(seq_len)
        init_lag[3:9] = 1.0
        self.lag_weights = nn.Parameter(init_lag)
        
        # 融合层：序列表示 + 农业特征
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim + global_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x_seq, x_global):
        B, L, D = x_seq.shape
        # 序列嵌入 + 月份嵌入
        x_feat = self.embedding(x_seq)
        month_ids = torch.arange(L, device=x_seq.device).unsqueeze(0).expand(B, -1)
        x_month = self.month_embedding(month_ids)
        x = x_feat + x_month
        
        out = self.transformer(x, mask=self.causal_mask)
        weights = torch.softmax(self.lag_weights, dim=0)
        seq_repr = (out * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (B, E)
        
        # 融合全局农业特征
        combined = torch.cat([seq_repr, x_global], dim=1)  # (B, E + G)
        return self.fusion(combined).squeeze(-1)

# ----------------------------
# 训练设置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seq_input_dim = X_tr_seq.shape[-1]
global_input_dim = agro_tr.shape[1]

model = AgroAwareTransformerYieldPredictor(
    seq_len=12,
    seq_input_dim=seq_input_dim,
    global_input_dim=global_input_dim,
    embed_dim=128,
    nhead=8,
    num_layers=2,
    dropout=0.1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

train_dataset = YieldDataset(X_tr_seq, agro_tr, y_tr)
val_dataset = YieldDataset(X_val_seq, agro_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----------------------------
# 训练循环
# ----------------------------
best_val_rmse = float('inf')
for epoch in range(50):
    model.train()
    for x_seq, x_global, y in train_loader:
        x_seq, x_global, y = x_seq.to(device), x_global.to(device), y.to(device)
        pred = model(x_seq, x_global)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x_seq, x_global, y in val_loader:
            x_seq, x_global, y = x_seq.to(device), x_global.to(device), y.to(device)
            pred = model(x_seq, x_global)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(y.cpu().numpy())
    
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_agro_aware_model.pth")

    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f}")

# ----------------------------
# 最终评估
# ----------------------------
model.load_state_dict(torch.load("best_agro_aware_model.pth", map_location=device))
model.eval()

with torch.no_grad():
    val_preds = []
    for x_seq, x_global, _ in val_loader:
        x_seq, x_global = x_seq.to(device), x_global.to(device)
        pred = model(x_seq, x_global)
        val_preds.append(pred.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    final_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print("\n Final Validation RMSE (Agro-Aware Model):", f"{final_rmse:.4f}")

# 打印学习到的月份权重
lag_weights = torch.softmax(model.lag_weights, dim=0).detach().cpu().numpy()
print("\n Learned lag weights (Month 1 to 12):")
for i, w in enumerate(lag_weights, 1):
    print(f"  Month {i:2d}: {w:.4f}")

# ----------------------------
# 测试预测 & 提交
# ----------------------------
test_dataset = YieldDataset(X_test_seq, agro_test_final)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    test_preds = []
    for x_seq, x_global in test_loader:
        x_seq, x_global = x_seq.to(device), x_global.to(device)
        pred = model(x_seq, x_global)
        test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds)

submission = pd.DataFrame({
    "Field_ID": test_df["Field_ID"],
    "Yield": np.clip(test_preds, 0, None)
})
submission.to_csv("submission_agro_aware.csv", index=False)
print("\n Submission saved to submission_agro_aware.csv")
