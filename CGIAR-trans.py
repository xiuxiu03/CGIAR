import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
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
# 定义作物生长季（4月到9月，对应索引3~8）
# ----------------------------
GROWTH_MONTHS = list(range(3, 9))  # Apr=3, ..., Sep=8

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

# 创建 Field_ID 到 index 的映射（仅基于训练集）
all_field_ids = train_df["Field_ID"].unique()
field_id_to_idx = {fid: idx for idx, fid in enumerate(all_field_ids)}
num_fields = len(all_field_ids)

# ----------------------------
# 辅助函数：构建结构化气候序列 + 土壤特征
# ----------------------------
def build_features_structured(df, aux_df, growth_months=GROWTH_MONTHS):
    soil_cols = [col for col in aux_df.columns if col.startswith("soil_")]
    climate_seq_list = []
    soil_feat_list = []
    y_list = []
    field_id_list = []
    var_names = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]

    for _, row in df.iterrows():
        fid = row["Field_ID"]
        year = int(row["Year"])
        
        if fid in aux_df.index:
            aux_row = aux_df.loc[fid]
            soil_feat = aux_row[soil_cols].values.astype(np.float32)
            climate_seq = np.zeros((12, len(var_names)), dtype=np.float32)
            for month in growth_months:
                base = f"climate_{year}_{month+1}_"
                for j, var in enumerate(var_names):
                    col = f"{base}{var}"
                    if col in aux_row.index:
                        val = aux_row[col]
                        if isinstance(val, (int, float)) and not pd.isna(val):
                            climate_seq[month, j] = float(val)
            climate_seq_list.append(climate_seq)
            soil_feat_list.append(soil_feat)
        else:
            climate_seq_list.append(np.zeros((12, len(var_names)), dtype=np.float32))
            soil_feat_list.append(np.zeros(len(soil_cols), dtype=np.float32))
    
        if "Yield" in row:
            y_list.append(row["Yield"])
        field_id_list.append(fid)

    climate_seqs = np.stack(climate_seq_list)
    soil_feats = np.stack(soil_feat_list)
    y = np.array(y_list, dtype=np.float32) if y_list else None
    return climate_seqs, soil_feats, y, field_id_list

# ----------------------------
# 构建静态特征（用于 XGBoost）
# ----------------------------
def build_static_features(df, aux_df, growth_months=GROWTH_MONTHS):
    soil_cols = [col for col in aux_df.columns if col.startswith("soil_")]
    climate_vars = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
    
    features = []
    for _, row in df.iterrows():
        fid, year = row["Field_ID"], int(row["Year"])
        if fid not in aux_df.index:
            feat_dim = len(soil_cols) + len(climate_vars) * 4
            features.append(np.zeros(feat_dim))
            continue
        
        aux_row = aux_df.loc[fid]
        soil_feat = aux_row[soil_cols].values.astype(np.float32)
        
        climate_vals = {var: [] for var in climate_vars}
        for month in growth_months:
            for var in climate_vars:
                col = f"climate_{year}_{month+1}_{var}"
                val = aux_row[col] if col in aux_row else np.nan
                climate_vals[var].append(val)
        
        stat_feats = []
        for var in climate_vars:
            arr = np.array(climate_vals[var], dtype=np.float32)
            arr = np.nan_to_num(arr, nan=0.0)
            stat_feats.extend([
                np.mean(arr),
                np.std(arr),
                np.min(arr),
                np.max(arr)
            ])
        
        feat = np.concatenate([soil_feat, stat_feats])
        features.append(feat)
    
    return np.stack(features).astype(np.float32)

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
# 构建结构化特征
# ----------------------------
climate_train, soil_train, y_train, train_field_ids = build_features_structured(train_df, aux_df)
climate_test, soil_test, _, test_field_ids = build_features_structured(test_df, aux_df)

climate_train = clean_array_3d(climate_train)
climate_test = clean_array_3d(climate_test)
soil_train = clean_array_2d(soil_train)
soil_test = clean_array_2d(soil_test)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y_train = np.clip(y_train, 0.0, 200.0)

# ----------------------------
# 标准化
# ----------------------------
N, T, C = climate_train.shape
S = soil_train.shape[1]

climate_train_flat = climate_train.reshape(-1, C)
climate_scaler = StandardScaler()
climate_train_scaled_flat = climate_scaler.fit_transform(climate_train_flat)
climate_train_scaled = climate_train_scaled_flat.reshape(N, T, C)

climate_test_flat = climate_test.reshape(-1, C)
climate_test_scaled = climate_scaler.transform(climate_test_flat).reshape(-1, T, C)

soil_scaler = StandardScaler()
soil_train_scaled = soil_scaler.fit_transform(soil_train)
soil_test_scaled = soil_scaler.transform(soil_test)

soil_train_tiled = np.tile(soil_train_scaled[:, None, :], (1, T, 1))
soil_test_tiled = np.tile(soil_test_scaled[:, None, :], (1, T, 1))

X_train_full = np.concatenate([climate_train_scaled, soil_train_tiled], axis=-1)
X_test_full = np.concatenate([climate_test_scaled, soil_test_tiled], axis=-1)

# 转换 Field_ID 为索引
train_field_indices = np.array([field_id_to_idx.get(fid, 0) for fid in train_field_ids])
test_field_indices = np.array([field_id_to_idx.get(fid, 0) for fid in test_field_ids])

# ----------------------------
# 按 Field_ID 分组划分
# ----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(X_train_full, groups=train_df["Field_ID"]))

X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
y_tr, y_val = y_train[tr_idx], y_train[val_idx]
field_tr, field_val = train_field_indices[tr_idx], train_field_indices[val_idx]

# ----------------------------
# Dataset
# ----------------------------
class YieldDataset(Dataset):
    def __init__(self, X, field_ids, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.field_ids = torch.tensor(field_ids, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.field_ids[idx], self.y[idx]
        return self.X[idx], self.field_ids[idx]

# ----------------------------
# Enhanced Multi-Head Feature Attention Module
# ----------------------------
class EnhancedFeatureAttention(nn.Module):
    def __init__(self, embed_dim, field_emb_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.field_proj = nn.Linear(field_emb_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_emb, field_emb):
        B, L, D = x_emb.shape
        field_context = self.field_proj(field_emb).unsqueeze(1)  # [B, 1, D]
        
        Q = self.q_proj(x_emb).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(field_context).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(field_context).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = torch.softmax((Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attn_out = (attn_weights @ V).transpose(1, 2).reshape(B, L, D)
        attn_out = self.dropout(self.out_proj(attn_out))
        return self.layer_norm(x_emb + attn_out)

# ----------------------------
# 改进模型：使用 EnhancedFeatureAttention
# ----------------------------
class TimeShiftedTransformerYieldPredictor(nn.Module):
    def __init__(self, seq_len=12, input_dim=34, embed_dim=128, nhead=8, num_layers=2, dropout=0.1, num_fields=10000):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        self.field_embed = nn.Embedding(num_embeddings=num_fields, embedding_dim=16)
        self.field_dropout = nn.Dropout(dropout)
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        
        # 替换为 multi-head feature attention
        self.feature_attn = EnhancedFeatureAttention(embed_dim, field_emb_dim=16, num_heads=4, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        prior_logits = torch.tensor([1.0, 1.0, 1.5, 3.0, 3.5, 3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0])
        self.register_buffer("prior_logits", prior_logits)
        self.lag_weights = nn.Parameter(torch.log(prior_logits + 1e-6))
        
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x, field_ids):
        B, L, D = x.shape
        x_emb = self.input_embedding(x)
        field_emb = self.field_dropout(self.field_embed(field_ids))
        x_emb_weighted = self.feature_attn(x_emb, field_emb)
        out = self.transformer(x_emb_weighted, mask=self.causal_mask)
        time_weights = torch.softmax(self.lag_weights, dim=0)
        weighted_repr = (out * time_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return self.regressor(weighted_repr).squeeze(-1)

    def get_kl_loss(self):
        learned_probs = torch.softmax(self.lag_weights, dim=0)
        prior_probs = torch.softmax(self.prior_logits, dim=0)
        kl = torch.sum(prior_probs * torch.log((prior_probs + 1e-8) / (learned_probs + 1e-8)))
        return kl

# ----------------------------
# 训练深度模型
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_tr.shape[-1]

model = TimeShiftedTransformerYieldPredictor(
    seq_len=12,
    input_dim=input_dim,
    embed_dim=128,
    nhead=8,
    num_layers=2,
    dropout=0.1,
    num_fields=num_fields
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

train_dataset = YieldDataset(X_tr, field_tr, y_tr)
val_dataset = YieldDataset(X_val, field_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

best_val_rmse = float('inf')
LAMBDA_KL = 0.05

for epoch in range(50):
    model.train()
    for x, field_ids, y in train_loader:
        x, field_ids, y = x.to(device), field_ids.to(device), y.to(device)
        pred = model(x, field_ids)
        loss_pred = criterion(pred, y)
        loss_kl = LAMBDA_KL * model.get_kl_loss()
        loss = loss_pred + loss_kl
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 获取最佳模型预测（简化：直接用最后 epoch）
model.eval()
deep_val_preds, deep_train_preds = [], []
with torch.no_grad():
    for x, field_ids, y in val_loader:
        pred = model(x.to(device), field_ids.to(device))
        deep_val_preds.append(pred.cpu().numpy())
    for x, field_ids, y in train_loader:
        pred = model(x.to(device), field_ids.to(device))
        deep_train_preds.append(pred.cpu().numpy())
deep_val_preds = np.concatenate(deep_val_preds)
deep_train_preds = np.concatenate(deep_train_preds)

torch.save(model.state_dict(), "best_transformer.pth")

# ----------------------------
# 训练 XGBoost
# ----------------------------
X_static_train = build_static_features(train_df, aux_df)
X_static_test = build_static_features(test_df, aux_df)
X_static_train = clean_array_2d(X_static_train)
X_static_test = clean_array_2d(X_static_test)

X_static_tr, X_static_val = X_static_train[tr_idx], X_static_train[val_idx]

xgb_model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective='reg:squarederror'
)
xgb_model.fit(X_static_tr, y_tr)

xgb_val_pred = xgb_model.predict(X_static_val)
xgb_train_pred = xgb_model.predict(X_static_tr)

# ----------------------------
# 融合：训练 meta-regressor（线性加权）
# ----------------------------
meta_X_train = np.column_stack([deep_train_preds, xgb_train_pred])
meta_X_val = np.column_stack([deep_val_preds, xgb_val_pred])

meta_model = LinearRegression()
meta_model.fit(meta_X_train, y_tr)

ensemble_val_pred = meta_model.predict(meta_X_val)
final_val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_pred))
print(f"\n✅ Final Ensemble Validation RMSE: {final_val_rmse:.4f}")

# ----------------------------
# 测试预测
# ----------------------------
# 深度模型预测
test_dataset = YieldDataset(X_test_full, test_field_indices)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

deep_test_preds = []
with torch.no_grad():
    for x, field_ids in test_loader:
        pred = model(x.to(device), field_ids.to(device))
        deep_test_preds.append(pred.cpu().numpy())
deep_test_preds = np.concatenate(deep_test_preds)

# XGBoost 预测
xgb_test_pred = xgb_model.predict(X_static_test)

# 融合
meta_X_test = np.column_stack([deep_test_preds, xgb_test_pred])
final_test_pred = meta_model.predict(meta_X_test)

submission = pd.DataFrame({
    "Field_ID": test_df["Field_ID"],
    "Yield": np.clip(final_test_pred, 0, None)
})
submission.to_csv("submission_hybrid_transformer_xgb.csv", index=False)
print("\n📁 Hybrid submission saved.")
