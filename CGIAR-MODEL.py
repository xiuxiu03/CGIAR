import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
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
# 定义作物生长季（4月到9月，对应索引3~8）
# ----------------------------
GROWTH_MONTHS = list(range(3, 9))  # 0-based: Apr=3, May=4, ..., Sep=8

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

# 创建 Field_ID 到 index 的映射（用于嵌入）
all_field_ids = pd.concat([train_df["Field_ID"], test_df["Field_ID"]]).unique()
field_id_to_idx = {fid: idx for idx, fid in enumerate(all_field_ids)}
num_fields = len(all_field_ids)

# ----------------------------
# 辅助函数：构建结构化气候序列 + 土壤特征（带生长季掩码）
# ----------------------------
def build_features_structured(df, aux_df, growth_months=GROWTH_MONTHS):
    soil_cols = [col for col in aux_df.columns if col.startswith("soil_")]
    climate_seq_list = []
    soil_feat_list = []
    y_list = []
    field_id_list = []  # ← 新增：记录 Field_ID
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
        field_id_list.append(fid)  # ← 记录

    climate_seqs = np.stack(climate_seq_list)
    soil_feats = np.stack(soil_feat_list)
    y = np.array(y_list, dtype=np.float32) if y_list else None
    return climate_seqs, soil_feats, y, field_id_list

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

X_train_full = np.concatenate([climate_train_scaled, soil_train_tiled], axis=-1)  # (N, 12, input_dim)
X_test_full = np.concatenate([climate_test_scaled, soil_test_tiled], axis=-1)

# 转换 Field_ID 为索引
train_field_indices = np.array([field_id_to_idx[fid] for fid in train_field_ids])
test_field_indices = np.array([field_id_to_idx[fid] for fid in test_field_ids])

# 划分验证集（注意同步划分 field indices）
indices = np.arange(len(X_train_full))
tr_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_tr, X_val = X_train_full[tr_idx], X_train_full[val_idx]
y_tr, y_val = y_train[tr_idx], y_train[val_idx]
field_tr, field_val = train_field_indices[tr_idx], train_field_indices[val_idx]

# ----------------------------
# Dataset（新增返回 field_id index）
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
# 改进模型：加入特征注意力 + 因子异质性
# ----------------------------
class TimeShiftedTransformerYieldPredictor(nn.Module):
    def __init__(self, seq_len=12, input_dim=14+20, embed_dim=128, nhead=8, num_layers=2, dropout=0.1, num_fields=10000):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Field ID 嵌入（用于异质性）
        self.field_embed = nn.Embedding(num_embeddings=num_fields, embedding_dim=32)
        
        # 特征注意力网络（接收 [input_dim + field_emb] → 输出特征权重）
        self.feature_attn_net = nn.Sequential(
            nn.Linear(input_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Softmax(dim=-1)
        )
        
        # 主干 Transformer
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # ===== 时间注意力（保留先验）=====
        prior_logits = torch.tensor([1.0, 1.0, 1.5, 3.0, 3.5, 3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0])
        self.register_buffer("prior_logits", prior_logits)
        self.lag_weights = nn.Parameter(torch.log(prior_logits + 1e-6))
        # =================================

        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x, field_ids):
        B, L, D = x.shape  # (B, 12, input_dim)
        
        # --- Step 1: 特征注意力（带地块异质性）---
        field_emb = self.field_embed(field_ids)  # (B, 32)
        field_emb_expanded = field_emb.unsqueeze(1).expand(-1, L, -1)  # (B, 12, 32)
        attn_input = torch.cat([x, field_emb_expanded], dim=-1)  # (B, 12, input_dim+32)
        feat_weights = self.feature_attn_net(attn_input)  # (B, 12, input_dim)
        x_weighted = x * feat_weights  # (B, 12, input_dim)

        # --- Step 2: Transformer 编码 ---
        x_emb = self.embedding(x_weighted)  # (B, L, E)
        out = self.transformer(x_emb, mask=self.causal_mask)  # (B, L, E)

        # --- Step 3: 时间注意力（带先验）---
        time_weights = torch.softmax(self.lag_weights, dim=0)  # (L,)
        weighted_repr = (out * time_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)  # (B, E)

        return self.regressor(weighted_repr).squeeze(-1)

    def get_kl_loss(self):
        learned_probs = torch.softmax(self.lag_weights, dim=0)
        prior_probs = torch.softmax(self.prior_logits, dim=0)
        kl = torch.sum(prior_probs * torch.log((prior_probs + 1e-8) / (learned_probs + 1e-8)))
        return kl

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
    dropout=0.1,
    num_fields=num_fields
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

train_dataset = YieldDataset(X_tr, field_tr, y_tr)
val_dataset = YieldDataset(X_val, field_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----------------------------
# 训练循环
# ----------------------------
best_val_rmse = float('inf')
LAMBDA_KL = 0.1

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

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x, field_ids, y in val_loader:
            x, field_ids, y = x.to(device), field_ids.to(device), y.to(device)
            pred = model(x, field_ids)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(y.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    val_var = np.var(val_targets - val_preds)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_dacm_like_model.pth")

    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f} | Residual Variance: {val_var:.4f}")

# ----------------------------
# 最终评估
# ----------------------------
model.load_state_dict(torch.load("best_dacm_like_model.pth", map_location=device))
model.eval()
with torch.no_grad():
    val_preds = []
    for x, field_ids, _ in val_loader:
        x, field_ids = x.to(device), field_ids.to(device)
        pred = model(x, field_ids)
        val_preds.append(pred.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    final_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    final_var = np.var(y_val - val_preds)

print("\n Final Validation Metrics:")
print(f"RMSE: {final_rmse:.4f}")
print(f"Residual Variance: {final_var:.4f}")

# 打印时间注意力权重
lag_weights = torch.softmax(model.lag_weights, dim=0).detach().cpu().numpy()
print("\n Learned lag weights (time attention, month 1 to 12):")
for i, w in enumerate(lag_weights, 1):
    print(f"  Month {i:2d}: {w:.4f}")

# ----------------------------
# 测试预测 & 提交
# ----------------------------
test_dataset = YieldDataset(X_test_full, test_field_indices)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    test_preds = []
    for x, field_ids in test_loader:
        x, field_ids = x.to(device), field_ids.to(device)
        pred = model(x, field_ids)
        test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds)

submission = pd.DataFrame({
    "Field_ID": test_df["Field_ID"],
    "Yield": np.clip(test_preds, 0, None)
})
submission.to_csv("submission_dacm_like_with_prior.csv", index=False)
print("\n Submission saved to submission_dacm_like_with_prior.csv")
