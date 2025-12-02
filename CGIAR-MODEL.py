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
# 路径配置（请根据实际路径修改）
# ----------------------------
DATA_DIR = "./data"
train_path = os.path.join(DATA_DIR, "Train.csv")
test_path = os.path.join(DATA_DIR, "test_field_ids_with_year.csv")  # 必须包含真实 Yield 列（4列）
aux_path = os.path.join(DATA_DIR, "fields_w_additional_info.csv")

# ----------------------------
# 全局参数
# ----------------------------
CLIMATE_VARS = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
SEQ_LEN = 12

# ----------------------------
# 加载数据
# ----------------------------
train_df = pd.read_csv(train_path, header=None)
train_df.columns = ["Field_ID", "Year", "Quality", "Yield"]
train_df["Yield"] = pd.to_numeric(train_df["Yield"], errors="coerce")
train_df = train_df.dropna(subset=["Yield"]).reset_index(drop=True)

test_df = pd.read_csv(test_path, header=None)
test_df.columns = ["Field_ID", "Year", "Quality", "Yield"]
test_df["Yield"] = pd.to_numeric(test_df["Yield"], errors="coerce")
test_df = test_df.dropna(subset=["Yield"]).reset_index(drop=True)

aux_df = pd.read_csv(aux_path)
aux_df.set_index("Field_ID", inplace=True)

# 所有训练中出现的 Field_ID（用于 embedding）
all_train_fields = train_df["Field_ID"].unique()
field_to_idx = {fid: i for i, fid in enumerate(all_train_fields)}
num_fields = len(all_train_fields)

# 土壤变量
soil_cols = [col for col in aux_df.columns if col.startswith("soil_")]
input_vars = CLIMATE_VARS + soil_cols
input_dim = len(input_vars)

# ----------------------------
# 构建时序特征
# ----------------------------
def build_sequence(df, aux_df, field_to_idx, num_fields):
    X_list, y_list, field_indices = [], [], []
    for _, row in df.iterrows():
        fid, year = row["Field_ID"], int(row["Year"])
        seq = np.zeros((SEQ_LEN, input_dim), dtype=np.float32)
        if fid in aux_df.index:
            aux_row = aux_df.loc[fid]
            for month in range(SEQ_LEN):
                base = f"climate_{year}_{month+1}_"
                for j, var in enumerate(input_vars):
                    col = base + var
                    if col in aux_row.index and pd.notna(aux_row[col]):
                        seq[month, j] = float(aux_row[col])
        X_list.append(seq)
        y_list.append(row["Yield"])
        # 若 Field_ID 不在训练集中，映射为 0（安全回退）
        idx = field_to_idx.get(fid, 0)
        field_indices.append(idx)
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    return X, y, np.array(field_indices)

# 清洗函数
def clean_and_clip(arr):
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -1e6, 1e6).astype(np.float32)

# 构建训练数据
X_train, y_train, train_field_idx = build_sequence(train_df, aux_df, field_to_idx, num_fields)
X_train = clean_and_clip(X_train)
y_train = clean_and_clip(y_train.reshape(-1, 1)).flatten()
y_train = np.clip(y_train, 0.0, 200.0)

# 标准化
N, T, D = X_train.shape
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train.reshape(-1, D))
X_train = X_train_flat.reshape(N, T, D)

# 构建测试数据（必须存在）
X_test, y_test, test_field_idx = build_sequence(test_df, aux_df, field_to_idx, num_fields)
X_test = clean_and_clip(X_test)
y_test = clean_and_clip(y_test.reshape(-1, 1)).flatten()
y_test = np.clip(y_test, 0.0, 200.0)
X_test = scaler.transform(X_test.reshape(-1, D)).reshape(-1, T, D)

# ----------------------------
# Group Split（按 Field_ID 划分验证集）
# ----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(gss.split(X_train, groups=train_df["Field_ID"]))

X_tr, X_val = X_train[tr_idx], X_train[val_idx]
y_tr, y_val = y_train[tr_idx], y_train[val_idx]
field_tr, field_val = train_field_idx[tr_idx], train_field_idx[val_idx]

# ----------------------------
# Dataset
# ----------------------------
class YieldDataset(Dataset):
    def __init__(self, X, fields, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.fields = torch.tensor(fields, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.fields[idx], self.y[idx]

# ----------------------------
# 模型（保留因果掩码 + 特征注意力）
# ----------------------------
class CausalTransformerYieldModel(nn.Module):
    def __init__(self, seq_len=12, input_dim=34, embed_dim=128, nhead=8, num_layers=2, dropout=0.1, num_fields=10000):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        
        self.field_embed = nn.Embedding(num_embeddings=num_fields, embedding_dim=16)
        self.field_dropout = nn.Dropout(dropout)
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        self.feat_attn = nn.Sequential(
            nn.Linear(embed_dim + 16, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.Sigmoid()
        )
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        prior = torch.tensor([1.0, 1.0, 1.5, 3.0, 3.5, 3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0])
        self.time_logit = nn.Parameter(torch.log(prior + 1e-6))
        
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x, field_ids):
        B, L, _ = x.shape
        f_emb = self.field_dropout(self.field_embed(field_ids))  # (B, 16)
        f_emb_exp = f_emb.unsqueeze(1).expand(-1, L, -1)         # (B, L, 16)
        x_emb = self.input_proj(x)                               # (B, L, E)
        attn_in = torch.cat([x_emb, f_emb_exp], dim=-1)
        weights = self.feat_attn(attn_in)
        x_weighted = x_emb * weights
        out = self.transformer(x_weighted, mask=self.causal_mask)
        time_w = torch.softmax(self.time_logit, dim=0)
        pooled = (out * time_w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return self.head(pooled).squeeze(-1)

# ----------------------------
# 训练
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CausalTransformerYieldModel(
    seq_len=SEQ_LEN,
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
test_dataset = YieldDataset(X_test, test_field_idx, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

best_val_rmse = float('inf')
for epoch in range(50):
    model.train()
    for x, fid, y in train_loader:
        x, fid, y = x.to(device), fid.to(device), y.to(device)
        pred = model(x, fid)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x, fid, y in val_loader:
            x, fid, y = x.to(device), fid.to(device), y.to(device)
            pred = model(x, fid)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(y.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    val_var = np.var(val_targets - val_preds)
    
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_model.pth")
    
    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f} | Residual Var: {val_var:.4f}")

# ----------------------------
# 最终评估：验证集 + 测试集
# ----------------------------
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# 验证集
with torch.no_grad():
    val_preds, val_targets = [], []
    for x, fid, y in val_loader:
        x, fid, y = x.to(device), fid.to(device), y.to(device)
        pred = model(x, fid)
        val_preds.append(pred.cpu().numpy())
        val_targets.append(y.cpu().numpy())
val_preds = np.concatenate(val_preds)
val_targets = np.concatenate(val_targets)
final_val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
final_val_var = np.var(val_targets - val_preds)

# 测试集
with torch.no_grad():
    test_preds, test_targets = [], []
    for x, fid, y in test_loader:
        x, fid, y = x.to(device), fid.to(device), y.to(device)
        pred = model(x, fid)
        test_preds.append(pred.cpu().numpy())
        test_targets.append(y.cpu().numpy())
test_preds = np.concatenate(test_preds)
test_targets = np.concatenate(test_targets)
final_test_rmse = np.sqrt(mean_squared_error(test_targets, test_preds))
final_test_var = np.var(test_targets - test_preds)

# ----------------------------
# 输出结果
# ----------------------------
print("\n" + "="*60)
print(f"FINAL VALIDATION RMSE: {final_val_rmse:.4f}")
print(f"FINAL VALIDATION RESIDUAL VARIANCE: {final_val_var:.4f}")
print(f"FINAL TEST RMSE: {final_test_rmse:.4f}")
print(f"FINAL TEST RESIDUAL VARIANCE: {final_test_var:.4f}")
print("="*60)

# 可选：保存预测结果
pd.DataFrame({
    "Field_ID": test_df["Field_ID"],
    "True_Yield": test_targets,
    "Pred_Yield": test_preds
}).to_csv("test_predictions.csv", index=False)
print("Test predictions saved to 'test_predictions.csv'")

