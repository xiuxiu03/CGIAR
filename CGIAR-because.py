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
# 路径配置（请根据实际情况修改）
# ----------------------------
DATA_DIR = "./data"  # 确保此目录下有以下三个文件

train_file = os.path.join(DATA_DIR, "Train.csv")
test_file = os.path.join(DATA_DIR, "test_field_ids_with_year.csv")
aux_file = os.path.join(DATA_DIR, "fields_w_additional_info.csv")

# ----------------------------
# 加载数据
# ----------------------------
# 训练集：无 header，需手动指定列名
train_df = pd.read_csv(train_file, header=None)
train_df.columns = ["Field_ID", "Year", "Quality", "Yield"]
train_df['Yield'] = pd.to_numeric(train_df['Yield'], errors='coerce')
train_df = train_df.dropna(subset=['Yield']).reset_index(drop=True)

# 测试集：有 Field_ID 和 Year
test_df = pd.read_csv(test_file)

# 辅助特征（气候 + 土壤）
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
        
        # 提取土壤特征
        soil_vals = aux_row[soil_cols].values.astype(np.float32)
        if np.all(~np.isfinite(soil_vals)):
            continue
        
        # 提取气候序列（12个月 × 14变量）
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
        raise ValueError("No valid samples after filtering — check data alignment!")
    
    climate_arr = np.stack(climate_list)
    soil_arr = np.stack(soil_list)
    y_arr = np.array(y_list, dtype=np.float32) if is_train else None
    
    # 安全清洗：inf → nan → 0，裁剪极端值
    climate_arr = np.where(np.isinf(climate_arr), np.nan, climate_arr)
    soil_arr = np.where(np.isinf(soil_arr), np.nan, soil_arr)
    
    climate_arr = np.nan_to_num(climate_arr, nan=0.0, posinf=0.0, neginf=0.0)
    soil_arr = np.nan_to_num(soil_arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    climate_arr = np.clip(climate_arr, -1e6, 1e6).astype(np.float32)
    soil_arr = np.clip(soil_arr, -1e6, 1e6).astype(np.float32)
    
    return climate_arr, soil_arr, y_arr, valid_indices

# ----------------------------
# 构建训练 & 测试特征
# ----------------------------
climate_train, soil_train, y_train, train_valid_idx = build_and_filter_features(train_df, aux_df, is_train=True)
climate_test, soil_test, _, _ = build_and_filter_features(test_df, aux_df, is_train=False)

print(f"✅ Train samples after filtering: {len(climate_train)}")
print(f"✅ Test samples after filtering: {len(climate_test)}")

# 更新训练 DataFrame（用于按 Field_ID 分组）
train_df_filtered = train_df.iloc[train_valid_idx].reset_index(drop=True)

# ----------------------------
# 标准化 & 拼接特征
# ----------------------------
N, T, C = climate_train.shape  # T=12, C=14
S = soil_train.shape[1]        # 土壤特征数（如 20）

# 气候标准化
climate_train_flat = climate_train.reshape(-1, C)
climate_scaler = StandardScaler()
climate_train_scaled_flat = climate_scaler.fit_transform(climate_train_flat)
climate_train_scaled = climate_train_scaled_flat.reshape(N, T, C)

climate_test_flat = climate_test.reshape(-1, C)
climate_test_scaled = climate_scaler.transform(climate_test_flat).reshape(-1, T, C)

# 土壤标准化
soil_scaler = StandardScaler()
soil_train_scaled = soil_scaler.fit_transform(soil_train)
soil_test_scaled = soil_scaler.transform(soil_test)

# 将土壤特征复制到每个时间步
soil_train_tiled = np.tile(soil_train_scaled[:, None, :], (1, T, 1))
soil_test_tiled = np.tile(soil_test_scaled[:, None, :], (1, T, 1))

# 最终输入：[climate (14) + soil (S)]
X_train_full = np.concatenate([climate_train_scaled, soil_train_tiled], axis=-1)
X_test_full = np.concatenate([climate_test_scaled, soil_test_tiled], axis=-1)

y_train = np.clip(y_train, 0.0, 200.0)  # 合理裁剪产量范围

# ----------------------------
# 按 Field_ID 划分（防数据泄露）
# ----------------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X_train_full, y_train, groups=train_df_filtered["Field_ID"]))

X_tr, X_val = X_train_full[train_idx], X_train_full[val_idx]
y_tr, y_val = y_train[train_idx], y_train[val_idx]

print(f"📊 Final splits → Train: {len(X_tr)}, Val: {len(X_val)}")

# ----------------------------
# PyTorch Dataset
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
# 模型定义
# ----------------------------
class TimeShiftedTransformerYieldPredictor(nn.Module):
    def __init__(self, seq_len=12, input_dim=34, embed_dim=128, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.month_embedding = nn.Embedding(seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lag_logits = nn.Parameter(torch.randn(seq_len))  # 可学习的时间权重
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        # 因果掩码（仅允许看过去）
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
# 农业先验分布（4–9月为生长季）
# ----------------------------
def get_agricultural_prior(seq_len=12, growing_season=(3, 9)):
    prior = np.zeros(seq_len)
    prior[growing_season[0]:growing_season[1]] = 1.0
    prior = prior / prior.sum()  # 归一化为概率分布
    return torch.tensor(prior, dtype=torch.float32)

# ----------------------------
# 训练设置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_tr.shape[-1]  # 应为 14 + S（如 34）

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
lambda_kl = 0.1  # KL 正则强度（可调）

train_dataset = YieldDataset(X_tr, y_tr)
val_dataset = YieldDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----------------------------
# 训练循环
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

    # 验证
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
# 生成提交文件
# ----------------------------
model.load_state_dict(torch.load("best_model_kl_regularized.pth", map_location=device))
model.eval()

with torch.no_grad():
    # 验证集最终 RMSE
    val_preds = []
    for x, _ in val_loader:
        x = x.to(device)
        pred = model(x)
        val_preds.append(pred.cpu().numpy())
    final_rmse = np.sqrt(mean_squared_error(y_val, np.concatenate(val_preds)))
    print(f"\n🎯 Final Validation RMSE: {final_rmse:.4f}")

    # 打印学习到的时间权重
    lag_weights = model.get_lag_weights().detach().cpu().numpy()
    print("\n📅 Learned lag weights (Month 1 to 12):")
    for i, w in enumerate(lag_weights, 1):
        print(f"  Month {i:2d}: {w:.4f}")

    # 预测测试集
    test_dataset = YieldDataset(X_test_full)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_preds = []
    for x in test_loader:
        x = x.to(device)
        pred = model(x)
        test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds)

# 保存提交文件
submission = pd.DataFrame({
    "Field_ID": test_df["Field_ID"],
    "Yield": np.clip(test_preds, 0, None)  # 产量不能为负
})
submission.to_csv("submission_kl_regularized.csv", index=False)
print(f"\n✅ Submission saved to 'submission_kl_regularized.csv'")
