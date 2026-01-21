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
# 1. 加载数据
# ----------------------------
yield_file = "Amador_yield_2018.csv"
climate_file = "daymet_2022_monthly_valid_only.csv"

yield_df = pd.read_csv(yield_file)
climate_df = pd.read_csv(climate_file)

# 清理列名
yield_df.columns = yield_df.columns.str.strip()
yield_df['key_crop_name'] = yield_df['key_crop_name'].str.strip()

# 确保 climate_df 有 'year' 列
assert 'year' in climate_df.columns, "Climate file must contain 'year' column!"
assert 'month' in climate_df.columns, "Climate file must contain 'month' column!"

print(f"Yield data years: {sorted(yield_df['year'].unique())}")
print(f"Climate data years: {sorted(climate_df['year'].unique())}")

# ----------------------------
# 2. 聚合气候数据：按 year-month 计算全县平均
# ----------------------------
climate_vars = ['tmax', 'tmin', 'prcp', 'srad', 'vp', 'pet', 'swe', 'dayl']
monthly_avg = climate_df.groupby(['year', 'month'])[climate_vars].mean().reset_index()

def get_yearly_climate_features(year, monthly_df):
    """提取某年12个月的气候特征 (96-dim)"""
    year_data = monthly_df[monthly_df['year'] == year].sort_values('month')
    if len(year_data) != 12:
        return None
    return year_data[climate_vars].values.flatten()  # (12, 8) -> (96,)

# ----------------------------
# 3. 构建样本：每个 (year, crop) 作为一个样本
# ----------------------------
X, y, meta = [], [], []

for _, row in yield_df.iterrows():
    year = int(row['year'])
    crop = row['key_crop_name']
    yield_val = float(row['yield'])
    
    feat = get_yearly_climate_features(year, monthly_avg)
    if feat is not None and not np.isnan(yield_val):
        X.append(feat)
        y.append(yield_val)
        meta.append({'year': year, 'crop': crop})

X = np.array(X, dtype=np.float32)  # (N, 96)
y = np.array(y, dtype=np.float32)  # (N,)
meta_df = pd.DataFrame(meta)

print(f"\nTotal samples: {len(X)}")
print("Sample breakdown:")
print(meta_df.groupby(['crop', 'year']).size())

# ----------------------------
# 4. 数据清洗与标准化
# ----------------------------
# 清理异常值
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X = np.clip(X, -1e5, 1e5)
y = np.clip(y, 0.0, 200.0)

# 划分训练/验证集（由于样本少，使用留一法或小验证集）
if len(X) > 5:
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    # 如果 <=5 个样本，全部用于训练（仅演示）
    X_tr, X_val, y_tr, y_val = X, X, y, y

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

# ----------------------------
# 5. Dataset 类
# ----------------------------
class YieldDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# ----------------------------
# 6. 修改后的 Transformer 模型（适配 96-dim 输入）
# ----------------------------
class TransformerYieldPredictor(nn.Module):
    def __init__(self, input_dim, embed_dim=64, nhead=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # (B, 1, E)
        out = self.transformer(x)           # (B, 1, E)
        out = out.squeeze(1)                # (B, E)
        return self.regressor(out).squeeze(-1)

# ----------------------------
# 7. 训练设置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_tr_scaled.shape[1]
model = TransformerYieldPredictor(input_dim=input_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

train_dataset = YieldDataset(X_tr_scaled, y_tr)
val_dataset = YieldDataset(X_val_scaled, y_val)

# 小样本：batch_size = 全部
batch_size = min(8, len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ----------------------------
# 8. 训练循环
# ----------------------------
best_val_rmse = float('inf')
num_epochs = 100 if len(X) < 10 else 50

for epoch in range(num_epochs):
    model.train()
    for x, y_batch in train_loader:
        x, y_batch = x.to(device), y_batch.to(device)
        pred = model(x)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x, y_batch in val_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            pred = model(x)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(y_batch.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_transformer.pth")

    if epoch % 20 == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch+1:3d} | Val RMSE: {val_rmse:.4f}")

# ----------------------------
# 9. 最终评估
# ----------------------------
model.load_state_dict(torch.load("best_transformer.pth", map_location=device))
model.eval()
with torch.no_grad():
    final_preds = []
    for x, _ in val_loader:
        x = x.to(device)
        pred = model(x)
        final_preds.append(pred.cpu().numpy())
    final_preds = np.concatenate(final_preds)
    final_rmse = np.sqrt(mean_squared_error(y_val, final_preds))

print("\n" + "="*40)
print("Final Validation Metrics:")
print(f"RMSE: {final_rmse:.4f}")
print(f"Samples used: {len(y_val)}")

# ----------------------------
# 10. （可选）生成“测试”预测（这里用训练数据演示）
# ----------------------------
# 假设我们要预测所有年份的 yield（实际应用中可替换为未来年份气候）
test_X_scaled = scaler.transform(X)  # 使用全部数据作为“测试”
test_dataset = YieldDataset(test_X_scaled)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

with torch.no_grad():
    test_preds = []
    for x in test_loader:
        x = x.to(device)
        pred = model(x)
        test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds)

# 创建提交文件（模拟）
submission = pd.DataFrame({
    "year": meta_df["year"],
    "crop": meta_df["crop"],
    "predicted_yield": np.clip(test_preds, 0, None)
})
submission.to_csv("submission_transformer_county.csv", index=False)
print("\nSubmission saved to submission_transformer_county.csv")
print(submission.head())
