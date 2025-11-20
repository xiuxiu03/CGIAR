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
# 路径配置（假设所有 CSV 在当前目录）
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
# 构建气候特征：根据 Year 动态选择对应年份的12个月气候变量
# ----------------------------
def extract_climate_features(field_id, year, aux_row):
    # 构造列名前缀，例如 climate_2019_1_pr, ..., climate_2019_12_vs
    climate_cols = []
    for month in range(1, 13):
        base = f"climate_{year}_{month}_"
        month_cols = [col for col in aux_row.index if col.startswith(base)]
        climate_cols.extend(month_cols)
    # 如果某些年份缺失（如2018在训练集中不存在），用0填充
    features = []
    for col in climate_cols:
        features.append(aux_row[col] if col in aux_row else 0.0)
    return np.array(features, dtype=np.float32)


# 提取每个样本的完整辅助特征（土壤 + 对应年份气候）
def build_features(df, aux_df):
    soil_cols = [col for col in aux_df.columns if col.startswith("soil_")]
    X_list = []
    y_list = []

    for _, row in df.iterrows():
        fid = row["Field_ID"]
        year = int(row["Year"])
        if fid not in aux_df.index:
            # 若无辅助信息，用零向量
            soil_feat = np.zeros(len(soil_cols))
            climate_feat = np.zeros(12 * 14)  # 12个月 × 14气候变量（aet, def, pdsi... vs）
        else:
            aux_row = aux_df.loc[fid]
            soil_feat = aux_row[soil_cols].values.astype(np.float32)
            climate_feat = extract_climate_features(fid, year, aux_row)
        full_feat = np.concatenate([soil_feat, climate_feat])
        X_list.append(full_feat)
        if "Yield" in row:
            y_list.append(row["Yield"])

    X = np.stack(X_list)
    y = np.array(y_list) if y_list else None
    return X, y


X_train, y_train = build_features(train_df, aux_df)
X_test, _ = build_features(test_df, aux_df)

# ----------------------------
# 标准化
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 划分训练/验证集
# ----------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)


# ----------------------------
# Dataset 类
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
# Transformer 回归模型
# ----------------------------
class TransformerYieldPredictor(nn.Module):
    def __init__(self, input_dim, embed_dim=128, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, D)
        x = self.embedding(x).unsqueeze(1)  # (B, 1, E) — treat as sequence of length 1
        out = self.transformer(x)  # (B, 1, E)
        out = out.squeeze(1)  # (B, E)
        return self.regressor(out).squeeze(-1)


# ----------------------------
# 训练配置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_tr.shape[1]
model = TransformerYieldPredictor(input_dim=input_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

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
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

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
    val_var = np.var(val_targets - val_preds)  # 残差方差

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_transformer.pth")

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Val RMSE: {val_rmse:.4f} | Residual Variance: {val_var:.4f}")

# ----------------------------
# 最终评估 & 测试预测
# ----------------------------
model.load_state_dict(torch.load("best_transformer.pth"))
model.eval()

# 验证集最终指标
with torch.no_grad():
    val_preds = []
    for x, _ in val_loader:
        x = x.to(device)
        pred = model(x)
        val_preds.append(pred.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    final_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    final_var = np.var(y_val - val_preds)

print("\n✅ Final Validation Metrics:")
print(f"RMSE: {final_rmse:.4f}")
print(f"Residual Variance: {final_var:.4f}")

# 测试集预测
test_dataset = YieldDataset(X_test_scaled)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
with torch.no_grad():
    test_preds = []
    for x in test_loader:
        x = x.to(device)
        pred = model(x)
        test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds)

# 保存提交文件
submission = pd.DataFrame({
    "Field_ID": test_df["Field_ID"],
    "Yield": np.clip(test_preds, 0, None)
})
submission.to_csv("submission_transformer.csv", index=False)

print("\n✅ Submission saved to submission_transformer.csv")
