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
# 辅助函数：提取气候特征（增强鲁棒性）
# ----------------------------
def extract_climate_features(field_id, year, aux_row):
    features = []
    for month in range(1, 13):
        base = f"climate_{year}_{month}_"
        # 所有以该前缀开头的气候变量（共14种）
        for var in ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]:
            col = f"{base}{var}"
            if col in aux_row.index:
                val = aux_row[col]
                # 处理非数值情况
                if isinstance(val, (int, float)) and not pd.isna(val):
                    features.append(float(val))
                else:
                    features.append(np.nan)
            else:
                features.append(np.nan)
    return np.array(features, dtype=np.float32)

# ----------------------------
# 构建特征
# ----------------------------
def build_features(df, aux_df):
    soil_cols = [col for col in aux_df.columns if col.startswith("soil_")]
    X_list = []
    y_list = []

    for _, row in df.iterrows():
        fid = row["Field_ID"]
        year = int(row["Year"])
        
        if fid in aux_df.index:
            aux_row = aux_df.loc[fid]
            soil_feat = aux_row[soil_cols].values.astype(np.float32)
            climate_feat = extract_climate_features(fid, year, aux_row)
        else:
            soil_feat = np.full(len(soil_cols), np.nan, dtype=np.float32)
            climate_feat = np.full(12 * 14, np.nan, dtype=np.float32)  # 12 months × 14 vars
        
        full_feat = np.concatenate([soil_feat, climate_feat])
        X_list.append(full_feat)
        
        if "Yield" in row:
            y_list.append(row["Yield"])
    
    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32) if y_list else None
    return X, y

X_train, y_train = build_features(train_df, aux_df)
X_test, _ = build_features(test_df, aux_df)

def clean_array(X):
    X = np.where(np.isinf(X), np.nan, X)          # inf → NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)  # NaN/inf → 0
    X = np.clip(X, -1e6, 1e6)                     # 防止过大值
    return X.astype(np.float32)

X_train = clean_array(X_train)
X_test = clean_array(X_test)

# 清洗目标变量
y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y_train = np.clip(y_train, 0.0, 200.0)  # 合理产量范围（单位依数据而定）

# ----------------------------
# 标准化 & 划分验证集
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42
)

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
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# ----------------------------
# Transformer 模型
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
        x = self.embedding(x).unsqueeze(1)  # (B, 1, E)
        out = self.transformer(x)           # (B, 1, E)
        out = out.squeeze(1)                # (B, E)
        return self.regressor(out).squeeze(-1)

# ----------------------------
# 训练设置
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
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
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
    val_var = np.var(val_targets - val_preds)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_transformer.pth")

    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f} | Residual Variance: {val_var:.4f}")

# ----------------------------
# 最终评估
# ----------------------------
model.load_state_dict(torch.load("best_transformer.pth", map_location=device))
model.eval()

with torch.no_grad():
    val_preds = []
    for x, _ in val_loader:
        x = x.to(device)
        pred = model(x)
        val_preds.append(pred.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    final_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    final_var = np.var(y_val - val_preds)

print("\n Final Validation Metrics:")
print(f"RMSE: {final_rmse:.4f}")
print(f"Residual Variance: {final_var:.4f}")

# ----------------------------
# 测试预测 & 提交
# ----------------------------
test_dataset = YieldDataset(X_test_scaled)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    test_preds = []
    for x in test_loader:
        x = x.to(device)
        pred = model(x)
        test_preds.append(pred.cpu().numpy())
    test_preds = np.concatenate(test_preds)

submission = pd.DataFrame({
    "Field_ID": test_df["Field_ID"],
    "Yield": np.clip(test_preds, 0, None)  # 禁止负产量
})
submission.to_csv("submission_transformer.csv", index=False)
print("\n Submission saved to submission_transformer.csv")

