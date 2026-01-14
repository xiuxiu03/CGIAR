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

import random
import xarray as xr
from scipy.spatial.distance import cdist

# ----------------------------
# 设置随机种子
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ----------------------------
# 路径配置
# ----------------------------
DATA_DIR = "./data"
NC_FILE = "Alpine_DayMet_2008.nc"  # ←←← 修改为你实际的 .nc 文件路径
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

# 🔴 关键检查：确保 aux_df 有 latitude 和 longitude
if not ('latitude' in aux_df.columns and 'longitude' in aux_df.columns):
    raise ValueError("aux_df 必须包含 'latitude' 和 'longitude' 列！")

# ----------------------------
# 定义模型所需变量名（14个）
# ----------------------------
var_names = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]

# Daymet 变量 → 模型变量映射（注意：prcp 是降水）
daymet_to_model = {
    'tmax': 'tmmx',
    'tmin': 'tmmn',
    'prcp': 'pr',     # Daymet 中降水变量名为 'prcp'
    'srad': 'srad',
    'swe': 'swe',
    'pet': 'pet',
    'vp': 'vap'       # vapor pressure
}

# ----------------------------
# 辅助函数：从 NetCDF 构建月度气候特征
# ----------------------------
def build_features_from_netcdf(df, aux_df, ds, var_names, daymet_to_model):
    """
    从 Daymet NetCDF 构建 (N, 12, len(var_names)) 特征矩阵
    """
    # 提取时间信息（假设 ds.time 是 datetime64）
    times = pd.to_datetime(ds['time'].values)
    months = times.month.values  # shape: (365,)
    
    # 提取经纬度网格
    lats = ds['lat'].values   # (y, x)
    lons = ds['lon'].values   # (y, x)
    flat_lats = lats.ravel()
    flat_lons = lons.ravel()
    y_indices, x_indices = np.meshgrid(np.arange(lats.shape[0]), np.arange(lats.shape[1]), indexing='ij')
    flat_y = y_indices.ravel()
    flat_x = x_indices.ravel()
    
    climate_seq_list = []
    
    for _, row in df.iterrows():
        fid = row["Field_ID"]
        year = int(row["Year"])
        
        if fid not in aux_df.index:
            climate_seq_list.append(np.zeros((12, len(var_names)), dtype=np.float32))
            continue
            
        lat = aux_df.loc[fid, 'latitude']
        lon = aux_df.loc[fid, 'longitude']
        
        # 找最近格点
        coords = np.column_stack((flat_lats, flat_lons))
        target = np.array([[lat, lon]])
        distances = cdist(target, coords, metric='euclidean')[0]
        idx = np.argmin(distances)
        y_idx, x_idx = int(flat_y[idx]), int(flat_x[idx])
        
        # 初始化月度特征 (12 months × C variables)
        monthly_features = np.zeros((12, len(var_names)), dtype=np.float32)
        
        # 遍历可用的 Daymet 变量
        for d_var, m_var in daymet_to_model.items():
            if d_var not in ds.data_vars:
                continue
            try:
                # 提取该格点的时间序列 (365,)
                daily_vals = ds[d_var].isel(y=y_idx, x=x_idx).values.astype(np.float32)
                
                # 按月份聚合（使用月均值）
                for month in range(1, 13):
                    mask = (months == month)
                    if np.any(mask):
                        val = np.nanmean(daily_vals[mask])
                        if not np.isnan(val) and np.isfinite(val):
                            var_idx = var_names.index(m_var)
                            monthly_features[month-1, var_idx] = val
            except Exception as e:
                print(f"Warning: Field {fid}, variable {d_var} error: {e}")
                continue
        
        climate_seq_list.append(monthly_features)
    
    return np.stack(climate_seq_list)  # (N, 12, C)

# ----------------------------
# 清洗函数（保持不变）
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
# 加载 NetCDF 并构建特征
# ----------------------------
print("Loading NetCDF file...")
ds = xr.open_dataset(NC_FILE)

print("Building climate features from NetCDF...")
climate_train = build_features_from_netcdf(train_df, aux_df, ds, var_names, daymet_to_model)
climate_test = build_features_from_netcdf(test_df, aux_df, ds, var_names, daymet_to_model)

# 获取土壤特征（保持不变）
soil_cols = [col for col in aux_df.columns if col.startswith("soil_")]
soil_train = np.stack([aux_df.loc[row["Field_ID"], soil_cols].values.astype(np.float32)
                      if row["Field_ID"] in aux_df.index else np.zeros(len(soil_cols))
                      for _, row in train_df.iterrows()])
soil_test = np.stack([aux_df.loc[row["Field_ID"], soil_cols].values.astype(np.float32)
                     if row["Field_ID"] in aux_df.index else np.zeros(len(soil_cols))
                     for _, row in test_df.iterrows()])

# 获取标签
y_train = train_df['Yield'].values.astype(np.float32)

# ----------------------------
# 数据清洗
# ----------------------------
climate_train = clean_array_3d(climate_train)
climate_test = clean_array_3d(climate_test)
soil_train = clean_array_2d(soil_train)
soil_test = clean_array_2d(soil_test)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y_train = np.clip(y_train, 0.0, 200.0)

# ----------------------------
# 标准化（保持不变）
# ----------------------------
N, T, C = climate_train.shape
S = soil_train.shape[1]

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

# 拼接：每个时间步都包含土壤信息
soil_train_tiled = np.tile(soil_train_scaled[:, None, :], (1, T, 1))  # (N, 12, S)
soil_test_tiled = np.tile(soil_test_scaled[:, None, :], (1, T, 1))
X_train_full = np.concatenate([climate_train_scaled, soil_train_tiled], axis=-1)  # (N, 12, C+S)
X_test_full = np.concatenate([climate_test_scaled, soil_test_tiled], axis=-1)

# 划分验证集
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train, test_size=0.2, random_state=42
)

# ----------------------------
# Dataset 类（保持不变）
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
# 模型定义（保持不变）
# ----------------------------
class TimeShiftedTransformerYieldPredictor(nn.Module):
    def __init__(self, seq_len=12, input_dim=14+20, embed_dim=128, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lag_weights = nn.Parameter(torch.randn(seq_len))
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x, mask=self.causal_mask)
        weights = torch.softmax(self.lag_weights, dim=0)
        weighted_repr = (out * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return self.regressor(weighted_repr).squeeze(-1)

# ----------------------------
# 训练设置（保持不变）
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
criterion = nn.MSELoss()

train_dataset = YieldDataset(X_tr, y_tr)
val_dataset = YieldDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----------------------------
# 训练循环（保持不变）
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
        torch.save(model.state_dict(), "best_time_shifted_transformer.pth")

    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f} | Residual Variance: {val_var:.4f}")

# ----------------------------
# 最终评估与提交（保持不变）
# ----------------------------
model.load_state_dict(torch.load("best_time_shifted_transformer.pth", map_location=device))
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

lag_weights = torch.softmax(model.lag_weights, dim=0).detach().cpu().numpy()
print("\n Learned lag weights (month 1 to 12):")
for i, w in enumerate(lag_weights, 1):
    print(f"  Month {i:2d}: {w:.4f}")

test_dataset = YieldDataset(X_test_full)
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
    "Yield": np.clip(test_preds, 0, None)
})
submission.to_csv("submission_time_shifted_transformer_nc.csv", index=False)
print("\n Submission saved to submission_time_shifted_transformer_nc.csv")
