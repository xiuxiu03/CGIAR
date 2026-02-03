import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# 0. 阿里云 DashScope + 嵌入工具
# ----------------------------
try:
    import dashscope
    from dashscope import TextEmbedding
    import tiktoken
except ImportError:
    raise ImportError("请安装依赖: pip install dashscope tiktoken")

dashscope.api_key = "sk-65aa3b4c924b43e29bbffe9430eeb010"  # ← 替换为你的 Key

def get_single_embedding(text: str, model: str = "text-embedding-v2") -> np.ndarray:
    """获取单句嵌入（无需分块）"""
    response = TextEmbedding.call(model=model, input=text.strip())
    if response.status_code != 200:
        raise RuntimeError(f"Embedding failed: {response}")
    emb = np.array(response.output["embeddings"][0]["embedding"], dtype=np.float32)
    return emb

# ----------------------------
# 1. 变量级嵌入生成与聚类验证
# ----------------------------
variable_descriptions = {
    "soil_bulk_density": "土壤容重，反映土壤紧实程度，影响根系生长和水分渗透。",
    "soil_cec": "土壤阳离子交换量，表征土壤保肥能力，数值越高，养分保持能力越强。",
    "soil_coarse_fragments": "土壤中粗碎屑（如砾石）的含量，影响土壤持水性和耕作性能。",
    "soil_clay": "黏粒含量，决定土壤的保水性、通气性和结构稳定性。",
    "soil_nitrogen": "土壤全氮含量，是衡量土壤肥力的重要指标之一。",
    "soil_organic_carbon_density": "单位体积土壤中有机碳的质量，用于评估碳储存能力。",
    "soil_organic_carbon_stock": "单位面积土壤剖面中储存的有机碳总量，常用于碳汇核算。",
    "soil_ph": "土壤酸碱度，影响养分有效性及微生物活性。",
    "soil_sand": "砂粒含量，砂质高的土壤排水快但保肥能力弱。",
    "soil_silt": "粉粒含量，介于砂与黏粒之间，影响土壤质地和保水性。",
    "soil_organic_carbon": "土壤有机碳含量，反映土壤有机质水平和健康状况。",
    "aet": "实际蒸散发，表示地表水分通过蒸发和植物蒸腾返回大气的总量。",
    "def": "水分亏缺，即潜在蒸散发与实际供水之间的差额，反映干旱胁迫程度。",
    "pdsi": "帕尔默干旱指数，综合降水与蒸散发评估长期干旱状况。",
    "pet": "潜在蒸散发，在水分充足条件下可能发生的最大蒸散量。",
    "pr": "降水量，指一定时期内降落到地面的液态或固态水总量。",
    "ro": "地表径流，降水未入渗而沿地表流动的部分，影响水资源与侵蚀。",
    "soil_moisture": "土壤湿度，表征土壤中含水量，直接影响作物生长和水文过程。",
    "srad": "地表太阳辐射，驱动光合作用、蒸发和地表能量平衡。",
    "swe": "雪水当量，指积雪融化后对应的水深，是冬季水资源的重要指标。",
    "tmmn": "月平均最低气温，反映夜间或冷季低温状况。",
    "tmmx": "月平均最高气温，反映白天或暖季高温状况。",
    "vap": "水汽压，表示空气中水汽的分压力，与湿度密切相关。",
    "vpd": "饱和水汽压差，表征大气干燥程度，影响植物蒸腾和水分胁迫。",
    "vs": "风速，影响蒸发、传热、花粉传播及风蚀过程。"
}

print("🔍 正在为每个变量生成嵌入...")
var_embeddings = {}
for var, desc in variable_descriptions.items():
    full_text = f"{var}: {desc}"
    emb = get_single_embedding(full_text)
    var_embeddings[var] = emb

var_names = list(var_embeddings.keys())
X_var = np.stack([var_embeddings[name] for name in var_names])  # (25, 1536)

# 聚类验证
print("\n📊 正在进行聚类分析...")
X_scaled = StandardScaler().fit_transform(X_var)
n_clusters = 4
cluster_labels = AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X_scaled)

# TSNE 可视化
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(var_names)-1))
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='tab10', s=120, edgecolor='k')
plt.colorbar(scatter)
for i, name in enumerate(var_names):
    plt.text(X_tsne[i, 0] + 0.8, X_tsne[i, 1] + 0.8, name, fontsize=9)
plt.title("变量嵌入聚类 (TSNE + 层次聚类)", fontsize=14)
plt.tight_layout()
plt.savefig("variable_embedding_clusters.png", dpi=150)
plt.close()

# 打印聚类结果
print("\n✅ 聚类结果:")
for cid in range(n_clusters):
    members = [var_names[i] for i, label in enumerate(cluster_labels) if label == cid]
    print(f"  Cluster {cid}: {', '.join(members)}")

# ----------------------------
# 2. 构建全局 prompt 嵌入（仅语义，无数据）→ 直接使用 1536 维
# ----------------------------
prompt_text = """
你是一个气候与土壤数据分析助手。以下是东非地区预测玉米产量时常用的土壤属性与气象指标说明：
""" + "\n".join([f"{k}：{v}" for k, v in variable_descriptions.items()])

global_embedding_full = get_single_embedding(prompt_text)  # shape: (1536,)

# ✅ 修复：不再使用 PCA（单样本无法降维）
global_embedding = global_embedding_full  # 直接使用原始嵌入
print(f"\n✅ 使用原始全局嵌入，维度: {global_embedding.shape[0]}")

# ----------------------------
# 3. 数据加载与预处理（支持消融开关）
# ----------------------------
USE_GLOBAL_CONTEXT = True  # ← 设置为 False 即可关闭嵌入（消融实验）

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

DATA_DIR = "./data"
train_file = os.path.join(DATA_DIR, "Train.csv")
test_file = os.path.join(DATA_DIR, "test_field_ids_with_year.csv")
aux_file = os.path.join(DATA_DIR, "fields_w_additional_info.csv")

GROWTH_MONTHS = list(range(3, 9))  # Apr=3 to Sep=8 (0-based index)

train_df = pd.read_csv(train_file, header=None)
train_df.columns = ["Field_ID", "Year", "Quality", "Yield"]
train_df['Yield'] = pd.to_numeric(train_df['Yield'], errors='coerce')
train_df = train_df.dropna(subset=['Yield']).reset_index(drop=True)
test_df = pd.read_csv(test_file)
aux_df = pd.read_csv(aux_file)
aux_df.set_index("Field_ID", inplace=True)

def build_features(df, aux_df, global_embedding=None, use_global=True, growth_months=GROWTH_MONTHS):
    soil_cols = [col for col in aux_df.columns if col.startswith("soil_")]
    var_names = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
    
    climate_seq_list = []
    soil_feat_list = []
    y_list = []

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

    climate_seqs = np.stack(climate_seq_list)
    soil_feats = np.stack(soil_feat_list)
    y = np.array(y_list, dtype=np.float32) if y_list else None

    N = climate_seqs.shape[0]
    if use_global and global_embedding is not None:
        X_global = np.tile(global_embedding, (N, 1)).astype(np.float32)
    else:
        X_global = np.zeros((N, 1), dtype=np.float32)  # 消融：无嵌入

    return climate_seqs, soil_feats, X_global, y

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

# 构建特征
if USE_GLOBAL_CONTEXT:
    climate_train, soil_train, X_global_train, y_train = build_features(
        train_df, aux_df, global_embedding, use_global=True
    )
    climate_test, soil_test, X_global_test, _ = build_features(
        test_df, aux_df, global_embedding, use_global=True
    )
else:
    climate_train, soil_train, X_global_train, y_train = build_features(
        train_df, aux_df, use_global=False
    )
    climate_test, soil_test, X_global_test, _ = build_features(
        test_df, aux_df, use_global=False
    )

# 清洗 & 标准化
climate_train = clean_array_3d(climate_train)
climate_test = clean_array_3d(climate_test)
soil_train = clean_array_2d(soil_train)
soil_test = clean_array_2d(soil_test)
y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y_train = np.clip(y_train, 0.0, 200.0)

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
X_train_seq = np.concatenate([climate_train_scaled, soil_train_tiled], axis=-1)
X_test_seq = np.concatenate([climate_test_scaled, soil_test_tiled], axis=-1)

indices = np.arange(len(X_train_seq))
tr_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
X_seq_tr, X_seq_val = X_train_seq[tr_idx], X_train_seq[val_idx]
X_global_tr, X_global_val = X_global_train[tr_idx], X_global_train[val_idx]
y_tr, y_val = y_train[tr_idx], y_train[val_idx]

# ----------------------------
# 4. 模型定义
# ----------------------------
class YieldDataset(Dataset):
    def __init__(self, X_seq, X_global, y=None):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_global = torch.tensor(X_global, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X_seq)
    def __getitem__(self, i):
        if self.y is not None:
            return (self.X_seq[i], self.X_global[i]), self.y[i]
        return (self.X_seq[i], self.X_global[i])

class TimeShiftedTransformerWithGlobal(nn.Module):
    def __init__(self, seq_len=12, input_dim=34, embed_dim=128, global_dim=1536, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_proj = nn.Linear(global_dim, embed_dim)
        self.dropout_global = nn.Dropout(dropout)  # ← 新增：防止过拟合
        self.lag_weights = nn.Parameter(torch.randn(seq_len))
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x_seq, x_global):
        x = self.embedding(x_seq)
        out = self.transformer(x, mask=self.causal_mask)
        weights = torch.softmax(self.lag_weights, dim=0)
        seq_repr = (out * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        global_repr = self.global_proj(x_global)
        global_repr = self.dropout_global(global_repr)  # ← 应用 dropout
        fused = torch.cat([seq_repr, global_repr], dim=-1)
        return self.regressor(fused).squeeze(-1)

# 动态设置 global_dim
global_dim = X_global_tr.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeShiftedTransformerWithGlobal(
    seq_len=12,
    input_dim=X_seq_tr.shape[-1],
    embed_dim=128,
    global_dim=global_dim,
    nhead=8,
    num_layers=2,
    dropout=0.1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

train_dataset = YieldDataset(X_seq_tr, X_global_tr, y_tr)
val_dataset = YieldDataset(X_seq_val, X_global_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----------------------------
# 5. 训练
# ----------------------------
best_val_rmse = float('inf')
for epoch in range(50):
    model.train()
    for (x_seq, x_global), y in train_loader:
        x_seq, x_global, y = x_seq.to(device), x_global.to(device), y.to(device)
        pred = model(x_seq, x_global)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for (x_seq, x_global), y in val_loader:
            x_seq, x_global, y = x_seq.to(device), x_global.to(device), y.to(device)
            pred = model(x_seq, x_global)
            val_preds.append(pred.cpu().numpy())
            val_targets.append(y.cpu().numpy())

    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    val_var = np.var(val_targets - val_preds)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        suffix = "_with_context" if USE_GLOBAL_CONTEXT else "_no_context"
        torch.save(model.state_dict(), f"best_model{suffix}.pth")

    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f} | Residual Var: {val_var:.4f}")

# 加载最佳模型
suffix = "_with_context" if USE_GLOBAL_CONTEXT else "_no_context"
model.load_state_dict(torch.load(f"best_model{suffix}.pth", map_location=device))
model.eval()
with torch.no_grad():
    val_preds = []
    for (x_seq, x_global), _ in val_loader:
        x_seq, x_global = x_seq.to(device), x_global.to(device)
        pred = model(x_seq, x_global)
        val_preds.append(pred.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    final_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    final_var = np.var(y_val - val_preds)

print(f"\n{'🟢 WITH CONTEXT' if USE_GLOBAL_CONTEXT else '🔴 WITHOUT CONTEXT'} Final Validation Metrics:")
print(f"RMSE: {final_rmse:.4f}")
print(f"Residual Variance: {final_var:.4f}")

# 输出滞后权重
lag_weights = torch.softmax(model.lag_weights, dim=0).detach().cpu().numpy()
print("\n Learned lag weights (month 1 to 12):")
for i, w in enumerate(lag_weights, 1):
    marker = " ← GROWTH SEASON" if 4 <= i <= 9 else ""
    print(f"  Month {i:2d}: {w:.4f}{marker}")

# ----------------------------
# 6. 生成提交文件
# ----------------------------
test_dataset = YieldDataset(X_test_seq, X_global_test)
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
output_file = f"submission{suffix}.csv"
submission.to_csv(output_file, index=False)
print(f"\n✅ Submission saved to {output_file}")
