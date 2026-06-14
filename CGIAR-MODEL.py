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
except ImportError:
    raise ImportError("请安装依赖: pip install dashscope")

dashscope.api_key = "sk-ws-H.RELLDYE.4yZ0.MEUCIQCwwp0eNnXiESR1SiSohy4xGi5JSiV1dtga7XXrXTTdEwIgG6DbRUEK8-W-sNQlfMl9BC9T05LQqwhe_O4U3AA9sEE"  # ← 替换为你的 Key

def get_single_embedding(text: str, model: str = "text-embedding-v2") -> np.ndarray:
    response = TextEmbedding.call(model=model, input=text.strip())
    if response.status_code != 200:
        raise RuntimeError(f"Embedding failed: {response}")
    emb = np.array(response.output["embeddings"][0]["embedding"], dtype=np.float32)
    return emb

# ----------------------------
# 1. 变量描述与预计算变量级嵌入
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

print("🔍 预计算变量级嵌入...")
var_embeddings = {}
for var, desc in variable_descriptions.items():
    full_text = f"{var}: {desc}"
    emb = get_single_embedding(full_text)
    var_embeddings[var] = emb
print(f"✅ 已生成 {len(var_embeddings)} 个变量嵌入")

# ----------------------------
# 2. 数据加载
# ----------------------------
DATA_DIR = "./data"
train_file = os.path.join(DATA_DIR, "Train.csv")
test_file = os.path.join(DATA_DIR, "test_field_ids_with_year.csv")
aux_file = os.path.join(DATA_DIR, "fields_w_additional_info.csv")

GROWTH_MONTHS = list(range(3, 9))  # Apr=3 to Sep=8 (0-based)

train_df = pd.read_csv(train_file, header=None)
train_df.columns = ["Field_ID", "Year", "Quality", "Yield"]
train_df['Yield'] = pd.to_numeric(train_df['Yield'], errors='coerce')
train_df = train_df.dropna(subset=['Yield']).reset_index(drop=True)
test_df = pd.read_csv(test_file)
aux_df = pd.read_csv(aux_file)
aux_df.set_index("Field_ID", inplace=True)

CLIMATE_VARS = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
SOIL_COLS = [col for col in aux_df.columns if col.startswith("soil_")]

# ----------------------------
# 3. 为每个训练样本生成语义 embedding（核心新增）
# ----------------------------
def compute_sample_embedding(climate_seq, soil_feat, climate_vars, soil_cols, var_emb_dict):
    """
    基于样本实际值，加权聚合变量嵌入，生成样本级语义向量。
    """
    climate_mean = np.nanmean(climate_seq[GROWTH_MONTHS], axis=0)  # (14,)
    all_vals = np.concatenate([climate_mean, soil_feat])  # (14 + S,)
    all_names = climate_vars + soil_cols

    embeds, weights = [], []
    for val, name in zip(all_vals, all_names):
        if np.isnan(val) or np.isinf(val):
            continue
        # 映射名称到 var_embeddings 的 key
        key = name if name in var_emb_dict else None
        if key is None and name.startswith("soil_"):
            key = name  # 土壤变量名直接匹配
        if key in var_emb_dict:
            embeds.append(var_emb_dict[key])
            weights.append(abs(val) + 1e-6)  # 避免除零

    if not embeds:
        return np.zeros_like(next(iter(var_emb_dict.values())))
    
    embeds = np.stack(embeds)
    weights = np.array(weights)
    weights = weights / weights.sum()
    return (weights[:, None] * embeds).sum(axis=0)

print("\n🧬 正在为每个训练样本生成语义 embedding...")
sample_embs, valid_y, valid_soil, valid_climate, valid_indices = [], [], [], [], []

for i, row in train_df.iterrows():
    fid, year = row["Field_ID"], int(row["Year"])
    if fid not in aux_df.index:
        continue
    aux_row = aux_df.loc[fid]
    
    # 提取土壤
    soil_feat = aux_row[SOIL_COLS].values.astype(np.float32)
    
    # 提取气候序列
    climate_seq = np.full((12, len(CLIMATE_VARS)), np.nan, dtype=np.float32)
    for month in range(12):
        for j, var in enumerate(CLIMATE_VARS):
            col = f"climate_{year}_{month+1}_{var}"
            if col in aux_row.index and pd.notna(aux_row[col]):
                climate_seq[month, j] = float(aux_row[col])
    
    # 生成 embedding
    emb = compute_sample_embedding(climate_seq, soil_feat, CLIMATE_VARS, SOIL_COLS, var_embeddings)
    sample_embs.append(emb)
    valid_y.append(row["Yield"])
    valid_soil.append(soil_feat)
    valid_climate.append(climate_seq)
    valid_indices.append(i)

sample_embs = np.stack(sample_embs)  # (N, 1536)
valid_y = np.array(valid_y, dtype=np.float32)
valid_soil = np.stack(valid_soil)
valid_climate = np.stack(valid_climate)
valid_climate_mean = np.nanmean(valid_climate[:, GROWTH_MONTHS, :], axis=1)  # (N, 14)

print(f"✅ 成功生成 {len(sample_embs)} 个样本 embedding")

# ----------------------------
# 4. 聚类与内部一致性验证
# ----------------------------
print("\n📊 对样本 embedding 进行聚类...")
scaler = StandardScaler()
sample_embs_scaled = scaler.fit_transform(sample_embs)
n_clusters = 5
cluster_labels = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(sample_embs_scaled)

# TSNE 可视化
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_embs)-1))
X_tsne = tsne.fit_transform(sample_embs_scaled)
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter)
plt.title("样本级 Embedding 聚类 (TSNE)")
plt.tight_layout()
plt.savefig("sample_embedding_clusters.png", dpi=150)
plt.close()

# 内部一致性分析
print("\n🔍 聚类内部一致性分析:")
global_yield_var = np.var(valid_y)
global_soil_var = np.mean(np.nanvar(valid_soil, axis=0))
global_climate_var = np.mean(np.nanvar(valid_climate_mean, axis=0))

intra_vars = []
for cid in range(n_clusters):
    mask = cluster_labels == cid
    n = mask.sum()
    if n == 0:
        continue
    y_cluster = valid_y[mask]
    soil_cluster = valid_soil[mask]
    climate_cluster = valid_climate_mean[mask]
    
    yield_var = np.var(y_cluster)
    soil_pairwise = np.mean([
        np.linalg.norm(soil_cluster[i] - soil_cluster[j])
        for i in range(n) for j in range(i+1, n)
    ]) if n > 1 else 0.0
    climate_pairwise = np.mean([
        np.linalg.norm(climate_cluster[i] - climate_cluster[j])
        for i in range(n) for j in range(i+1, n)
    ]) if n > 1 else 0.0
    
    intra_vars.append(yield_var)
    print(f"Cluster {cid} (n={n:3d}): Yield Var={yield_var:.2f}, Soil L2={soil_pairwise:.3f}, Climate L2={climate_pairwise:.3f}")

avg_intra_yield_var = np.mean(intra_vars)
print(f"\n🌍 全局参考: Yield Var={global_yield_var:.2f}")
print(f"📌 平均簇内 Yield 方差: {avg_intra_yield_var:.2f}")

if avg_intra_yield_var < global_yield_var * 0.7:
    print("\n✅ 结论：样本 embedding 聚类有效！簇内样本在产量、土壤、气候上更相似。")
    EMBEDDING_IS_VALID = True
else:
    print("\n⚠️ 结论：聚类效果不显著，embedding 可能未捕获有效语义。")
    EMBEDDING_IS_VALID = False

# ----------------------------
# 5. 构建全局 prompt embedding（用于模型）
# ----------------------------
prompt_text = "东非玉米产量预测相关变量：" + "；".join(
    [f"{k}表示{k_desc}" for k, k_desc in variable_descriptions.items()]
)
global_embedding_full = get_single_embedding(prompt_text)
global_embedding = global_embedding_full  # 不再 PCA
print(f"\n✅ 全局 prompt embedding 维度: {global_embedding.shape[0]}")

# ----------------------------
# 6. 特征构建（支持消融）
# ----------------------------
USE_GLOBAL_CONTEXT = True  # ← 控制是否注入 embedding

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

def clean_array(X):
    X = np.where(np.isinf(X), np.nan, X)
    X = np.nan_to_num(X, nan=0.0)
    return np.clip(X, -1e6, 1e6).astype(np.float32)

def build_features(df, aux_df, global_emb=None, use_global=True):
    climate_list, soil_list, y_list = [], [], []
    for _, row in df.iterrows():
        fid, year = row["Field_ID"], int(row["Year"])
        if fid in aux_df.index:
            aux_row = aux_df.loc[fid]
            soil_feat = aux_row[SOIL_COLS].values.astype(np.float32)
            climate_seq = np.zeros((12, len(CLIMATE_VARS)), dtype=np.float32)
            for month in GROWTH_MONTHS:
                for j, var in enumerate(CLIMATE_VARS):
                    col = f"climate_{year}_{month+1}_{var}"
                    if col in aux_row.index and pd.notna(aux_row[col]):
                        climate_seq[month, j] = float(aux_row[col])
            climate_list.append(climate_seq)
            soil_list.append(soil_feat)
        else:
            climate_list.append(np.zeros((12, len(CLIMATE_VARS)), dtype=np.float32))
            soil_list.append(np.zeros(len(SOIL_COLS), dtype=np.float32))
        if "Yield" in row:
            y_list.append(row["Yield"])
    climate = clean_array(np.stack(climate_list))
    soil = clean_array(np.stack(soil_list))
    y = np.array(y_list, dtype=np.float32) if y_list else None
    N = climate.shape[0]
    if use_global and global_emb is not None:
        X_global = np.tile(global_emb, (N, 1)).astype(np.float32)
    else:
        X_global = np.zeros((N, 1), dtype=np.float32)
    return climate, soil, X_global, y

# 构建特征
if USE_GLOBAL_CONTEXT:
    climate_train, soil_train, X_global_train, y_train = build_features(train_df, aux_df, global_embedding, True)
    climate_test, soil_test, X_global_test, _ = build_features(test_df, aux_df, global_embedding, True)
else:
    climate_train, soil_train, X_global_train, y_train = build_features(train_df, aux_df, use_global=False)
    climate_test, soil_test, X_global_test, _ = build_features(test_df, aux_df, use_global=False)

y_train = clean_array(y_train)
y_train = np.clip(y_train, 0.0, 200.0)

# 标准化
N, T, C = climate_train.shape
climate_flat = climate_train.reshape(-1, C)
climate_scaler = StandardScaler()
climate_train_scaled = climate_scaler.fit_transform(climate_flat).reshape(N, T, C)
climate_test_scaled = climate_scaler.transform(climate_test.reshape(-1, C)).reshape(-1, T, C)

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
# 7. 模型定义
# ----------------------------
class YieldDataset(Dataset):
    def __init__(self, X_seq, X_global, y=None):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_global = torch.tensor(X_global, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X_seq)
    def __getitem__(self, i):
        return (self.X_seq[i], self.X_global[i]), self.y[i] if self.y is not None else (self.X_seq[i], self.X_global[i])

class TimeShiftedTransformerWithGlobal(nn.Module):
    def __init__(self, seq_len=12, input_dim=34, embed_dim=128, global_dim=1536, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_proj = nn.Linear(global_dim, embed_dim)
        self.dropout_global = nn.Dropout(dropout)
        self.lag_weights = nn.Parameter(torch.randn(seq_len))
        self.regressor = nn.Sequential(nn.Linear(embed_dim * 2, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x_seq, x_global):
        x = self.embedding(x_seq)
        out = self.transformer(x, mask=self.causal_mask)
        weights = torch.softmax(self.lag_weights, dim=0)
        seq_repr = (out * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        global_repr = self.dropout_global(self.global_proj(x_global))
        fused = torch.cat([seq_repr, global_repr], dim=-1)
        return self.regressor(fused).squeeze(-1)

# ----------------------------
# 8. 训练
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeShiftedTransformerWithGlobal(
    seq_len=12,
    input_dim=X_seq_tr.shape[-1],
    global_dim=X_global_tr.shape[1]
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

train_dataset = YieldDataset(X_seq_tr, X_global_tr, y_tr)
val_dataset = YieldDataset(X_seq_val, X_global_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

best_val_rmse = float('inf')
suffix = "_with_context" if USE_GLOBAL_CONTEXT else "_no_context"

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
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), f"best_model{suffix}.pth")
    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f}")

# ----------------------------
# 9. 生成提交
# ----------------------------
model.load_state_dict(torch.load(f"best_model{suffix}.pth", map_location=device))
model.eval()
test_dataset = YieldDataset(X_test_seq, X_global_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
with torch.no_grad():
    preds = []
    for x_seq, x_global in test_loader:
        x_seq, x_global = x_seq.to(device), x_global.to(device)
        pred = model(x_seq, x_global)
        preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds)

submission = pd.DataFrame({
    "Field_ID": test_df["Field_ID"],
    "Yield": np.clip(preds, 0, None)
})
submission.to_csv(f"submission{suffix}.csv", index=False)
print(f"\n✅ 提交文件已保存: submission{suffix}.csv")
print(f"🎯 Embedding 有效性判断: {'有效' if EMBEDDING_IS_VALID else '无效'}")
