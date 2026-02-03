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
# 0. 阿里云 DashScope + 长文本分块 Embedding
# ----------------------------
try:
    import dashscope
    from dashscope import TextEmbedding
    import tiktoken
except ImportError:
    raise ImportError("请安装依赖: pip install dashscope tiktoken")

# 🔑 替换为你的阿里云 API Key
dashscope.api_key = "sk-65aa3b4c924b43e29bbffe9430eeb010"

def split_text_into_chunks(text: str, max_tokens: int = 2000) -> list[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_text = encoding.decode(tokens[start:end])
        chunks.append(chunk_text)
        start = end
    return chunks

def get_embedding_for_long_text(text: str, model: str = "text-embedding-v2") -> np.ndarray:
    chunks = split_text_into_chunks(text, max_tokens=2000)
    print(f"📄 原始 prompt 被切分为 {len(chunks)} 个 chunk 进行 embedding")

    embeddings = []
    weights = []
    encoding = tiktoken.get_encoding("cl100k_base")

    for i, chunk in enumerate(chunks):
        token_count = len(encoding.encode(chunk))
        print(f"  → 调用 API 处理 chunk {i+1}/{len(chunks)} ({token_count} tokens)")
        
        response = TextEmbedding.call(
            model=model,
            input=chunk.strip()
        )
        if response.status_code != 200:
            raise RuntimeError(f"Chunk {i+1} embedding failed: {response}")
        
        emb = np.array(response.output["embeddings"][0]["embedding"], dtype=np.float32)
        embeddings.append(emb)
        weights.append(token_count)

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()
    weighted_emb = sum(w * emb for w, emb in zip(weights, embeddings))
    return weighted_emb.astype(np.float32)

# 提示词（东非玉米产量预测上下文）
prompt_text = """
你是一个气候与土壤数据分析助手。以下是东非地区预测玉米产量时常用的土壤属性与气象指标说明：
soil_bulk_density：土壤容重，反映土壤紧实程度，影响根系生长和水分渗透。
soil_cec：土壤阳离子交换量，表征土壤保肥能力，数值越高，养分保持能力越强。
soil_coarse_fragments：土壤中粗碎屑（如砾石）的含量，影响土壤持水性和耕作性能。
soil_clay：黏粒含量，决定土壤的保水性、通气性和结构稳定性。
soil_nitrogen：土壤全氮含量，是衡量土壤肥力的重要指标之一。
soil_organic_carbon_density：单位体积土壤中有机碳的质量，用于评估碳储存能力。
soil_organic_carbon_stock：单位面积土壤剖面中储存的有机碳总量，常用于碳汇核算。
soil_ph：土壤酸碱度，影响养分有效性及微生物活性。
soil_sand：砂粒含量，砂质高的土壤排水快但保肥能力弱。
soil_silt：粉粒含量，介于砂与黏粒之间，影响土壤质地和保水性。
soil_organic_carbon：土壤有机碳含量，反映土壤有机质水平和健康状况。
气候相关指标包括：
aet（Actual Evapotranspiration）：实际蒸散发，表示地表水分通过蒸发和植物蒸腾返回大气的总量。
def（Water Deficit）：水分亏缺，即潜在蒸散发与实际供水之间的差额，反映干旱胁迫程度。
pdsi（Palmer Drought Severity Index）：帕尔默干旱指数，综合降水与蒸散发评估长期干旱状况。
pet（Potential Evapotranspiration）：潜在蒸散发，在水分充足条件下可能发生的最大蒸散量。
pr（Precipitation）：降水量，指一定时期内降落到地面的液态或固态水总量。
ro（Runoff）：地表径流，降水未入渗而沿地表流动的部分，影响水资源与侵蚀。
soil_moisture：土壤湿度，表征土壤中含水量，直接影响作物生长和水文过程。
srad（Surface Solar Radiation）：地表太阳辐射，驱动光合作用、蒸发和地表能量平衡。
swe（Snow Water Equivalent）：雪水当量，指积雪融化后对应的水深，是冬季水资源的重要指标。
tmmn（Mean Minimum Temperature）：月平均最低气温，反映夜间或冷季低温状况。
tmmx（Mean Maximum Temperature）：月平均最高气温，反映白天或暖季高温状况。
vap（Vapor Pressure）：水汽压，表示空气中水汽的分压力，与湿度密切相关。
vpd（Vapor Pressure Deficit）：饱和水汽压差，表征大气干燥程度，影响植物蒸腾和水分胁迫。
vs（Wind Speed）：风速，影响蒸发、传热、花粉传播及风蚀过程。
数据集来自2016到2019年各月指标的平均值数据如下所示：
土壤属性（全年各月相同）：
1月到12月，soil_bulk_density的数据平均值为118.16；
1月到12月，soil_cec的数据平均值为198.16；
1月到12月，soil_coarse_fragments的数据平均值为100.51；
1月到12月，soil_clay的数据平均值为439.36；
1月到12月，soil_nitrogen的数据平均值为1551.76；
1月到12月，soil_organic_carbon_density的数据平均值为309.32；
1月到12月，soil_organic_carbon_stock的数据平均值为60.48；
1月到12月，soil_ph的数据平均值为56.38；
1月到12月，soil_sand的数据平均值为290.39；
1月到12月，soil_silt的数据平均值为270.24；
1月到12月，soil_organic_carbon的数据平均值为277.47。
气候与水文变量（按月变化）：
1月，aet的数据平均值为709.31，def为821.39，pdsi为87.32，pet为1530.70，pr为51.59，ro为2.73，soil_moisture为406.91，srad为2397.36，swe为0.00，tmmn为138.99，tmmx为299.61，vap为1589.05，vpd为148.21，vs为281.25。
2月，aet的数据平均值为818.25，def为703.06，pdsi为50.12，pet为1521.31，pr为77.07，ro为4.22，soil_moisture为317.30，srad为2571.08，swe为0.00，tmmn为142.50，tmmx为309.78，vap为1597.10，vpd为170.36，vs为309.84。
3月，aet的数据平均值为805.04，def为794.31，pdsi为-62.40，pet为1599.35，pr为83.46，ro为5.32，soil_moisture为293.79，srad为2507.42，swe为0.00，tmmn为145.07，tmmx为303.15，vap为1696.47，vpd为128.15，vs为287.17。
4月，aet的数据平均值为1180.50，def为67.02，pdsi为-122.93，pet为1247.52，pr为222.01，ro为65.30，soil_moisture为680.48，srad为2122.03，swe为0.00，tmmn为146.25，tmmx为286.70，vap为1849.33，vpd为94.79，vs为242.62。
5月，aet的数据平均值为1057.46，def为118.09，pdsi为-206.88，pet为1175.55，pr为179.82，ro为73.74，soil_moisture为683.89，srad为2107.82，swe为0.00，tmmn为139.14，tmmx为275.62，vap为1890.26，vpd为74.02，vs为217.24。
6月，aet的数据平均值为763.88，def为360.84，pdsi为-232.32，pet为1124.73，pr为116.14，ro为54.56，soil_moisture为535.77，srad为2079.13，swe为0.00，tmmn为136.06，tmmx为275.33，vap为1795.43，vpd为87.32，vs为210.62。
7月，aet的数据平均值为574.95，def为587.48，pdsi为-323.84，pet为1162.43，pr为47.76，ro为2.83，soil_moisture为410.17，srad为2038.99，swe为0.00，tmmn为132.57，tmmx为272.36，vap为1710.61，vpd为91.77，vs为217.68。
8月，aet的数据平均值为820.22，def为393.83，pdsi为-377.14，pet为1214.05，pr为106.67，ro为25.26，soil_moisture为404.07，srad为2082.24，swe为0.00，tmmn为132.43，tmmx为275.52，vap为1717.87，vpd为96.49，vs为226.64。
9月，aet的数据平均值为943.88，def为332.42，pdsi为-394.34，pet为1276.30，pr为101.63，ro为10.24，soil_moisture为374.13，srad为2251.89，swe为0.00，tmmn为131.14，tmmx为284.16，vap为1743.16，vpd为104.33，vs为232.52。
10月，aet的数据平均值为1007.66，def为314.03，pdsi为-376.38，pet为1321.69，pr为130.08，ro为16.83，soil_moisture为498.86，srad为2182.70，swe为0.00，tmmn为137.83，tmmx为289.01，vap为1775.30，vpd为102.19，vs为234.37。
11月，aet的数据平均值为1108.34，def为172.47，pdsi为-380.39，pet为1280.80，pr为109.77，ro为8.87，soil_moisture为399.33，srad为2197.84，swe为0.00，tmmn为138.36，tmmx为291.27，vap为1801.88，vpd为109.34，vs为244.52。
12月，aet的数据平均值为770.90，def为633.86，pdsi为-370.28，pet为1404.74，pr为97.48，ro为11.08，soil_moisture为492.59，srad为2254.79，swe为0.00，tmmn为139.68，tmmx为292.40，vap为1683.09，vpd为118.73，vs为264.37。
"""

print("🚀 正在处理完整 prompt 的嵌入...")
global_embedding = get_embedding_for_long_text(prompt_text, model="text-embedding-v2")
print(f"✅ 全局嵌入生成成功，维度: {global_embedding.shape}")

# ----------------------------
# 后续代码完全不变（从 set_seed 开始到训练结束）
# ----------------------------
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

DATA_DIR = "./data"
train_file = os.path.join(DATA_DIR, "Train.csv")
test_file = os.path.join(DATA_DIR, "test_field_ids_with_year.csv")
aux_file = os.path.join(DATA_DIR, "fields_w_additional_info.csv")

GROWTH_MONTHS = list(range(3, 9))  # Apr=3, ..., Sep=8 (0-based)

train_df = pd.read_csv(train_file, header=None)
train_df.columns = ["Field_ID", "Year", "Quality", "Yield"]
train_df['Yield'] = pd.to_numeric(train_df['Yield'], errors='coerce')
train_df = train_df.dropna(subset=['Yield']).reset_index(drop=True)
test_df = pd.read_csv(test_file)
aux_df = pd.read_csv(aux_file)
aux_df.set_index("Field_ID", inplace=True)

def build_features_structured_with_global(df, aux_df, global_embedding, growth_months=GROWTH_MONTHS):
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
    X_global = np.tile(global_embedding, (N, 1)).astype(np.float32)

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

climate_train, soil_train, X_global_train, y_train = build_features_structured_with_global(train_df, aux_df, global_embedding)
climate_test, soil_test, X_global_test, _ = build_features_structured_with_global(test_df, aux_df, global_embedding)

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

class YieldDatasetWithContext(Dataset):
    def __init__(self, X_seq, X_global, y=None):
        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)
        self.X_global = torch.tensor(X_global, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
    def __len__(self): return len(self.X_seq)
    def __getitem__(self, i):
        if self.y is not None:
            return (self.X_seq[i], self.X_global[i]), self.y[i]
        return (self.X_seq[i], self.X_global[i])

class TimeShiftedTransformerWithGlobalContext(nn.Module):
    def __init__(self, seq_len=12, input_dim=14+20, embed_dim=128, global_dim=1536, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.global_proj = nn.Linear(global_dim, embed_dim)
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
        fused = torch.cat([seq_repr, global_repr], dim=-1)
        return self.regressor(fused).squeeze(-1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_seq_tr.shape[-1]
global_dim = global_embedding.shape[0]

model = TimeShiftedTransformerWithGlobalContext(
    seq_len=12,
    input_dim=input_dim,
    embed_dim=128,
    global_dim=global_dim,
    nhead=8,
    num_layers=2,
    dropout=0.1
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

train_dataset = YieldDatasetWithContext(X_seq_tr, X_global_tr, y_tr)
val_dataset = YieldDatasetWithContext(X_seq_val, X_global_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

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
        torch.save(model.state_dict(), "best_fused_time_shifted_model.pth")

    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f} | Residual Variance: {val_var:.4f}")

model.load_state_dict(torch.load("best_fused_time_shifted_model.pth", map_location=device))
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
print("\n Final Validation Metrics:")
print(f"RMSE: {final_rmse:.4f}")
print(f"Residual Variance: {final_var:.4f}")

lag_weights = torch.softmax(model.lag_weights, dim=0).detach().cpu().numpy()
print("\n Learned lag weights (month 1 to 12):")
for i, w in enumerate(lag_weights, 1):
    marker = " ← GROWTH SEASON" if 4 <= i <= 9 else ""
    print(f"  Month {i:2d}: {w:.4f}{marker}")

test_dataset = YieldDatasetWithContext(X_test_seq, X_global_test)
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
submission.to_csv("submission_fused_time_shifted_with_prompt_context.csv", index=False)
print("\n✅ Submission saved to submission_fused_time_shifted_with_prompt_context.csv")

