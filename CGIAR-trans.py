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
# 新增：导入 Qwen 相关模块
# ----------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

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

# ----------------------------
# 新增：加载本地 Qwen 模型（自动选择 CPU/GPU）
# ----------------------------
print("Loading Qwen model...")
device_qwen = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct", trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    "Qwen2-0.5B-Instruct",
    torch_dtype=torch.float16 if device_qwen == "cuda" else torch.float32,
    trust_remote_code=True
).to(device_qwen).eval()
print(f"Qwen loaded on {device_qwen}")

# ----------------------------
# 特征描述（用于构造 prompt）
# ----------------------------
VAR_DESCRIPTIONS = {
    "aet": "实际蒸散量",
    "def": "水分亏缺",
    "pdsi": "帕尔默干旱指数",
    "pet": "潜在蒸散量",
    "pr": "降水量",
    "ro": "地表径流",
    "soil": "土壤含水量",
    "srad": "太阳辐射",
    "swe": "雪水当量",
    "tmmn": "月最低气温",
    "tmmx": "月最高气温",
    "vap": "水汽压",
    "vpd": "饱和水汽压差",
    "vs": "风速"
}

def get_llm_feature_ranking(var_names, soil_dim):
    """
    调用本地 Qwen 模型，获取玉米产量预测中各气候和土壤变量的重要性排序。
    返回一个长度为 (len(var_names) + soil_dim) 的权重向量（非归一化）。
    """
    # 构造特征列表（带中文解释）
    feature_list = []
    for var in var_names:
        desc = VAR_DESCRIPTIONS.get(var, var)
        feature_list.append(f"{var}（{desc}）")
    
    soil_features = [f"soil_{i}" for i in range(soil_dim)]
    all_features = feature_list + soil_features

    prompt = (
        "你是一位农业气象专家。在预测中国玉米产量时，请根据农学常识，对以下环境变量按重要性从高到低排序（最重要的排最前）。"
        "仅输出一个 JSON 列表，格式为：[\"var1\", \"var2\", ...]，不要任何解释。\n\n"
        "变量列表：\n" + "\n".join(all_features)
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device_qwen)

    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **model_inputs,
            max_new_tokens=200,
            do_sample=False,  # 确保确定性
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    try:
        ranked_vars = json.loads(response.strip())
        if not isinstance(ranked_vars, list):
            raise ValueError("Not a list")
    except Exception as e:
        print(f"⚠️ Qwen 返回格式错误，使用默认排序。错误：{e}")
        print(f"Raw response: {response}")
        # 回退到农学常识排序
        high_impact = ["tmmx", "tmmn", "pr", "soil", "srad", "vpd"]
        medium_impact = ["aet", "pet", "pdsi", "vap"]
        low_impact = ["def", "ro", "swe", "vs"]
        ranked_vars = []
        for v in high_impact:
            if v in var_names:
                ranked_vars.append(v)
        for v in medium_impact:
            if v in var_names:
                ranked_vars.append(v)
        for v in low_impact:
            if v in var_names:
                ranked_vars.append(v)
        ranked_vars += soil_features

    # 构建权重：排名越前，权重越高（1 / rank）
    scores = np.zeros(len(var_names) + soil_dim, dtype=np.float32)
    for rank, var in enumerate(ranked_vars):
        if var.startswith("soil_"):
            idx = len(var_names) + int(var.split("_")[1])
        else:
            if var in var_names:
                idx = var_names.index(var)
            else:
                continue  # 忽略未知变量
        scores[idx] = 1.0 / (rank + 1)

    return scores

# ----------------------------
# 辅助函数：构建结构化气候序列 + 土壤特征（带生长季掩码）
# ----------------------------
def build_features_structured(df, aux_df, growth_months=GROWTH_MONTHS):
    soil_cols = [col for col in aux_df.columns if col.startswith("soil_")]
    climate_seq_list = []
    soil_feat_list = []
    y_list = []
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

    climate_seqs = np.stack(climate_seq_list)
    soil_feats = np.stack(soil_feat_list)
    y = np.array(y_list, dtype=np.float32) if y_list else None
    return climate_seqs, soil_feats, y

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
# 构建特征
# ----------------------------
climate_train, soil_train, y_train = build_features_structured(train_df, aux_df)
climate_test, soil_test, _ = build_features_structured(test_df, aux_df)

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

X_train_full = np.concatenate([climate_train_scaled, soil_train_tiled], axis=-1)
X_test_full = np.concatenate([climate_test_scaled, soil_test_tiled], axis=-1)

# ----------------------------
# 获取 LLM 特征权重（关键新增步骤）
# ----------------------------
var_names = ["aet", "def", "pdsi", "pet", "pr", "ro", "soil", "srad", "swe", "tmmn", "tmmx", "vap", "vpd", "vs"]
soil_dim = S
print("Calling Qwen to rank feature importance...")
feature_importance_scores = get_llm_feature_ranking(var_names, soil_dim)
print("LLM feature weights obtained.")

# ----------------------------
# 划分验证集
# ----------------------------
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_full, y_train, test_size=0.2, random_state=42
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
# 修改模型：支持 LLM 特征加权（不再使用 KL，仅加权输入）
# ----------------------------
class TimeShiftedTransformerYieldPredictor(nn.Module):
    def __init__(self, seq_len=12, input_dim=14+20, embed_dim=128, nhead=8, num_layers=2, dropout=0.1, feature_weights=None):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        
        if feature_weights is not None:
            self.register_buffer("feature_weights", feature_weights)
        else:
            self.feature_weights = None

        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 先验引导（保留，但训练时不加 KL loss）
        prior_logits = torch.tensor([1.0, 1.0, 1.5, 3.0, 3.5, 3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0])
        self.register_buffer("prior_logits", prior_logits)
        self.lag_weights = nn.Parameter(torch.log(prior_logits + 1e-6))
        
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        if self.feature_weights is not None:
            x = x * self.feature_weights  # ⭐ LLM 特征加权
        x = self.embedding(x)
        out = self.transformer(x, mask=self.causal_mask)
        weights = torch.softmax(self.lag_weights, dim=0)
        weighted_repr = (out * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return self.regressor(weighted_repr).squeeze(-1)

# ----------------------------
# 训练设置（移除 KL loss）
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
    feature_weights=torch.tensor(feature_importance_scores, dtype=torch.float32)
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

train_dataset = YieldDataset(X_tr, y_tr)
val_dataset = YieldDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----------------------------
# 训练循环（无 KL loss）
# ----------------------------
best_val_rmse = float('inf')

for epoch in range(50):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)  # ⭐ 仅 MSE loss
        
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

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_time_shifted_transformer_llm_weighted.pth")

    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f}")

# ----------------------------
# 最终评估 & 提交（保持不变）
# ----------------------------
model.load_state_dict(torch.load("best_time_shifted_transformer_llm_weighted.pth", map_location=device))
model.eval()
with torch.no_grad():
    val_preds = []
    for x, _ in val_loader:
        x = x.to(device)
        pred = model(x)
        val_preds.append(pred.cpu().numpy())
    final_rmse = np.sqrt(mean_squared_error(y_val, np.concatenate(val_preds)))
    print(f"\nFinal Val RMSE: {final_rmse:.4f}")

lag_weights = torch.softmax(model.lag_weights, dim=0).detach().cpu().numpy()
print("\nLearned lag weights:")
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
submission.to_csv("submission_llm_weighted_transformer.csv", index=False)
print("\n✅ Submission saved to submission_llm_weighted_transformer.csv")
