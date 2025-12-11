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
# 清洗函数
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
# 构建结构化特征
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
# 新增：使用 FLAN-T5 提取 LLM 先验
# ----------------------------
from transformers import T5ForConditionalGeneration, AutoTokenizer
import re
import json

def get_llm_monthly_prior(
    model_name="google/flan-t5-small",
    region="US Midwest",
    crop="corn (maize)",
    soil_type="loam"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[LLM Prior] Loading {model_name} on {device}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # 兼容 CPU
            low_cpu_mem_usage=True
        ).to(device)

        prompt = (
            f"You are an expert agronomist specializing in {crop} production in the {region} with {soil_type} soil. "
            "Assign a monthly importance score (0 to 10) for each month (January to December) regarding its impact on final yield. "
            "The tasseling-silking stage (most critical for yield) typically occurs in May. "
            "Return ONLY a JSON list of 12 numbers like [1.2, 1.0, 2.5, 4.0, 5.5, 4.8, 3.0, 1.8, 1.0, 0.9, 0.8, 0.7]."
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                num_beams=1
            )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[LLM Output] {output_text}")

        # 尝试解析 JSON 列表
        match = re.search(r"\[([\d\.\,\s\-]+)\]", output_text)
        if match:
            list_str = match.group(1)
            scores = [float(x.strip()) for x in list_str.split(",") if x.strip()]
            if len(scores) == 12:
                scores = np.array(scores, dtype=np.float32)
                scores = np.clip(scores, 0, 10)
                if scores.sum() > 0:
                    return scores / scores.sum()

    except Exception as e:
        print(f"[LLM Prior] Failed: {e}")

    # Fallback to default prior
    print("[LLM Prior] Using default prior.")
    default_prior = np.array([1.0, 1.0, 1.5, 3.0, 3.5, 3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0])
    return default_prior / default_prior.sum()

# 获取先验（全局）
llm_prior_probs = get_llm_monthly_prior()
np.save("llm_monthly_prior.npy", llm_prior_probs)
llm_prior_logits = np.log(llm_prior_probs + 1e-8)

# ----------------------------
# 模型定义（注入 LLM 先验）
# ----------------------------
class TimeShiftedTransformerYieldPredictor(nn.Module):
    def __init__(self, seq_len=12, input_dim=34, embed_dim=128, nhead=8, num_layers=2, dropout=0.1, prior_logits=None):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        if prior_logits is None:
            prior_logits = torch.ones(seq_len)
        else:
            prior_logits = torch.tensor(prior_logits, dtype=torch.float32)
        self.register_buffer("prior_probs", torch.softmax(prior_logits, dim=0))
        self.lag_weights = nn.Parameter(prior_logits.clone())

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

    def get_kl_loss(self):
        learned_probs = torch.softmax(self.lag_weights, dim=0)
        kl = torch.sum(self.prior_probs * torch.log((self.prior_probs + 1e-8) / (learned_probs + 1e-8)))
        return kl

# ----------------------------
# 训练设置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_tr.shape[-1]  # 14 + S

model = TimeShiftedTransformerYieldPredictor(
    seq_len=12,
    input_dim=input_dim,
    embed_dim=128,
    nhead=8,
    num_layers=2,
    dropout=0.1,
    prior_logits=llm_prior_logits
).to(device)

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
LAMBDA_KL = 0.1

for epoch in range(50):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss_pred = criterion(pred, y)
        loss_kl = LAMBDA_KL * model.get_kl_loss()
        loss = loss_pred + loss_kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

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
    val_var = np.var(val_targets - val_preds)

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_time_shifted_transformer.pth")

    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f} | Residual Variance: {val_var:.4f}")

# ----------------------------
# 最终评估
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

# 打印学习到的权重
lag_weights = torch.softmax(model.lag_weights, dim=0).detach().cpu().numpy()
print("\n Learned lag weights (month 1 to 12):")
for i, w in enumerate(lag_weights, 1):
    print(f"  Month {i:2d}: {w:.4f}")

# ----------------------------
# 测试预测 & 提交
# ----------------------------
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
submission.to_csv("submission_time_shifted_transformer_growing_season.csv", index=False)
print("\n Submission saved to submission_time_shifted_transformer_growing_season.csv")
