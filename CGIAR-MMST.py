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
train_file = os.path.join(DATA_DIR, "Train.csv")
test_file = os.path.join(DATA_DIR, "test_field_ids_with_year.csv")
aux_file = os.path.join(DATA_DIR, "fields_w_additional_info.csv")

# ----------------------------
# 定义作物生长季（4月到9月，对应索引3～8）
# ----------------------------
GROWTH_MONTHS = list(range(3, 9))  # Apr=3, ..., Sep=8 (0-based)

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
# 辅助函数：构建结构化气候序列 + 土壤特征
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
# Time-Shifted Feature Extractor
# ----------------------------
class TimeShiftedFeatureExtractor(nn.Module):
    def __init__(self, seq_len=12, input_dim=34, embed_dim=128, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lag_weights = nn.Parameter(torch.randn(seq_len))
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def forward(self, x):
        x = self.embedding(x)
        out = self.transformer(x, mask=self.causal_mask)
        weights = torch.softmax(self.lag_weights, dim=0)
        weighted_repr = (out * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return weighted_repr

# ----------------------------
# LLM 回归头（使用本地 Qwen-1.8B-Chat）
# ----------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

LOCAL_LLM_PATH = "/home/zhongfangxiu/.cache/modelscope/hub/models/qwen/Qwen-1_8B-Chat"

class LLMRegressor(nn.Module):
    def __init__(self, feature_dim=128, llm_local_path=LOCAL_LLM_PATH):
        super().__init__()
        print(f"✅ Loading Qwen-1.8B-Chat from local path: {llm_local_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_local_path,
            trust_remote_code=True,
            use_fast=False,           # Qwen 不支持 fast tokenizer
            local_files_only=True     # 禁止联网
        )
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_local_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True     # 关键：仅使用本地文件
        )
        
        # 冻结原始参数
        for param in self.llm.parameters():
            param.requires_grad = False

        # ⚠️ Qwen-1.8B 是 Qwen1 架构，attention 层名为 "c_attn"
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["c_attn"],   # ← 正确模块名（不是 q_proj/v_proj！）
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

        self.proj = nn.Linear(feature_dim, self.llm.config.hidden_size)
        self.regressor = nn.Linear(self.llm.config.hidden_size, 1)

    def forward(self, features):
        prompt_embeds = self.proj(features).unsqueeze(1)  # (B, 1, hidden_size)
        outputs = self.llm(inputs_embeds=prompt_embeds, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        return self.regressor(last_hidden).squeeze(-1)

# ----------------------------
# 组合模型
# ----------------------------
class YieldPredictorWithLLM(nn.Module):
    def __init__(self, feature_extractor, llm_regressor):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.llm_regressor = llm_regressor

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.llm_regressor(features)

# ----------------------------
# 训练设置
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_tr.shape[-1]

feature_extractor = TimeShiftedFeatureExtractor(
    seq_len=12,
    input_dim=input_dim,
    embed_dim=128,
    nhead=8,
    num_layers=2,
    dropout=0.1
)

llm_regressor = LLMRegressor(feature_dim=128)

model = YieldPredictorWithLLM(feature_extractor, llm_regressor).to(device)

# 只优化 LLM 回归头中的可训练参数（LoRA + proj + regressor）
optimizer = torch.optim.AdamW(
    model.llm_regressor.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)
criterion = nn.MSELoss()

train_dataset = YieldDataset(X_tr, y_tr)
val_dataset = YieldDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ----------------------------
# 训练循环
# ----------------------------
best_val_rmse = float('inf')
for epoch in range(30):
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

    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        torch.save(model.state_dict(), "best_llm_yield_predictor.pth")

    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f}")

# ----------------------------
# 最终评估
# ----------------------------
model.load_state_dict(torch.load("best_llm_yield_predictor.pth", map_location=device))
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

# 打印学习到的滞后权重
lag_weights = torch.softmax(model.feature_extractor.lag_weights, dim=0).detach().cpu().numpy()
print("\n Learned lag weights (month 1 to 12):")
for i, w in enumerate(lag_weights, 1):
    print(f"  Month {i:2d}: {w:.4f}")

# ----------------------------
# 测试预测 & 提交
# ----------------------------
test_dataset = YieldDataset(X_test_full)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
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
submission.to_csv("submission_llm_lora.csv", index=False)
print("\n Submission saved to submission_llm_lora.csv")
