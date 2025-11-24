import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ----------------------------
# ... [前面的数据加载、特征构建、标准化、分组划分等代码完全不变] ...
# （此处省略，与你之前代码一致）
# ----------------------------

# ----------------------------
# ⭐ 带 KL 正则的 Time-Shifted Transformer
# ----------------------------
class TimeShiftedTransformerYieldPredictor(nn.Module):
    def __init__(self, seq_len=12, input_dim=14+20, embed_dim=128, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.month_embedding = nn.Embedding(seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 可学习滞后权重（logits）
        self.lag_logits = nn.Parameter(torch.randn(seq_len))
        
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask)

    def get_lag_weights(self):
        return torch.softmax(self.lag_logits, dim=0)  # (12,)

    def forward(self, x):
        B, L, D = x.shape
        x_feat = self.embedding(x)
        month_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x_month = self.month_embedding(month_ids)
        x = x_feat + x_month
        
        out = self.transformer(x, mask=self.causal_mask)
        weights = self.get_lag_weights()  # (L,)
        weighted_repr = (out * weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
        return self.regressor(weighted_repr).squeeze(-1)


# ----------------------------
# 定义农业先验分布（生长季 4–9月 高权重）
# ----------------------------
def get_agricultural_prior(seq_len=12, growing_season=(3, 9)):
    """
    growing_season: (start_month_idx, end_month_idx), e.g., (3,9) for Apr-Sep (0-indexed)
    """
    prior = np.zeros(seq_len)
    prior[growing_season[0]:growing_season[1]] = 1.0
    # 可选：让峰值在 6-8 月更高（更精细）
    # prior[5:8] *= 2.0  # Jul-Aug more important
    prior = prior / prior.sum()
    return torch.tensor(prior, dtype=torch.float32)


# ----------------------------
# 训练设置
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
criterion_mse = nn.MSELoss()

# 农业先验（固定）
agri_prior = get_agricultural_prior().to(device)
lambda_kl = 0.1  # ← 可调超参！建议尝试 [0.01, 0.1, 0.5, 1.0]

train_dataset = YieldDataset(X_tr, y_tr)
val_dataset = YieldDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ----------------------------
# 训练循环（含 KL 正则）
# ----------------------------
best_val_rmse = float('inf')
for epoch in range(50):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        mse_loss = criterion_mse(pred, y)
        
        # 计算 KL 正则项
        learned_probs = model.get_lag_weights()  # (12,)
        # KL(prior || learned) = sum prior * log(prior / learned)
        kl_loss = torch.sum(agri_prior * torch.log(agri_prior / (learned_probs + 1e-8)))
        
        total_loss = mse_loss + lambda_kl * kl_loss
        
        optimizer.zero_grad()
        total_loss.backward()
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
        torch.save(model.state_dict(), "best_model_kl_regularized.pth")

    # 打印 KL loss 和权重（可选）
    with torch.no_grad():
        current_weights = model.get_lag_weights().cpu().numpy()
        current_kl = np.sum(agri_prior.cpu().numpy() * np.log(agri_prior.cpu().numpy() / (current_weights + 1e-8)))
    print(f"Epoch {epoch+1:2d} | Val RMSE: {val_rmse:.4f} | KL Loss: {current_kl:.4f}")

# ----------------------------
# 最终评估
# ----------------------------
model.load_state_dict(torch.load("best_model_kl_regularized.pth", map_location=device))
model.eval()

with torch.no_grad():
    val_preds = []
    for x, _ in val_loader:
        x = x.to(device)
        pred = model(x)
        val_preds.append(pred.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    final_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

print("\n Final Validation RMSE (with KL Regularization):", f"{final_rmse:.4f}")

# 打印学习到的 lag weights
lag_weights = model.get_lag_weights().detach().cpu().numpy()
print("\n Learned lag weights (Month 1 to 12):")
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
submission.to_csv("submission_kl_regularized.csv", index=False)
print("\n Submission saved to submission_kl_regularized.csv")
