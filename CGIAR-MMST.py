# ==============================
# 玉米产量预测：基于 MMST-ViT 思想的实现
# 参考论文: MMST-ViT: Climate Change-aware Crop Yield Prediction via Multi-Modal Spatial-Temporal Vision Transformer (CVPR 2023)
# 数据适配: Field-level data with Sentinel-2 time series and static soil/climate features.
# ==============================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from timm.models.vision_transformer import Block  # 使用 timm 库简化 ViT Block 的实现
import warnings

warnings.filterwarnings("ignore")

# ----------------------------
# 路径配置
# ----------------------------
DATA_DIR = "./data"
IMG_TRAIN_DIR = os.path.join(DATA_DIR, "Image_arrays_train")
IMG_TEST_DIR = os.path.join(DATA_DIR, "Image_arrays_test")


# ----------------------------
# 数据加载与预处理
# ----------------------------

def load_image(field_id, is_train=True, normalize=True):
    """加载并标准化 Sentinel-2 时间序列图像"""
    folder = IMG_TRAIN_DIR if is_train else IMG_TEST_DIR
    path = os.path.join(folder, f"{field_id}.npy")
    if not os.path.exists(path):
        # 如果文件不存在，返回零张量
        return np.zeros((360, 41, 41), dtype=np.float32)
    img = np.load(path).astype(np.float32)
    if normalize:
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)
    return img


def get_aux_features(field_id, soil_climate_df):
    """获取田块的静态辅助特征（土壤、气候等）"""
    if field_id in soil_climate_df.index:
        return soil_climate_df.loc[field_id].values.astype(np.float32)
    return np.zeros(soil_climate_df.shape[1], dtype=np.float32)


# ----------------------------
# 自定义 Dataset
# ----------------------------

class MMSTViTDataset(Dataset):
    def __init__(self, df, soil_climate, is_train=True, aux_scaler=None):
        self.df = df.reset_index(drop=True)
        self.soil_climate = soil_climate
        self.is_train = is_train

        self.images = []
        self.aux_feats = []
        self.targets = []

        for _, row in self.df.iterrows():
            fid = row["Field_ID"]
            self.images.append(load_image(fid, is_train))
            self.aux_feats.append(get_aux_features(fid, soil_climate))
            if is_train:
                self.targets.append(row["Yield"])

        self.images = np.stack(self.images)  # Shape: (N, T=360, H=41, W=41)
        self.aux_feats = np.stack(self.aux_feats)  # Shape: (N, D_aux)

        # 标准化辅助特征
        if aux_scaler is None:
            self.aux_scaler = StandardScaler()
            self.aux_feats = self.aux_scaler.fit_transform(self.aux_feats)
        else:
            self.aux_feats = aux_scaler.transform(self.aux_feats)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx])  # (T, H, W)
        aux = torch.tensor(self.aux_feats[idx])  # (D_aux,)
        if self.is_train:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return img, aux, y
        return img, aux


# ----------------------------
# 模型组件定义 (简化版 MMST-ViT)
# ----------------------------

class PatchEmbed(nn.Module):
    """将时间序列图像分割成 patch 并进行嵌入"""

    def __init__(self, img_size=(41, 41), patch_size=7, in_chans=1, embed_dim=768, temporal_dim=360):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim

        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # 对每个时间步的图像进行 patch 嵌入
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        x: (B, T, H, W)
        Returns: (B, T, N_patches, embed_dim)
        """
        B, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # Reshape to (B*T, 1, H, W) for Conv2d
        x = x.view(B * T, self.in_chans, H, W)
        x = self.proj(x)  # (B*T, embed_dim, grid_h, grid_w)
        x = x.flatten(2).transpose(1, 2)  # (B*T, N_patches, embed_dim)
        # Reshape back to include temporal dimension
        x = x.view(B, T, self.num_patches, self.embed_dim)  # (B, T, N_patches, embed_dim)
        return x


class MultiModalTransformer(nn.Module):
    """简化版 Multi-Modal Transformer (仅处理图像，缺少 ys)"""

    def __init__(self, embed_dim=768, depth=2, num_heads=12, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        """
        x: (B, T, N_patches, embed_dim)
        Returns: (B, T, N_patches, embed_dim)
        """
        B, T, N, D = x.shape
        # 将 T 和 N 合并，应用 Transformer
        x = x.view(B, T * N, D)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.view(B, T, N, D)
        return x


class TemporalTransformer(nn.Module):
    """Temporal Transformer，融合长期特征 (yl -> aux)"""

    def __init__(self, embed_dim=768, aux_dim=64, depth=2, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.aux_proj = nn.Linear(aux_dim, embed_dim)  # 将辅助特征投影到 embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 分类 token
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 1, embed_dim))  # 位置编码 (for cls + temporal)

        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # 初始化 cls token 和 pos embed
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)

    def forward(self, x, aux):
        """
        x: (B, T, embed_dim) - 来自 Spatial Transformer 或直接是全局池化的图像特征
        aux: (B, aux_dim) - 长期辅助特征 (yl 的代理)
        Returns: (B, embed_dim) - 最终表示
        """
        B, T, _ = x.shape

        # 将辅助特征投影并与时间序列特征拼接
        aux_proj = self.aux_proj(aux).unsqueeze(1)  # (B, 1, embed_dim)
        # 这里我们简单地将 aux_proj 作为额外的 "时间步" 加入
        x = torch.cat([x, aux_proj], dim=1)  # (B, T+1, embed_dim)

        # 添加 cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + T + 1, embed_dim)

        # 添加位置编码
        x = x + self.pos_embed

        # 应用 Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # 返回 cls token 的输出
        return x[:, 0]  # (B, embed_dim)


class SimplifiedMMSTViT(nn.Module):
    """简化版 MMST-ViT 模型，适配 field-level 数据"""

    def __init__(self,
                 img_size=(41, 41),
                 patch_size=7,
                 in_chans=1,
                 num_classes=1,
                 embed_dim=768,
                 mm_depth=2,
                 temp_depth=2,
                 num_heads=12,
                 mlp_ratio=4.,
                 aux_dim=64,
                 temporal_dim=360):
        super().__init__()
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, temporal_dim=temporal_dim)

        # 2. Multi-Modal Transformer (简化)
        self.mm_transformer = MultiModalTransformer(embed_dim=embed_dim, depth=mm_depth,
                                                    num_heads=num_heads, mlp_ratio=mlp_ratio)

        # 3. Global Average Pooling over spatial patches (替代 Spatial Transformer)
        # 论文中的 Spatial Transformer 处理 county 网格间的依赖，这里我们对单个 field 的 patch 做平均
        # 输出形状: (B, T, embed_dim)

        # 4. Temporal Transformer
        self.temporal_transformer = TemporalTransformer(embed_dim=embed_dim, aux_dim=aux_dim,
                                                        depth=temp_depth, num_heads=num_heads,
                                                        mlp_ratio=mlp_ratio)

        # 5. Regression Head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.initialize_weights()

    def initialize_weights(self):
        # 其他权重通常由 Block 初始化
        pass

    def forward(self, img, aux):
        """
        img: (B, T=360, H=41, W=41)
        aux: (B, aux_dim) - 长期静态特征
        """
        # 1. Patch Embedding
        x = self.patch_embed(img)  # (B, T, N_patches, embed_dim)

        # 2. Multi-Modal Transformer
        x = self.mm_transformer(x)  # (B, T, N_patches, embed_dim)

        # 3. Spatial Global Average Pooling (简化 Spatial Transformer)
        x = x.mean(dim=2)  # (B, T, embed_dim)

        # 4. Temporal Transformer (融合 aux)
        x = self.temporal_transformer(x, aux)  # (B, embed_dim)

        # 5. Regression Head
        x = self.head(x)  # (B, 1)
        return x.squeeze(-1)  # (B,)


# ----------------------------
# 主训练流程
# ----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 数据加载 ---
    train_df = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    train_df['Yield'] = pd.to_numeric(train_df['Yield'], errors='coerce')
    train_df = train_df.dropna(subset=['Yield']).reset_index(drop=True)

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_field_ids_with_year.csv"))
    soil_climate = pd.read_excel(os.path.join(DATA_DIR, "samply.xlsx"), engine='openpyxl')
    soil_climate.set_index("Field_ID", inplace=True)

    # --- 数据集划分 ---
    train_meta, val_meta = train_test_split(train_df, test_size=0.2, random_state=42)

    # --- 创建 Dataset 和 DataLoader ---
    train_dataset = MMSTViTDataset(train_meta, soil_climate, is_train=True)
    val_dataset = MMSTViTDataset(val_meta, soil_climate, is_train=True, aux_scaler=train_dataset.aux_scaler)
    test_dataset = MMSTViTDataset(test_df, soil_climate, is_train=False, aux_scaler=train_dataset.aux_scaler)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # --- 模型初始化 ---
    aux_dim = train_dataset.aux_feats.shape[1]
    model = SimplifiedMMSTViT(
        img_size=(41, 41),
        patch_size=7,  # 41/7 不是整数，实际会向下取整，可能需要调整或使用 padding
        in_chans=1,
        embed_dim=256,  # 降低维度以适应较小数据集和计算资源
        mm_depth=2,
        temp_depth=2,
        num_heads=8,
        mlp_ratio=4.,
        aux_dim=aux_dim,
        temporal_dim=360
    ).to(device)

    # --- 优化器和损失函数 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.MSELoss()

    # --- 训练循环 ---
    best_val_rmse = float('inf')
    num_epochs = 30

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for img, aux, y in train_loader:
            img, aux, y = img.to(device), aux.to(device), y.to(device)

            optimizer.zero_grad()
            pred = model(img, aux)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # --- 验证 ---
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for img, aux, y in val_loader:
                img, aux, y = img.to(device), aux.to(device), y.to(device)
                pred = model(img, aux)
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_mse = np.mean((np.array(val_preds) - np.array(val_targets)) ** 2)
        val_rmse = np.sqrt(val_mse)
        avg_train_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        # --- 保存最佳模型 ---
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_mmst_vit_model.pth"))
            print(f"New best model saved with Val RMSE: {best_val_rmse:.4f}")

    # --- 测试预测 ---
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "best_mmst_vit_model.pth")))
    model.eval()
    final_preds = []

    with torch.no_grad():
        for img, aux in test_loader:
            img, aux = img.to(device), aux.to(device)
            pred = model(img, aux)
            final_preds.extend(pred.cpu().numpy())

    # --- 保存提交文件 ---
    submission = pd.DataFrame({
        "Field_ID": test_df["Field_ID"],
        "Yield": np.clip(final_preds, 0, None)  # 保证产量非负
    })
    submission.to_csv(os.path.join(DATA_DIR, "submission_MMST_ViT.csv"), index=False)
    print("Submission file 'submission_MMST_ViT.csv' saved.")


if __name__ == "__main__":
    main()