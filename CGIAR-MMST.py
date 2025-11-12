import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=(41, 41), patch_size=7, in_chans=1, embed_dim=768, temporal_dim=360):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim

        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, T, H, W = x.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0)
            H, W = H + pad_h, W + pad_w

        assert H % self.patch_size == 0 and W % self.patch_size == 0
        x = x.view(B * T, self.in_chans, H, W)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.view(B, T, self.num_patches, self.embed_dim)
        return x


class MultiModalTransformer(nn.Module):
    def __init__(self, embed_dim=768, depth=2, num_heads=12, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, T, N, D = x.shape
        x = x.view(B, T * N, D)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.view(B, T, N, D)
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=768, aux_dim=64, depth=2, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.aux_proj = nn.Linear(aux_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 1, embed_dim))

        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)

    def forward(self, x, aux):
        B, T, _ = x.shape
        aux_proj = self.aux_proj(aux).unsqueeze(1)
        x = torch.cat([x, aux_proj], dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)

        return x[:, 0]


class SimplifiedMMSTViT(nn.Module):
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

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim, temporal_dim=temporal_dim)

        self.mm_transformer = MultiModalTransformer(embed_dim=embed_dim, depth=mm_depth,
                                                    num_heads=num_heads, mlp_ratio=mlp_ratio)

        self.temporal_transformer = TemporalTransformer(embed_dim=embed_dim, aux_dim=aux_dim,
                                                        depth=temp_depth, num_heads=num_heads,
                                                        mlp_ratio=mlp_ratio)

        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, img, aux):
        x = self.patch_embed(img)
        x = self.mm_transformer(x)
        x = x.mean(dim=2)
        x = self.temporal_transformer(x, aux)
        x = self.head(x)
        return x.squeeze(-1)


# ----------------------------
# 主训练流程
# ----------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df = pd.read_csv(os.path.join(DATA_DIR, "Train.csv"))
    train_df['Yield'] = pd.to_numeric(train_df['Yield'], errors='coerce')
    train_df = train_df.dropna(subset=['Yield']).reset_index(drop=True)

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_field_ids_with_year.csv"))
    soil_climate = pd.read_excel(os.path.join(DATA_DIR, "samply.xlsx"), engine='openpyxl')
    soil_climate.set_index("Field_ID", inplace=True)

    train_meta, val_meta = train_test_split(train_df, test_size=0.2, random_state=42)

    train_dataset = MMSTViTDataset(train_meta, soil_climate, is_train=True)
    val_dataset = MMSTViTDataset(val_meta, soil_climate, is_train=True, aux_scaler=train_dataset.aux_scaler)
    test_dataset = MMSTViTDataset(test_df, soil_climate, is_train=False, aux_scaler=train_dataset.aux_scaler)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    aux_dim = train_dataset.aux_feats.shape[1]
    model = SimplifiedMMSTViT(
        img_size=(41, 41),
        patch_size=7,
        in_chans=1,
        embed_dim=256,
        mm_depth=2,
        temp_depth=2,
        num_heads=8,
        mlp_ratio=4.,
        aux_dim=aux_dim,
        temporal_dim=360
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    criterion = nn.MSELoss()

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

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_mmst_vit_model.pth"))
            print(f"New best model saved with Val RMSE: {best_val_rmse:.4f}")

    print("Loading best model for inference...")
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "best_mmst_vit_model.pth")))
    model.eval()
    final_preds = []

    with torch.no_grad():
        for img, aux in test_loader:
            img, aux = img.to(device), aux.to(device)
            pred = model(img, aux)
            final_preds.extend(pred.cpu().numpy())

    submission = pd.DataFrame({
        "Field_ID": test_df["Field_ID"],
        "Yield": np.clip(final_preds, 0, None)
    })
    submission.to_csv(os.path.join(DATA_DIR, "submission_MMST_ViT.csv"), index=False)
    print("Submission file 'submission_MMST_ViT.csv' saved.")


if __name__ == "__main__":
    main()
