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

DATA_DIR = "./data"
IMG_TRAIN_DIR = os.path.join(DATA_DIR, "Image_arrays_train")
IMG_TEST_DIR = os.path.join(DATA_DIR, "Image_arrays_test")


# ----------------------------
# 数据加载：新增时间下采样
# ----------------------------

def load_image(field_id, is_train=True, normalize=True, time_stride=6):
    """加载图像并进行时间下采样（360 -> 60）"""
    folder = IMG_TRAIN_DIR if is_train else IMG_TEST_DIR
    path = os.path.join(folder, f"{field_id}.npy")
    if not os.path.exists(path):
        img = np.zeros((360, 41, 41), dtype=np.float32)
    else:
        img = np.load(path).astype(np.float32)
    
    # 时间下采样：每 time_stride 帧取 1 帧
    img = img[::time_stride]  # (60, 41, 41)

    if normalize and img.size > 0:
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)
    return img


def get_aux_features(field_id, soil_climate_df):
    if field_id in soil_climate_df.index:
        return soil_climate_df.loc[field_id].values.astype(np.float32)
    return np.zeros(soil_climate_df.shape[1], dtype=np.float32)


class MMSTViTDataset(Dataset):
    def __init__(self, df, soil_climate, is_train=True, aux_scaler=None, time_stride=6):
        self.df = df.reset_index(drop=True)
        self.soil_climate = soil_climate
        self.is_train = is_train
        self.time_stride = time_stride

        self.images = []
        self.aux_feats = []
        self.targets = []

        for _, row in self.df.iterrows():
            fid = row["Field_ID"]
            self.images.append(load_image(fid, is_train, time_stride=time_stride))
            self.aux_feats.append(get_aux_features(fid, soil_climate))
            if is_train:
                self.targets.append(row["Yield"])

        self.images = np.stack(self.images)  # Now: (N, 60, 41, 41)
        self.aux_feats = np.stack(self.aux_feats)

        if aux_scaler is None:
            self.aux_scaler = StandardScaler()
            self.aux_feats = self.aux_scaler.fit_transform(self.aux_feats)
        else:
            self.aux_feats = aux_scaler.transform(self.aux_feats)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx])  # (60, 41, 41)
        aux = torch.tensor(self.aux_feats[idx])
        if self.is_train:
            y = torch.tensor(self.targets[idx], dtype=torch.float32)
            return img, aux, y
        return img, aux


# ----------------------------
# ViT 组件（保持不变）
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


# ----------------------------
# Patch Embedding（带 padding + contiguous）
# ----------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(41, 41), patch_size=7, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        pad_h = (patch_size - img_size[0] % patch_size) % patch_size
        pad_w = (patch_size - img_size[1] % patch_size) % patch_size
        self.pad_h = pad_h
        self.pad_w = pad_w

        new_h = img_size[0] + pad_h
        new_w = img_size[1] + pad_w
        self.grid_size = (new_h // patch_size, new_w // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, T, H, W = x.shape
        if self.pad_h > 0 or self.pad_w > 0:
            x = F.pad(x, (0, self.pad_w, 0, self.pad_h), mode='constant', value=0)
            H += self.pad_h
            W += self.pad_w

        x = x.view(B * T, self.in_chans, H, W)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x.contiguous()
        x = x.view(B, T, self.num_patches, self.embed_dim)
        return x


# ----------------------------
# 模型组件（轻量化）
# ----------------------------

class MultiModalTransformer(nn.Module):
    def __init__(self, embed_dim=128, depth=2, num_heads=4, mlp_ratio=2., qkv_bias=True, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B, T, N, D = x.shape
        x = x.contiguous()
        x = x.view(B, T * N, D)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.contiguous()
        x = x.view(B, T, N, D)
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, embed_dim=128, aux_dim=64, depth=2, num_heads=4, mlp_ratio=2., qkv_bias=True,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.aux_proj = nn.Linear(aux_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, embed_dim))

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
        B, T, D = x.shape
        aux_proj = self.aux_proj(aux).unsqueeze(1)
        x = torch.cat([x, aux_proj], dim=1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        pos_tokens = torch.zeros_like(x)
        pos_tokens[:, :2] = self.pos_embed
        x = x + pos_tokens
        x = x.contiguous()
        x = self.blocks(x)
        x = self.norm(x)
        return x[:, 0]


class SimplifiedMMSTViT(nn.Module):
    def __init__(self,
                 img_size=(41, 41),
                 patch_size=7,
                 in_chans=1,
                 num_classes=1,
                 embed_dim=128,      # ↓ 减小
                 mm_depth=2,
                 temp_depth=2,
                 num_heads=4,        # ↓ 减小
                 mlp_ratio=2.,       # ↓ 减小
                 aux_dim=64):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        self.mm_transformer = MultiModalTransformer(
            embed_dim=embed_dim,
            depth=mm_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        self.temporal_transformer = TemporalTransformer(
            embed_dim=embed_dim,
            aux_dim=aux_dim,
            depth=temp_depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, img, aux):
        x = self.patch_embed(img)           # (B, 60, 36, 128)
        x = x.contiguous()
        x = self.mm_transformer(x)          # (B, 60, 36, 128)
        x = x.mean(dim=2)                   # (B, 60, 128)
        x = x.contiguous()
        x = self.temporal_transformer(x, aux)  # (B, 128)
        x = self.head(x)
        return x.squeeze(-1)


# ----------------------------
# 主函数（使用小 batch）
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

    # ⚠️ 关键：time_stride=6 → 360 → 60 frames
    train_dataset = MMSTViTDataset(train_meta, soil_climate, is_train=True, time_stride=6)
    val_dataset = MMSTViTDataset(val_meta, soil_climate, is_train=True, aux_scaler=train_dataset.aux_scaler, time_stride=6)
    test_dataset = MMSTViTDataset(test_df, soil_climate, is_train=False, aux_scaler=train_dataset.aux_scaler, time_stride=6)

    # ⚠️ 关键：batch_size=4
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

    aux_dim = train_dataset.aux_feats.shape[1]
    model = SimplifiedMMSTViT(
        img_size=(41, 41),
        patch_size=7,
        in_chans=1,
        embed_dim=128,
        mm_depth=2,
        temp_depth=2,
        num_heads=4,
        mlp_ratio=2.,
        aux_dim=aux_dim
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
    print("Submission file saved.")


if __name__ == "__main__":
    main()
