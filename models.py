import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core Requirement 1: BasicBlock
# "BasicBlock: Conv2d → Normalization → ReLU"
# Normalization choisie par défaut : BatchNorm2d (moderne pour CNN classiques)
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer=nn.BatchNorm2d):
        """
        BasicBlock = Conv2d(3x3, padding=1) + Normalization + ReLU
        - norm_layer est paramétrable (BatchNorm2d ou InstanceNorm2d)
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


# ---------------------------------------------------------------------------
# Core Requirement 3: DownBlock
# "DownBlock: one downsampling step (BasicBlock → MaxPool2d)"
# ---------------------------------------------------------------------------
class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer=nn.BatchNorm2d):
        """
        DownBlock = BasicBlock(in_channels → out_channels) puis MaxPool2d(2)
        - Réduit H et W d’un facteur 2
        """
        super().__init__()
        self.block = BasicBlock(in_channels, out_channels, norm_layer)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.pool(x)
        return x


# ---------------------------------------------------------------------------
# Core Requirement 4: UpBlock
# "UpBlock: ConvTranspose2d → BasicBlock"
# ---------------------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_layer=nn.InstanceNorm2d):
        """
        UpBlock = ConvTranspose2d (upsampling x2) puis BasicBlock(out_channels → out_channels)
        - On utilise InstanceNorm2d par défaut (bon choix pour segmentation / génération)
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.block = BasicBlock(out_channels, out_channels, norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.block(x)
        return x


# ===========================================================================
# Task 1: Image Classification
# Goal: Input (B, 3, H, W) → Output (B, 10)
# - Backbone: plusieurs DownBlock
# - Head: nn.AdaptiveAvgPool2d((1, 1)) puis nn.Linear(C, num_classes)
# - Modern Head: on utilise AdaptiveAvgPool2d au lieu de Flatten naïf.
# ===========================================================================
class ImageClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        """
        CNN de classification moderne :
        - Backbone: empilement de DownBlock (BatchNorm2d)
        - Head: AdaptiveAvgPool2d((1, 1)) → flatten → Linear(C, num_classes)
        """
        super().__init__()

        # Canaux intermédiaires (tu peux les ajuster si tu veux)
        channels = [3, 32, 64, 128, 256]

        # Encoder / backbone avec DownBlocks
        self.down1 = DownBlock(channels[0], channels[1], norm_layer=nn.BatchNorm2d)
        self.down2 = DownBlock(channels[1], channels[2], norm_layer=nn.BatchNorm2d)
        self.down3 = DownBlock(channels[2], channels[3], norm_layer=nn.BatchNorm2d)
        self.down4 = DownBlock(channels[3], channels[4], norm_layer=nn.BatchNorm2d)

        # Core Requirement 5: "Modern Head" avec AdaptiveAvgPool2d
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone
        x = self.down1(x)  # (B, 3, H, W) → (B, 32, H/2, W/2)
        x = self.down2(x)  # → (B, 64, H/4, W/4)
        x = self.down3(x)  # → (B, 128, H/8, W/8)
        x = self.down4(x)  # → (B, 256, H/16, W/16)

        # Head moderne
        x = self.global_pool(x)  # → (B, 256, 1, 1)
        x = torch.flatten(x, 1)  # → (B, 256)
        x = self.fc(x)           # → (B, num_classes)
        return x


# ===========================================================================
# Task 2: Image Segmentation
# Goal: (B, 3, 512, 512) → (B, 5, 512, 512)
# - Encoder-Decoder ConvNet
# - Ne pas changer la résolution finale (512x512)
# - Utiliser BasicBlock, DownBlock et UpBlock
# - Dernière couche = Conv2d 1x1 pour passer à num_classes (hors block)
# - Norm layer recommandé: InstanceNorm2d
# ===========================================================================
class ImageSegmenter(nn.Module):
    def __init__(self, num_classes: int = 5):
        """
        Modèle de segmentation type U-Net simplifié :
        - Encoder: DownBlocks (avec InstanceNorm2d)
        - Decoder: UpBlocks (avec InstanceNorm2d)
        - Skip connections simples pour meilleure précision
        - Sortie finale: Conv2d 1x1 pour produire num_classes canaux
        """
        super().__init__()

        # On choisit des canaux modérés pour rester raisonnable
        ch_enc = [3, 32, 64, 128]

        # Encoder avec InstanceNorm2d
        self.enc1 = BasicBlock(ch_enc[0], ch_enc[1], norm_layer=nn.InstanceNorm2d)
        self.pool1 = nn.MaxPool2d(2, 2)  # 512 → 256

        self.enc2 = BasicBlock(ch_enc[1], ch_enc[2], norm_layer=nn.InstanceNorm2d)
        self.pool2 = nn.MaxPool2d(2, 2)  # 256 → 128

        self.enc3 = BasicBlock(ch_enc[2], ch_enc[3], norm_layer=nn.InstanceNorm2d)
        # Pas de pool supplémentaire pour garder une profondeur raisonnable

        # Decoder: UpBlocks remontent la résolution
        self.up2 = UpBlock(ch_enc[3], ch_enc[2], norm_layer=nn.InstanceNorm2d)  # 128 → 256
        self.up1 = UpBlock(ch_enc[2], ch_enc[1], norm_layer=nn.InstanceNorm2d)  # 256 → 512

        # Fusion simple des skip connections (concat) suivi d’un BasicBlock
        self.dec2 = BasicBlock(ch_enc[2] + ch_enc[2], ch_enc[2], norm_layer=nn.InstanceNorm2d)
        self.dec1 = BasicBlock(ch_enc[1] + ch_enc[1], ch_enc[1], norm_layer=nn.InstanceNorm2d)

        # Dernière couche 1x1 hors block (comme demandé)
        self.classifier = nn.Conv2d(ch_enc[1], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)       # (B, 3, 512, 512) → (B, 32, 512, 512)
        x_p1 = self.pool1(x1)   # → (B, 32, 256, 256)

        x2 = self.enc2(x_p1)    # → (B, 64, 256, 256)
        x_p2 = self.pool2(x2)   # → (B, 64, 128, 128)

        x3 = self.enc3(x_p2)    # → (B, 128, 128, 128)

        # Decoder
        u2 = self.up2(x3)       # → (B, 64, 256, 256)
        # Skip connection avec x2
        cat2 = torch.cat([u2, x2], dim=1)  # (B, 64+64, 256, 256)
        d2 = self.dec2(cat2)              # → (B, 64, 256, 256)

        u1 = self.up1(d2)       # → (B, 32, 512, 512)
        # Skip connection avec x1
        cat1 = torch.cat([u1, x1], dim=1)  # (B, 32+32, 512, 512)
        d1 = self.dec1(cat1)              # → (B, 32, 512, 512)

        # Conv 1x1 finale pour num_classes
        out = self.classifier(d1)  # → (B, num_classes, 512, 512)
        return out


# ===========================================================================
# Task 3: Bounding Box Estimation
# Goal: (B, 3, H, W) → (B, 4)
# - Même backbone convolutionnel que le classifieur
# - Head avec AdaptiveAvgPool2d((1, 1)) → Linear(C, 4)
# - Task de régression : pas d’activation finale
# ===========================================================================
class BBoxRegressor(nn.Module):
    def __init__(self, num_coords: int = 4):
        """
        Modèle de régression de bounding box :
        - Backbone identique (structurellement) à celui du classifier
        - Head: AdaptiveAvgPool2d((1, 1)) → Linear(C, num_coords)
        - Pas d’activation finale (régression)
        """
        super().__init__()

        channels = [3, 32, 64, 128, 256]

        # Même idée de backbone que pour ImageClassifier (BatchNorm2d)
        self.down1 = DownBlock(channels[0], channels[1], norm_layer=nn.BatchNorm2d)
        self.down2 = DownBlock(channels[1], channels[2], norm_layer=nn.BatchNorm2d)
        self.down3 = DownBlock(channels[2], channels[3], norm_layer=nn.BatchNorm2d)
        self.down4 = DownBlock(channels[3], channels[4], norm_layer=nn.BatchNorm2d)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[-1], num_coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)

        x = self.global_pool(x)  # → (B, C, 1, 1)
        x = torch.flatten(x, 1)  # → (B, C)
        x = self.fc(x)           # → (B, num_coords)
        # Pas de sigmoid/tanh ici : c’est une régression pure
        return x


# ===========================================================================
# Task 4: Image Generation (Autoencoder)
# Goal: (B, 3, 512, 512) → (B, 3, 512, 512)
# - Encoder: empilement de DownBlocks
# - Decoder: empilement de UpBlocks
# - Sortie finale: torch.sigmoid pour contraindre les pixels dans [0, 1]
# - Norm layer recommandé: InstanceNorm2d
# ===========================================================================
class ImageGenerator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        """
        Autoencoder convolutionnel :
        - Encoder: DownBlocks (InstanceNorm2d)
        - Decoder: UpBlocks (InstanceNorm2d)
        - Sortie: sigmoïde pour pixels entre 0 et 1
        """
        super().__init__()

        # Encoder
        self.enc1 = DownBlock(in_channels, 32, norm_layer=nn.InstanceNorm2d)  # 512 → 256
        self.enc2 = DownBlock(32, 64, norm_layer=nn.InstanceNorm2d)          # 256 → 128
        self.enc3 = DownBlock(64, 128, norm_layer=nn.InstanceNorm2d)         # 128 → 64
        self.enc4 = DownBlock(128, 256, norm_layer=nn.InstanceNorm2d)        # 64 → 32

        # Decoder
        self.up1 = UpBlock(256, 128, norm_layer=nn.InstanceNorm2d)           # 32 → 64
        self.up2 = UpBlock(128, 64, norm_layer=nn.InstanceNorm2d)            # 64 → 128
        self.up3 = UpBlock(64, 32, norm_layer=nn.InstanceNorm2d)             # 128 → 256
        self.up4 = UpBlock(32, out_channels, norm_layer=nn.InstanceNorm2d)   # 256 → 512

        # Pas de BasicBlock ici après la dernière UpBlock, on applique directement sigmoid
        # (on pourrait ajouter un petit BasicBlock si on veut plus de capacité)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # Decoder
        u1 = self.up1(x4)
        u2 = self.up2(u1)
        u3 = self.up3(u2)
        u4 = self.up4(u3)

        # Contraindre la sortie dans [0, 1]
        out = torch.sigmoid(u4)
        return out
