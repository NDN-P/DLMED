"""Net.py - the file that defines the structure of the model (slide: "Class Net").

Multi-task network for one brain MRI slice (1 x H x W, grayscale):

    Feature extraction : ResNet-34 encoder (ImageNet-pretrained backbone from torchvision.models)
    Task 1 (segmentation)   : U-Net decoder + 3x3 conv  -> tumor mask logits  (B, 1, H, W)
    Task 2 (classification) : global average pool + Linear -> tumor type logits (B, 3)

Both heads share the same encoder, so the two tasks are learned together (multi-task learning).
The layer names are the same as in the DLMED repo's model (src/model.py, segmentation_models_pytorch
U-Net with a ResNet-34 encoder and an auxiliary classification head), so its trained weights load directly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

classes = ('meningioma', 'glioma', 'pituitary')


class DecoderBlock(nn.Module):
    """Upsample x2, concatenate the encoder feature map of the same size (skip connection), then 2 x (Conv-BN-ReLU)."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    """U-Net decoder: five blocks, 512 -> 256 -> 128 -> 64 -> 32 -> 16 channels, back to the input size."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            DecoderBlock(512, 256, 256),  # H/32 -> H/16
            DecoderBlock(256, 128, 128),  # H/16 -> H/8
            DecoderBlock(128, 64, 64),    # H/8  -> H/4
            DecoderBlock(64, 64, 32),     # H/4  -> H/2
            DecoderBlock(32, 0, 16),      # H/2  -> H
        ])

    def forward(self, x, skips):
        for i, block in enumerate(self.blocks):
            x = block(x, skips[i] if i < len(skips) else None)
        return x


class Net(nn.Module):
    def __init__(self, num_classes=len(classes), pretrained=True):
        super().__init__()

        # ---- Feature extraction (shared by both tasks): ResNet-34 backbone ----
        self.encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        # MRI is grayscale: replace the 3-channel first conv with a 1-channel one.
        # With pretrained weights the RGB filters are summed, so the ImageNet features are kept.
        rgb_conv = self.encoder.conv1
        self.encoder.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            self.encoder.conv1.weight.data = rgb_conv.weight.data.sum(dim=1, keepdim=True)
        self.encoder.fc = nn.Identity()  # the ImageNet classifier is not used

        # ---- Task 1: segmentation (U-Net decoder + 1-channel output) ----
        self.decoder = Decoder()
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

        # ---- Task 2: classification (on the deepest encoder feature map) ----
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes),
        )

        self._init_new_layers()

    def _init_new_layers(self):
        # Encoder: ImageNet weights. New layers: same initialization as segmentation_models_pytorch.
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for head in (self.segmentation_head, self.classification_head):
            for m in head.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    # Defining how the data flows through the layers (forward pass)
    def forward(self, x):
        e = self.encoder
        f1 = e.relu(e.bn1(e.conv1(x)))   # (B,  64, H/2,  W/2)
        f2 = e.layer1(e.maxpool(f1))     # (B,  64, H/4,  W/4)
        f3 = e.layer2(f2)                # (B, 128, H/8,  W/8)
        f4 = e.layer3(f3)                # (B, 256, H/16, W/16)
        f5 = e.layer4(f4)                # (B, 512, H/32, W/32)

        seg_logits = self.segmentation_head(self.decoder(f5, [f4, f3, f2, f1]))  # (B, 1, H, W)
        cls_logits = self.classification_head(f5)                                # (B, num_classes)
        return seg_logits, cls_logits


if __name__ == '__main__':
    model = Net(pretrained=False)
    print(model)
    seg_logits, cls_logits = model(torch.rand(2, 1, 256, 256))
    print(seg_logits.shape, cls_logits.shape)  # torch.Size([2, 1, 256, 256]) torch.Size([2, 3])
