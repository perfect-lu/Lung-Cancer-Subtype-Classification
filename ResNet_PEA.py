import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Position Embedding Module for 3D ----
class PositionEmbedding3D(nn.Module):
    def __init__(self, in_channels, reduction=16, top_k=0.1):
        super(PositionEmbedding3D, self).__init__()
        self.in_channels = in_channels
        self.top_k_ratio = top_k

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels)
        )

        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=1)
        self.saliency_fc = nn.Conv3d(in_channels, 1, kernel_size=1)

    def forward(self, F):
        B, C, D, H, W = F.shape
        spatial_size = D * H * W

        # Channel attention
        F_flat = F.view(B, C, -1)
        top_k = int(self.top_k_ratio * spatial_size)
        topk_vals, _ = torch.topk(F_flat, top_k, dim=2)
        channel_descriptor = topk_vals.mean(dim=2)
        Wc = torch.sigmoid(self.mlp(channel_descriptor)).view(B, C, 1, 1, 1)
        Fc = F * Wc

        # Spatial attention
        Sm = torch.sigmoid(self.saliency_fc(F))
        F_avg = torch.mean(F, dim=1, keepdim=True)
        F_max, _ = torch.max(F, dim=1, keepdim=True)
        Ms = torch.sigmoid(self.spatial_conv(torch.cat([F_avg, F_max], dim=1)))
        Ws = Sm * Ms
        Fs = F * Ws

        return Fc + Fs

# ---- BasicBlock & Bottleneck (Unchanged) ----
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(channels * self.expansion)
            )
    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, use_pe=True):
        super().__init__()

        self.use_pe = use_pe
        out_channels = channels * self.expansion

        self.conv1 = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        self.conv3 = nn.Conv3d(channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

        # ✅ Add Position Embedding here
        if self.use_pe:
            self.pe = PositionEmbedding3D(in_channels=out_channels)

    def forward(self, x):
        identity = self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        # ✅ Apply PE before residual addition
        if self.use_pe:
            out = self.pe(out)

        out += identity
        return self.relu(out)


# # ---- ResNet3D with PE ----
# class ResNet3D(nn.Module):
#     def __init__(self, block, layers, num_classes):
#         super().__init__()
#         self.in_channels = 64

#         self.conv1 = nn.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False)
#         self.bn1 = nn.BatchNorm3d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.max_pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

#         # ---- Add PE Module after layer4 ----
#         self.pe = PositionEmbedding3D(in_channels=512 * block.expansion)

#         self.avg_pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

#         # init
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#             elif isinstance(m, nn.BatchNorm3d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(self, block, channels, blocks, stride=1):
#         layers = [block(self.in_channels, channels, stride)]
#         self.in_channels = channels * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.in_channels, channels))
#         return nn.Sequential(*layers)

#     def forward(self, x, debug=False):
#         out = self.relu(self.bn1(self.conv1(x)))
#         if debug: print("After conv1:", out.shape)

#         out = self.max_pool(out)
#         if debug: print("After max_pool:", out.shape)

#         out = self.layer1(out)
#         if debug: print("After layer1:", out.shape)

#         out = self.layer2(out)
#         if debug: print("After layer2:", out.shape)

#         out = self.layer3(out)
#         if debug: print("After layer3:", out.shape)

#         out = self.layer4(out)
#         if debug: print("After layer4:", out.shape)

#         out = self.pe(out)
#         if debug: print("After PE:", out.shape)

#         out = self.avg_pool(out)
#         if debug: print("After avg_pool:", out.shape)

#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out

# # ---- Model Variants ----
# def resnet10(num_classes):
#     return ResNet3D(Bottleneck, [1, 1, 1, 1], num_classes=num_classes)

# def resnet18(num_classes):
#     return ResNet3D(Bottleneck, [2, 2, 2, 2], num_classes=num_classes)

# def resnet26(num_classes):
#     return ResNet3D(Bottleneck, [1, 2, 4, 1], num_classes=num_classes)

# def resnet34(num_classes):
#     return ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

# def resnet50(num_classes):
#     return ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


# def resnet101(num_classes):
#     return ResNet3D(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)


# def resnet152(num_classes):
#     return ResNet3D(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)

# def resnet200(num_classes):
#     return ResNet3D(Bottleneck, [3, 24, 36, 3], num_classes=num_classes)


class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=None, extract_feature=False):
        super().__init__()
        self.in_channels = 64
        self.extract_feature = extract_feature  # 是否只提取特征

        self.conv1 = nn.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=(1,3,3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.pe = PositionEmbedding3D(in_channels=512 * block.expansion)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        if not self.extract_feature:
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, channels, blocks, stride=1):
        layers = [block(self.in_channels, channels, stride)]
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, channels))
        return nn.Sequential(*layers)

    def forward(self, x, debug=False):
        out = self.relu(self.bn1(self.conv1(x)))
        if debug: print("After conv1:", out.shape)

        out = self.max_pool(out)
        if debug: print("After max_pool:", out.shape)

        out = self.layer1(out)
        if debug: print("After layer1:", out.shape)

        out = self.layer2(out)
        if debug: print("After layer2:", out.shape)

        out = self.layer3(out)
        if debug: print("After layer3:", out.shape)

        out = self.layer4(out)
        if debug: print("After layer4:", out.shape)

        out = self.pe(out)
        if debug: print("After PE:", out.shape)

        out = self.avg_pool(out)
        if debug: print("After avg_pool:", out.shape)

        out = out.view(out.size(0), -1)

        if self.extract_feature:
            # 只返回特征向量，不做分类
            return out
        else:
            out = self.fc(out)
            return out

# 只提取特征的resnet50函数
def resnet50_feature_extractor():
    # num_classes不需要传，因为不做分类
    return ResNet3D(Bottleneck, [3, 4, 6, 3], num_classes=None, extract_feature=True)

if __name__ == '__main__':
    model = resnet18(num_classes=3)
    x = torch.randn(8, 1, 128, 128, 128)
    y = model(x, debug=True)
    print("Output shape:", y.shape)
