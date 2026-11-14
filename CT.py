# ct_pea.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ConvBlock: 3×3×3 Conv + BN + ReLU (or larger kernels for multi-scale)
class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

# Position Embedding 3D (PEA)
class PositionEmbedding3D(nn.Module):
    def __init__(self, in_channels, reduction=16, top_k=0.1):
        super().__init__()
        self.top_k_ratio = top_k
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels//reduction, in_channels)
        )
        self.saliency_fc  = nn.Conv3d(in_channels, 1, kernel_size=1)
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=1)

    def forward(self, x):
        B, C, D, H, W = x.shape
        N = D * H * W
        # Channel attention
        xf = x.view(B, C, -1)
        k = max(1, int(self.top_k_ratio * N))
        topk, _ = torch.topk(xf, k, dim=2)
        desc = topk.mean(dim=2)
        wc = torch.sigmoid(self.mlp(desc)).view(B, C, 1, 1, 1)
        xc = x * wc
        # Spatial attention
        sal = torch.sigmoid(self.saliency_fc(x))
        x_avg = x.mean(dim=1, keepdim=True)
        x_max, _ = x.max(dim=1, keepdim=True)
        m = torch.sigmoid(self.spatial_conv(torch.cat([x_avg, x_max], dim=1)))
        xs = x * (sal * m)
        return xc + xs

# DDCBlock (Placeholder for Central Difference Convolution)
class DDCBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # TODO: Replace with actual CentralDifferenceConv3d if available
        self.cd_conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.branch = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.cd_conv(x) + self.branch(x)

# Bottleneck Block with PEA for ResNet50
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, ch, stride=1, use_pe=True):
        super().__init__()
        outc = ch * self.expansion
        self.conv1 = nn.Conv3d(in_ch, ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(ch)
        self.conv2 = nn.Conv3d(ch, ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(ch)
        self.conv3 = nn.Conv3d(ch, outc, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(outc)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != outc:
            self.down = nn.Sequential(
                nn.Conv3d(in_ch, outc, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(outc)
            )
        else:
            self.down = nn.Identity()
        if use_pe:
            self.pe = PositionEmbedding3D(in_channels=outc)

    def forward(self, x):
        idt = self.down(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if hasattr(self, 'pe'):
            out = self.pe(out)
        out += idt
        return self.relu(out)

# ResNet50 + PEA
class ResNetPEA(nn.Module):
    def __init__(self, block, layers, in_channels=1):
        super().__init__()
        self.in_ch = 64
        # Stem: Matches ResNet50 architecture
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        # ResNet50 layers: [3, 4, 6, 3]
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.pe = PositionEmbedding3D(in_channels=512 * block.expansion)

    def _make_layer(self, block, ch, blocks, stride):
        layers = [block(self.in_ch, ch, stride)]
        self.in_ch = ch * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_ch, ch))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)  # [B, 512, ...]
        f3 = self.layer3(f2)  # [B, 1024, ...]
        f4 = self.layer4(f3)  # [B, 2048, ...]
        f4 = self.pe(f4)
        return f2, f3, f4

# RoughBranch: Three heads for binary classification
class RoughBranch(nn.Module):
    def __init__(self, in_chs, base_ch=64):
        super().__init__()
        Cin = sum(in_chs)  # 512 + 1024 + 2048 = 3584
        self.ddc = DDCBlock(Cin, base_ch)
        self.p1 = ConvBlock(base_ch, base_ch)
        self.p2 = nn.Sequential(ConvBlock(base_ch, base_ch), ConvBlock(base_ch, base_ch))
        self.p3 = nn.Sequential(ConvBlock(base_ch, base_ch), ConvBlock(base_ch, base_ch), ConvBlock(base_ch, base_ch))
        self.fc1 = nn.Linear(base_ch, 2)
        self.fc2 = nn.Linear(base_ch, 2)
        self.fc3 = nn.Linear(base_ch, 2)

    def forward(self, f2, f3, f4):
        # Align spatial dimensions
        _, _, d2, h2, w2 = f2.shape
        _, _, d3, h3, w3 = f3.shape
        _, _, d4, h4, w4 = f4.shape
        target_d, target_h, target_w = max(d2, d3, d4), max(h2, h3, h4), max(w2, w3, w4)
        f2 = F.interpolate(f2, size=(target_d, target_h, target_w), mode='trilinear', align_corners=False)
        f3 = F.interpolate(f3, size=(target_d, target_h, target_w), mode='trilinear', align_corners=False)
        f4 = F.interpolate(f4, size=(target_d, target_h, target_w), mode='trilinear', align_corners=False)
        x = torch.cat([f2, f3, f4], dim=1)  # [B, 3584, ...]
        x = self.ddc(x)
        def head(xi, path, fc):
            y = path(xi)
            y = F.adaptive_avg_pool3d(y, 1).view(xi.size(0), -1)
            return fc(y)
        return (head(x, self.p1, self.fc1),
                head(x, self.p2, self.fc2),
                head(x, self.p3, self.fc3))

# FineBranch: Three-class classification
class FineBranch(nn.Module):
    def __init__(self, in_chs, base_ch=64, hidden=128, n_classes=3):
        super().__init__()
        self.d2 = DDCBlock(in_chs[0], base_ch)
        self.d3 = DDCBlock(in_chs[1], base_ch)
        self.d4 = DDCBlock(in_chs[2], base_ch)
        self.c2 = ConvBlock(base_ch, base_ch)
        self.c3 = ConvBlock(base_ch, base_ch)
        self.c4 = ConvBlock(base_ch, base_ch)
        self.fc1 = nn.Linear(base_ch * 3, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, f2, f3, f4):
        def proc(fi, ddc, cb):
            y = cb(ddc(fi))
            y = F.adaptive_avg_pool3d(y, 1).view(fi.size(0), -1)
            return y
        v2 = proc(f2, self.d2, self.c2)
        v3 = proc(f3, self.d3, self.c3)
        v4 = proc(f4, self.d4, self.c4)
        cat = torch.cat([v2, v3, v4], dim=1)
        h = F.relu(self.fc1(cat))
        return self.fc2(h)

# CTNet: Full network
class CTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Multi-scale separation
        self.conv3 = ConvBlock(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBlock(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv7 = ConvBlock(1, 32, kernel_size=7, stride=1, padding=3)
        # ResNet50 + PEA
        self.backbone = ResNetPEA(Bottleneck, [3, 4, 6, 3], in_channels=96)
        # Branches
        C2 = 128 * Bottleneck.expansion  # 512
        C3 = 256 * Bottleneck.expansion  # 1024
        C4 = 512 * Bottleneck.expansion  # 2048
        self.rough = RoughBranch([C2, C3, C4], base_ch=64)
        self.fine = FineBranch([C2, C3, C4], base_ch=64, hidden=128, n_classes=3)

    def forward(self, x):
        # Multi-scale convolutions
        f_small = self.conv3(x)   # [B, 32, D, H, W]
        f_medium = self.conv5(x)  # [B, 32, D, H, W]
        f_large = self.conv7(x)   # [B, 32, D, H, W]
        # Concatenate along channel dimension
        f_multi = torch.cat([f_small, f_medium, f_large], dim=1)  # [B, 96, D, H, W]
        # Deep feature extraction with ResNet50+PEA
        f2, f3, f4 = self.backbone(f_multi)  # [B, 512, ...], [B, 1024, ...], [B, 2048, ...]
        # Rough and Fine branches
        out_r = self.rough(f2, f3, f4)  # Tuple of 3 × [B, 2]
        out_f = self.fine(f2, f3, f4)   # [B, 3]
        return out_r, out_f

    @staticmethod
    def loss_cdl(rough_logits, fine_logits, T=1.0):
        with torch.no_grad():
            p_f = F.softmax(fine_logits / T, dim=1)[:, :2]
        loss = 0
        for o in rough_logits:
            loss += F.kl_div(F.log_softmax(o / T, dim=1), p_f, reduction='batchmean') * (T * T)
        return loss / len(rough_logits)

if __name__ == "__main__":
    net = CTNet().cuda()
    x = torch.randn(2, 1, 16, 112, 112).cuda()
    r, f = net(x)
    print([o.shape for o in r], f.shape)
    loss = CTNet.loss_cdl(r, f, T=2.0)
    print("CDL Loss:", loss.item())
