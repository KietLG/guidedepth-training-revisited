import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    """
    Taken from:
    https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4
    """
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        reduced = max(channel // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = torch.mean(x, dim=[2, 3])
        y = y.view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand(x.shape)


class Spatial_Attention_From_Guide(nn.Module):
    """
    Computes spatial attention map from guidance features.
    """
    def __init__(self, guide_channels):
        super().__init__()
        mid = max(guide_channels // 2, 8)
        self.conv = nn.Sequential(
            nn.Conv2d(guide_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, guide_feat):
        return self.conv(guide_feat)


class Guided_Upsampling_Block(nn.Module):
    def __init__(self, in_features, expand_features, out_features,
                 kernel_size=3, channel_attention=True,
                 spatial_attention=False,
                 guidance_type='full', guide_features=3,
                 skip_features=0, **kwargs):
        super(Guided_Upsampling_Block, self).__init__()

        self.channel_attention = channel_attention
        self.guidance_type = guidance_type
        self.spatial_attention = spatial_attention
        self.skip_features = skip_features

        padding = kernel_size // 2

        self.feature_conv = nn.Sequential(
            nn.Conv2d(in_features + skip_features, expand_features, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
            nn.BatchNorm2d(expand_features // 2),
            nn.ReLU(inplace=True)
        )

        if self.guidance_type == 'full':
            self.guide_conv = nn.Sequential(
                nn.Conv2d(guide_features, expand_features, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(expand_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(expand_features, expand_features // 2, kernel_size=1),
                nn.BatchNorm2d(expand_features // 2),
                nn.ReLU(inplace=True)
            )

            comb_features = (expand_features // 2) * 2

            if spatial_attention:
                self.spatial_attn = Spatial_Attention_From_Guide(expand_features // 2)

        else:
            comb_features = expand_features // 2

        self.comb_conv = nn.Sequential(
            nn.Conv2d(comb_features, expand_features, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(expand_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_features, in_features, kernel_size=1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True)
        )

        self.reduce = nn.Conv2d(in_features, out_features, kernel_size=1)

        if channel_attention:
            self.SE_block = SELayer(comb_features, reduction=1)

    def forward(self, guide, depth, skip=None, return_attn=False):
        if skip is not None:
            depth_input = torch.cat([depth, skip], dim=1)
        else:
            depth_input = depth

        x = self.feature_conv(depth_input)
        M_s = None

        if self.guidance_type == 'full':
            y = self.guide_conv(guide)
            if self.spatial_attention:
                M_s = self.spatial_attn(y)
                x = x * M_s
            xy = torch.cat([x, y], dim=1)
        else:
            xy = x

        if self.channel_attention:
            xy = self.SE_block(xy)

        residual = self.comb_conv(xy)
        out = self.reduce(residual + depth)

        if return_attn:
            return out, M_s
        return out


class LocalBinsLite(nn.Module):
    """
    Adapted from Bhat et al. (LocalBins, ECCV 2022) using lightweight convs.
    Predicts N_BINS local bin widths per pixel + softmax probabilities.
    Outputs continuous depth expected value map.
    """
    def __init__(self, in_channels, n_bins=16, min_depth=0.1, max_depth=10.0):
        super().__init__()
        self.n_bins = n_bins
        self.min_depth, self.max_depth = min_depth, max_depth
        self.bin_predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, n_bins, kernel_size=1)
        )
        self.prob_predictor = nn.Sequential(
            nn.Conv2d(in_channels, n_bins, kernel_size=3, padding=1)
        )

    def forward(self, feat):
        bin_widths = F.softmax(self.bin_predictor(feat), dim=1)
        bin_edges = torch.cumsum(bin_widths, dim=1) * (self.max_depth - self.min_depth) + self.min_depth
        bin_centers = bin_edges - bin_widths * (self.max_depth - self.min_depth) / 2.0
        probs = F.softmax(self.prob_predictor(feat), dim=1)
        depth = torch.sum(probs * bin_centers, dim=1, keepdim=True)
        return depth
