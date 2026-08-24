import torch
import torch.nn as nn
import torch.nn.functional as F

from model.DDRNet_23_slim import DualResNet_Backbone
from model.modules import Guided_Upsampling_Block, SELayer, LocalBinsLite

_SKIP_RAW_CHANNELS = {
    'none': 0,
    'single': 32,
}

_SKIP_REDUCED_CHANNELS = 16


class GuideDepth(nn.Module):
    def __init__(self, 
                 pretrained=True,
                 up_features=[64, 32, 16], 
                 inner_features=[64, 32, 16],
                 spatial_attention=False,
                 skip_connection='single',
                 deep_supervision_enable=True,
                 localbins_enable=True,
                 localbins_nbins=16,
                 **kwargs):
        super(GuideDepth, self).__init__()

        self.skip_connection = skip_connection
        self.deep_supervision_enable = deep_supervision_enable
        self.localbins_enable = localbins_enable

        if self.localbins_enable:
            self.localbins_head = LocalBinsLite(in_channels=up_features[2], n_bins=localbins_nbins)

        self.feature_extractor = DualResNet_Backbone(
            pretrained=pretrained, 
            features=up_features[0],
            skip_mode=skip_connection
        )

        skip_raw_channels = _SKIP_RAW_CHANNELS.get(skip_connection, 0)
        if skip_raw_channels > 0:
            self.skip_conv = nn.Conv2d(skip_raw_channels, _SKIP_REDUCED_CHANNELS, kernel_size=1)
            up1_skip_features = _SKIP_REDUCED_CHANNELS
        else:
            self.skip_conv = None
            up1_skip_features = 0

        self.up_1 = Guided_Upsampling_Block(
            in_features=up_features[0],
            expand_features=inner_features[0],
            out_features=up_features[1],
            kernel_size=3,
            channel_attention=True,
            spatial_attention=spatial_attention,
            guide_features=3,
            guidance_type="full",
            skip_features=up1_skip_features
        )

        self.up_2 = Guided_Upsampling_Block(
            in_features=up_features[1],
            expand_features=inner_features[1],
            out_features=up_features[2],
            kernel_size=3,
            channel_attention=True,
            spatial_attention=spatial_attention,
            guide_features=3,
            guidance_type="full",
            skip_features=0
        )

        self.up_3 = Guided_Upsampling_Block(
            in_features=up_features[2],
            expand_features=inner_features[2],
            out_features=1,
            kernel_size=3,
            channel_attention=True,
            spatial_attention=spatial_attention,
            guide_features=3,
            guidance_type="full",
            skip_features=0
        )

        if self.deep_supervision_enable:
            self.aux_head_1 = nn.Conv2d(up_features[1], 1, kernel_size=1)
            self.aux_head_2 = nn.Conv2d(up_features[2], 1, kernel_size=1)

    def forward(self, x):
        if self.skip_connection != 'none':
            y, skip = self.feature_extractor(x)
        else:
            y = self.feature_extractor(x)
            skip = None

        x_half = F.interpolate(x, scale_factor=.5)
        x_quarter = F.interpolate(x, scale_factor=.25)

        if skip is not None and self.skip_conv is not None:
            skip = self.skip_conv(skip)

        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.up_1(x_quarter, y, skip=skip)
        y_ds1 = y

        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        y = self.up_2(x_half, y)
        y_ds2 = y

        y = F.interpolate(y, scale_factor=2, mode='bilinear')
        depth_out = self.up_3(x, y)

        if self.training:
            aux_outputs = {}
            if self.deep_supervision_enable:
                aux_outputs['ds_1'] = self.aux_head_1(y_ds1)
                aux_outputs['ds_2'] = self.aux_head_2(y_ds2)
            if self.localbins_enable and y_ds2 is not None:
                aux_outputs['localbins_depth'] = self.localbins_head(y_ds2)
            return depth_out, aux_outputs

        return depth_out
