import torch
from torch import nn

from modules.conv import conv, conv_dw, conv_dw_no_bn
from torch.ao.quantization import QuantStub, DeQuantStub

class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x):
        x = self.align(x)
        x = self.conv(self.skip_add.add(x, self.trunk(x)))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )
        self.skip_add = torch.ao.nn.quantized.FloatFunctional()

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return self.skip_add.add(initial_features, trunk_features)


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.is_mixed = False
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)
        self.cat_op = torch.ao.nn.quantized.FloatFunctional()

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, backbone_features):
        if self.is_mixed:
            backbone_features = self.model[0](backbone_features)
            backbone_features = self.model[1](backbone_features)
            backbone_features = self.model[2](backbone_features).float()
            backbone_features = self.quant(backbone_features)
            for i in range(3, len(self.model)):
                backbone_features = self.model[i](backbone_features)
        else:
            backbone_features = self.quant(backbone_features)
            backbone_features = self.model(backbone_features)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(self.cat_op.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        final_results = [stages_output[-2], stages_output[-1]]

        return [self.dequant(out) for out in final_results]

    def fuse_model(self):
        from torch.ao.quantization import fuse_modules
        for m in self.modules():
            if type(m) == nn.Sequential:
                # Handle conv_dw: [Conv, BN, ReLU, Conv, BN, ReLU]
                if len(m) == 6:
                    if type(m[0]) == nn.Conv2d and type(m[1]) == nn.BatchNorm2d:
                        fuse_modules(m, ['0', '1', '2'], inplace=True)
                    if type(m[3]) == nn.Conv2d and type(m[4]) == nn.BatchNorm2d:
                        fuse_modules(m, ['3', '4', '5'], inplace=True)
                
                # Handle standard conv: [Conv, BN, ReLU]
                elif len(m) == 3:
                    if type(m[0]) == nn.Conv2d and type(m[1]) == nn.BatchNorm2d:
                        fuse_modules(m, ['0', '1', '2'], inplace=True)
                
                # Handle Conv + BN (no ReLU)
                elif len(m) == 2:
                    if type(m[0]) == nn.Conv2d and type(m[1]) == nn.BatchNorm2d:
                        fuse_modules(m, ['0', '1'], inplace=True)