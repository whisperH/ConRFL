import logging

import torch

from reid.models.aux_layers.grouping import GroupingUnit
from reid.models.gem_pool import GeneralizedMeanPoolingP
from reid.models.layers import *
from reid.utils.my_tools import get_pseudo_features
logger = logging.getLogger(__name__)
model_urls = {
    '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}



class MetaNon_local(nn.Module):
    def __init__(self, in_channels, norm, reduc_ratio=2):
        super(MetaNon_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // reduc_ratio

        self.g = MetaConv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            MetaBatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = MetaConv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = MetaConv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

class Non_local(nn.Module):
    def __init__(self, in_channels, bn_norm, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = in_channels // reduc_ratio

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            BatchNorm(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
                :param x: (b, t, h, w)
                :return x: (b, t, h, w)
        """
        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x
        return z

# Bottleneck of standard ResNet50/101, with kernel size equal to 1
class MetaBottleneck1x1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MetaBottleneck1x1, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = MetaConv2d(planes, planes, kernel_size=1, stride=stride,
                                padding=0, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# Bottleneck of standard ResNet50/101, with kernel size equal to 1
class Bottleneck1x1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1x1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bn_norm, with_ibn=False, with_se=False,
                 stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = MetaConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = MetaBatchNorm2d(planes)

        self.conv2 = MetaConv2d(planes, planes, kernel_size=3, stride=stride,
                                padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(planes)
        self.conv3 = MetaConv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = MetaBatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class ResNetPart(nn.Module):
    def __init__(
            self, last_stride, bn_norm, with_ibn, with_se, with_nl,
            block, num_class, layers, non_layers, part_dim, num_parts, args):
        self.inplanes = 64
        super().__init__()
        self.use_TSBN = args.use_TSBN
        self.conv1 = MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_se=with_se)

        # fmt: off
        if with_nl:
            self._build_nonlocal(layers, non_layers, bn_norm)
        else:
            self.NL_1_idx = self.NL_2_idx = self.NL_3_idx = self.NL_4_idx = []
        # fmt: on

        self.pooling_layer = GeneralizedMeanPoolingP(3)

        #head
        self.bottleneck = MetaBatchNorm2d(2048)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        self.classifier = MetaLinear(512*block.expansion, num_class, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        # coseg head
        self.num_parts = num_parts
        self.part_dim = part_dim

        self.grouping = GroupingUnit(512*block.expansion, self.num_parts)
        self.grouping.reset_parameters(init_weight=None, init_smooth_factor=None)

        # post-processing bottleneck block for the region features
        self.post_block = nn.Sequential(
            Bottleneck1x1(512*block.expansion, 512, stride=1, downsample=nn.Sequential(
                MetaConv2d(512*block.expansion, 512*block.expansion, kernel_size=1, stride=1, bias=False),
                MetaBatchNorm2d(512*block.expansion))),
            Bottleneck1x1(512*block.expansion, 512, stride=1),
            Bottleneck1x1(512*block.expansion, 512, stride=1),
            Bottleneck1x1(512*block.expansion, 512, stride=1),
        )
        self.decrease_dim_block = nn.Sequential(  # pcb
            MetaConv2d(512*block.expansion, self.part_dim, 1, 1, bias=False),  #
            MetaBatchNorm2d(self.part_dim),
            nn.ReLU(inplace=True))
        # the final batchnorm
        # self.groupingbn = nn.BatchNorm2d(512 * 4)
        feat_dim_part = self.part_dim * self.num_parts
        self.bottleneck_part = MetaBatchNorm2d(feat_dim_part)

        self.classifier_part = MetaLinear(feat_dim_part, num_class, bias=False)
        # self.classifier_part = CircleSoftmax(feat_dim_part, num_class)
        # coseg head
        if self.use_TSBN:
            self.task_specific_batch_norm = nn.ModuleList(MetaBatchNorm2d(512*block.expansion) for _ in range(args.train_domain_num))

            for bn in self.task_specific_batch_norm:
                bn.bias.requires_grad_(False)
                nn.init.constant_(bn.weight, 1)
                nn.init.constant_(bn.bias, 0)

        self.random_init()

    def _make_layer(self, block, planes, blocks, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                MetaConv2d(self.inplanes, planes * block.expansion,
                           kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_norm, with_ibn, with_se))

        return nn.Sequential(*layers)

    def _build_nonlocal(self, layers, non_layers, bn_norm):
        self.NL_1 = nn.ModuleList(
            [Non_local(256, bn_norm) for _ in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512, bn_norm) for _ in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024, bn_norm) for _ in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048, bn_norm) for _ in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

    def forward(self, x, targets=None, domains=None, training_phase=None, disti=False, fkd=False, **kwargs):
        result = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        NL1_counter = 0
        if len(self.NL_1_idx) == 0:
            self.NL_1_idx = [-1]
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
            if i == self.NL_1_idx[NL1_counter]:
                _, C, H, W = x.shape
                x = self.NL_1[NL1_counter](x)
                NL1_counter += 1
        # Layer 2
        NL2_counter = 0
        if len(self.NL_2_idx) == 0:
            self.NL_2_idx = [-1]
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)
            if i == self.NL_2_idx[NL2_counter]:
                _, C, H, W = x.shape
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0:
            self.NL_3_idx = [-1]
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)
            if i == self.NL_3_idx[NL3_counter]:
                _, C, H, W = x.shape
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0:
            self.NL_4_idx = [-1]
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)
            if i == self.NL_4_idx[NL4_counter]:
                _, C, H, W = x.shape
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        global_feat = self.pooling_layer(x)

        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]

        # inter
        # grouping module upon the feature maps outputed by the backbone
        # region_feature: BS, feat_dim, partnum
        # assign: BS, partnum, height, width
        # grouping_centers: BS, partnum, feat_dim
        print(x.shape)
        region_feature, assign, grouping_centers = self.grouping(x)
        region_feature = region_feature.contiguous().unsqueeze(3)
        # non-linear layers over the region features -- GNN
        region_feature = self.post_block(region_feature)
        part_feat = self.decrease_dim_block(region_feature)
        bn_part_feat = part_feat.contiguous().view(region_feature.size(0), -1)

        cls_outputs = self.classifier(bn_feat)
        cls_outputs_part = self.classifier_part(bn_part_feat)

        result.update({
            'bn_feat': bn_feat,
            'bn_feat_part': bn_part_feat,
            'cls_outputs': cls_outputs,
            'cls_outputs_part': cls_outputs_part,
            "backbone_feat": x,
            'global_feat': global_feat[..., 0, 0],
            "soft_assign": assign,
            "grouping_centers": grouping_centers,
        })
        # both use in model and old model， 蒸馏的时候固定所有域
        if fkd is True:
            if self.use_TSBN:
                fake_feat_list = get_pseudo_features(self.task_specific_batch_norm, training_phase,
                                                     global_feat, domains, unchange=True)
                result.update({
                    'fake_feat_list': fake_feat_list,
                })
            return result


        if self.training is False:
            return result
        else:
            # 不蒸馏的时候，学习本域的信息，其他域freeze
            if self.use_TSBN:
                fake_feat_list = get_pseudo_features(
                    self.task_specific_batch_norm, training_phase, global_feat, domains
                )
                result.update({
                    'fake_feat_list': fake_feat_list,
                })

        return result

    def random_init(self):
        for m in self.modules():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2./ n))
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MetaLinear):
                nn.init.normal_(m.weight, std=0.001)

def build_TMT_coseg_backbone(num_class, depth, pretrain=True, part_dim=256, num_parts=5, args=None):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    depth         = depth
    # fmt: on

    num_blocks_per_stage = {
        '18x': [2, 2, 2, 2],
        '34x': [3, 4, 6, 3],
        '50x': [3, 4, 6, 3],
        '101x': [3, 4, 23, 3],
    }[depth]

    nl_layers_per_stage = {
        '18x': [0, 0, 0, 0],
        '34x': [0, 0, 0, 0],
        '50x': [0, 2, 3, 0],
        '101x': [0, 2, 9, 0]
    }[depth]

    block = {
        '50x': Bottleneck,
        '101x': Bottleneck
    }[depth]

    with_nl = args.get("with_nl", True)
    # last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, non_layers
    model = ResNetPart(
        1, 'BN', False, False, with_nl, block,
        num_class, num_blocks_per_stage, nl_layers_per_stage,
        part_dim=part_dim, num_parts=num_parts, args=args
    )
    if pretrain:
        # cached_file = '/home/wenhang/.cache/torch/checkpoints/resnet50-19c8e357.pth'
        cached_file = args.cache_file
        state_dict = torch.load(cached_file)
        model.load_state_dict(state_dict, strict=False)


    return model