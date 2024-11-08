import logging
import math

from reid.models.gem_pool import GeneralizedMeanPoolingP
from reid.models.layers import *
from reid.utils.my_tools import get_pseudo_features
logger = logging.getLogger(__name__)
model_urls = {
    '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


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

class ResNet(nn.Module):
    def __init__(self, last_stride, bn_norm, with_ibn, with_se,with_nl,
                 block, num_class, layers, non_layers, args):
        self.inplanes = 64
        super().__init__()
        self.conv1 = MetaConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = MetaBatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        self.layer1 = self._make_layer(block, 64, layers[0], 1, bn_norm, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], 2, bn_norm, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], 2, bn_norm, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], last_stride, bn_norm, with_se=with_se)

        #head
        self.bottleneck = MetaBatchNorm2d(2048)
        self.bottleneck.bias.requires_grad_(False)
        nn.init.constant_(self.bottleneck.weight, 1)
        nn.init.constant_(self.bottleneck.bias, 0)

        self.pooling_layer = GeneralizedMeanPoolingP(3)

        self.classifier = MetaLinear(512*block.expansion, num_class, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

        self.task_specific_batch_norm = nn.ModuleList(MetaBatchNorm2d(512*block.expansion) for _ in range(5))

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

    def forward(self, x, targets=None, domains=None, training_phase=None, disti=False, fkd=False, **kwargs):
        result = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layer 1
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        global_feat = self.pooling_layer(x)

        bn_feat = self.bottleneck(global_feat)
        bn_feat = bn_feat[..., 0, 0]


        cls_outputs = self.classifier(bn_feat)

        result.update({
            'bn_feat': bn_feat,
            'cls_outputs': cls_outputs,
            "backbone_feat": x,
            'global_feat': global_feat[..., 0, 0],
        })
        # both use in model and old model
        if fkd is True:
            fake_feat_list = get_pseudo_features(self.task_specific_batch_norm, training_phase,
                                                 global_feat, domains, unchange=True)
            result.update({
                'fake_feat_list': fake_feat_list,
            })

        if self.training is False:
            return result
        else:
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

def build_resnet_backbone(num_class, depth, pretrain=True, args=None):
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
    with_nl = False
    model = ResNet(1, 'BN', False, False, with_nl, block,
                   num_class, num_blocks_per_stage, nl_layers_per_stage, args=args)
    if pretrain:
        cached_file = args.cache_file
        state_dict = torch.load(cached_file)
        model.load_state_dict(state_dict, strict=False)

    return model