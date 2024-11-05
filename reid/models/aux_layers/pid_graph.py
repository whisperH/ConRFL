import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import (ModuleList)
import copy
from torch import einsum
from reid.models.layers import MetaConv2d, MetaBatchNorm2d
from reid.utils.comm import get_world_size, GatherLayer, concat_all_gather
from reid.models.aux_layers.grouping import GroupingUnit, MGTBlock, MGABlock
def process_feat(adj, bn_feat, act):
    if act == None:
        node_weight = adj
    elif act == 'sigmoid':
        node_weight = F.sigmoid(adj)
    elif act == 'softmax':
        node_weight = F.softmax(adj, dim=1)
    else:
        raise 'unknown activate in process_feat'
    post_feat = torch.mm(node_weight, bn_feat)
    return post_feat


class InnerProductDecoder(torch.nn.Module):
    def __init__(self, sigma=1, act='softmax', self_match=True):
        super(InnerProductDecoder, self).__init__()
        self.sigma = sigma
        self.act = act
        self.self_match = self_match

    def forward(self, query_feat, gallery_feat, targets=None):
        adj = self.adj_compute(query_feat, gallery_feat)
        if targets is not None:
            target1 = targets.unsqueeze(1)
            mask = (target1 == target1.t())
            pair_labels = mask.float()
        else:
            pair_labels = None
        return adj, pair_labels

    def adj_compute(self, q_feat, g_feat, **kwargs):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """

        q_emb_org_norm = torch.norm(q_feat, 2, 1, True).clamp(min=1e-12)
        q_emb_org_norm = torch.div(q_feat, q_emb_org_norm)
        g_emb_org_norm = torch.norm(g_feat, 2, 1, True).clamp(min=1e-12)
        g_emb_org_norm = torch.div(g_feat, g_emb_org_norm)
        W = torch.mm(q_emb_org_norm, g_emb_org_norm.t())
        value = torch.div(W, self.sigma)
        return value

##===========================================================================================##

class TransformerDecoder(torch.nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers=8, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, q_feat, g_feat, **kwargs):
        r"""Pass the inputs through the decoder layer in turn.
        Args:
            q_feat: the sequence to the decoder (required).
            g_feat: the sequence from the last layer of the encoder (required).
        Shape:
            tgt: [q, h, w, d*n], where q is the query length, d is d_model, n is num_layers, and (h, w) is feature map size
            memory: [k, h, w, d*n], where k is the memory length
        """
        q_feat_list = q_feat.chunk(self.num_layers, dim=-1)
        g_feat_list = g_feat.chunk(self.num_layers, dim=-1)
        for i, mod in enumerate(self.layers):
            if i == 0:
                score = mod(q_feat_list[i], g_feat_list[i])
            else:
                score = score + mod(q_feat_list[i], g_feat_list[i])

        if self.norm is not None:
            q, k = score.size()
            score = score.view(-1, 1)
            score = self.norm(score)
            score = score.view(q, k)

        return score

class TransformerDecoderLayer(torch.nn.Module):
    r"""TransformerDecoderLayer is made up of feature matching and feedforward network.
    Args:
        d_model: the number of expected features in the input (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
    Examples::
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, dim_feedforward=2048)
        >>> memory = torch.rand(10, 24, 8, 512)
        >>> tgt = torch.rand(20, 24, 8, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, seq_len, d_model=512, dim_feedforward=2048):
        super(TransformerDecoderLayer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        score_embed = torch.randn(seq_len, seq_len)
        score_embed = score_embed + score_embed.t()
        self.score_embed = torch.nn.Parameter(score_embed.view(1, 1, seq_len, seq_len))
        self.fc0 = torch.nn.Linear(d_model, d_model)
        self.fc1 = torch.nn.Linear(d_model, d_model)
        self.bn1 = torch.nn.BatchNorm1d(1)
        self.fc2 = torch.nn.Linear(self.seq_len, dim_feedforward)
        self.bn2 = torch.nn.BatchNorm1d(dim_feedforward)
        self.relu = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(dim_feedforward, 1)
        self.bn3 = torch.nn.BatchNorm1d(1)

    def forward(self, q_feat, g_feat):
        r"""Pass the inputs through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
        Shape:
            tgt: [q, h, w, d], where q is the query length, d is d_model, and (h, w) is feature map size
            memory: [k, h, w, d], where k is the memory length
        """
        if q_feat.dim() == 4:
            q, h, w, d = q_feat.size()
            assert(h * w == self.seq_len and d == self.d_model)
            k, h, w, d = g_feat.size()
            assert(h * w == self.seq_len and d == self.d_model)
        elif q_feat.dim() == 2:
            q, d = q_feat.size()
            assert(d == self.d_model)
            k, d = g_feat.size()
            assert(d == self.d_model)

        tgt = q_feat.view(q, -1, d)
        memory = g_feat.view(k, -1, d)
        query = self.fc0(tgt)
        key = self.fc0(memory)
        score = einsum('q t d, k s d -> q k s t', query, key) * self.score_embed.sigmoid()
        score = score.reshape(q * k, self.seq_len, self.seq_len)
        score = torch.cat((score.max(dim=1)[0], score.max(dim=2)[0]), dim=-1)
        score = score.view(-1, 1, self.seq_len)
        score = self.bn1(score).view(-1, self.seq_len)

        score = self.fc2(score)
        score = self.bn2(score)
        score = self.relu(score)
        score = self.fc3(score)
        score = score.view(-1, 2).sum(dim=-1, keepdim=True)
        score = self.bn3(score)
        score = score.view(q, k)
        return score

class TransMatch(torch.nn.Module):

    def __init__(self, seq_len, w, h, d_model=512, num_decoder_layers=8, dim_feedforward=2048):
        super().__init__()
        self.seq_len = seq_len
        self.w = w
        self.h = h
        self.d_model = d_model
        self.act = 'softmax'

        self.decoder_layer = TransformerDecoderLayer(seq_len, d_model, dim_feedforward)
        # decoder_norm = torch.nn.BatchNorm1d(1)
        decoder_norm = None
        self.decoder = TransformerDecoder(self.decoder_layer, num_decoder_layers, decoder_norm)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, q_feat, g_feat, targets=None):
        '''
        Args:
            features: bs, feat_dim

        Returns:
            score: bs, bs
        '''
        # For distributed training, gather all features from different process.
        if get_world_size() > 1:
            all_q_feat = torch.cat(GatherLayer.apply(q_feat), dim=0)
            all_g_feat = torch.cat(GatherLayer.apply(g_feat), dim=0)
            if targets is not None:
                all_targets = concat_all_gather(targets)
            else:
                all_targets = targets
        else:
            all_q_feat = q_feat
            all_g_feat = g_feat
            all_targets = targets
        if q_feat.dim() == 4:
            q_feat = all_q_feat.permute(0, 2, 3, 1).contiguous()
            g_feat = all_g_feat.permute(0, 2, 3, 1).contiguous()
        if q_feat.dim() == 2:
            bs = all_q_feat.size(0)
            q_feat = all_q_feat.reshape(bs, self.h, self.w, -1).contiguous()
            g_feat = all_g_feat.reshape(bs, self.h, self.w, -1).contiguous()

        score = self.decoder(q_feat, g_feat)
        if all_targets is not None:
            target1 = all_targets.unsqueeze(1)
            mask = (target1 == target1.t())
            pair_labels = mask.float()
        else:
            pair_labels = None
        return score, pair_labels

##===========================================================================================##


# Bottleneck of standard ResNet50/101, with kernel size equal to 1
class Bottleneck1x1(torch.nn.Module):
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

class TmatcherLayer(torch.nn.Module):
    r"""TransformerDecoderLayer is made up of feature matching and feedforward network.
    Args:
        d_model: the number of expected features in the input (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
    """

    def __init__(self, seq_len, d_model=512, dim_feedforward=2048):
        super(TmatcherLayer, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        score_embed = torch.randn(seq_len, seq_len)
        score_embed = score_embed + score_embed.t()
        self.score_embed = torch.nn.Parameter(score_embed.view(1, 1, seq_len, seq_len))
        self.fc0 = torch.nn.Linear(d_model, d_model)
        self.fc1 = torch.nn.Linear(d_model, d_model)
        self.bn1 = torch.nn.BatchNorm1d(1)
        self.fc2 = torch.nn.Linear(self.seq_len, dim_feedforward)
        self.bn2 = torch.nn.BatchNorm1d(dim_feedforward)
        self.relu = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(dim_feedforward, 1)
        self.bn3 = torch.nn.BatchNorm1d(1)

    def forward(self, q_feat, g_feat):
        r"""Pass the inputs through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
        Shape:
            tgt: [q, h, w, d], where q is the query length, d is d_model, and (h, w) is feature map size
            memory: [k, h, w, d], where k is the memory length
        """
        if q_feat.dim() == 4:
            q, h, w, d = q_feat.size()
            assert(h * w == self.seq_len and d == self.d_model)
            k, h, w, d = g_feat.size()
            assert(h * w == self.seq_len and d == self.d_model)
        elif q_feat.dim() == 2:
            q, d = q_feat.size()
            assert(d == self.d_model)
            k, d = g_feat.size()
            assert(d == self.d_model)
        #

class TmatcherDecoder(torch.nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers=8, norm=None):
        super(TmatcherDecoder, self).__init__()
        self.layers = ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, grouping_reigion, q_feat, **kwargs):
        r"""Pass the inputs through the decoder layer in turn.
        Args:
            q_feat: the sequence to the decoder (required).
            g_feat: the sequence from the last layer of the encoder (required).
        Shape:
            tgt: [q, h, w, d*n], where q is the query length, d is d_model, n is num_layers, and (h, w) is feature map size
            memory: [k, h, w, d*n], where k is the memory length
        """
        for i, mod in enumerate(self.layers):
            if i == 0:
                grouping_reigion = mod(grouping_reigion, q_feat)
            else:
                grouping_reigion = mod(grouping_reigion, q_feat)

        return grouping_reigion

class TMatch(torch.nn.Module):
    def __init__(self, num_parts, w, h, d_model=512,
                 num_decoder_layers=8, dim_feedforward=2048, part_dim=512,**kwargs):
        super().__init__()
        self.seq_len = h * w
        self.w = w
        self.h = h
        self.d_model = d_model
        self.act = 'softmax'
        self.num_parts = num_parts
        self.part_dim = part_dim

        self.weight = torch.nn.Parameter(torch.FloatTensor(num_parts, d_model, 1))   # n * 1024 * 1*1
        # post-processing bottleneck block for the region features
        self.post_block = nn.Sequential(
            Bottleneck1x1(2048, 512, stride=1, downsample=nn.Sequential(
                MetaConv2d(2048, 2048, kernel_size=1, stride=1, bias=False),
                MetaBatchNorm2d(2048))),
            Bottleneck1x1(2048, 512, stride=1),
            Bottleneck1x1(2048, 512, stride=1),
            Bottleneck1x1(2048, 512, stride=1),
        )
        self.decrease_dim_block = nn.Sequential(  # pcb
            MetaConv2d(2048, self.part_dim, 1, 1, bias=False),  #
            MetaBatchNorm2d(self.part_dim),
            nn.ReLU(inplace=True))

        self.decoder_layer = TmatcherLayer(self.seq_len, d_model, dim_feedforward)
        self.grouping = TmatcherDecoder(self.decoder_layer, num_decoder_layers)
        self.reset_parameters()

    def reset_parameters(self, init_weight=None, init_smooth_factor=None):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

        if init_weight is None:
            # msra init
            torch.nn.init.kaiming_normal_(self.weight)
            self.weight.data.clamp_(min=1e-5)
        else:
            # init weight based on clustering
            assert init_weight.shape == (self.num_parts, self.in_channels)
            with torch.no_grad():
                self.weight.copy_(init_weight.unsqueeze(2).unsqueeze(3))
        #
        # # set smooth factor to 0 (before sigmoid)
        # if init_smooth_factor is None:
        #     torch.nn.init.constant_(self.smooth_factor, 0)
        # else:
        #     # init smooth factor based on clustering
        #     assert init_smooth_factor.shape == (self.num_parts,)
        #     with torch.no_grad():
        #         self.smooth_factor.copy_(init_smooth_factor)

    def forward(self, q_feat, targets=None):
        '''
        Args:
            features: bs, feat_dim

        Returns:
            score: bs, bs
        '''
        # For distributed training, gather all features from different process.
        batch_size = q_feat.size(0)
        feat_dim = q_feat.size(1)
        if q_feat.dim() == 4:
            q_feat = q_feat.reshape(batch_size, feat_dim, -1).contiguous()
        if q_feat.dim() == 2:
            q_feat = q_feat.unsqueeze(-1)

        # inter
        # grouping module upon the feature maps outputed by the backbone
        # region_feature: BS, feat_dim, partnum
        # assign: BS, partnum, height, width
        # grouping_centers: BS, partnum, feat_dim
        # 2. generate the grouping centers  # 5 1024 1 1 --> 1 5 1024 --> B 5 1024  # 因为
        grouping_centers = self.weight.view(1, self.num_parts, self.in_channels).expand(
            batch_size, self.num_parts, self.d_model
        ).contiguous()
        region_feature, assign = self.grouping(grouping_centers, q_feat)

        region_feature = region_feature.contiguous().unsqueeze(3)
        # non-linear layers over the region features -- GNN
        region_feature = self.post_block(region_feature)
        part_feat = self.decrease_dim_block(region_feature)
        bn_part_feat = part_feat.contiguous().view(region_feature.size(0), -1)


        if targets is not None:
            target1 = targets.unsqueeze(1)
            mask = (target1 == target1.t())
            pair_labels = mask.float()
        else:
            pair_labels = None
        return bn_part_feat, assign, grouping_centers, pair_labels

##===========================================================================================##

# Multiple Grouping Transformer Aggregation
class MGTA(torch.nn.Module):
    def __init__(self, group_dict, out_channels=2048, **kwargs):
        super().__init__()
        self.in_channels = []
        self.num_head_list = []
        self.attn_dim_list = []
        self.mlp_ratio_list = []
        self.w = []
        self.h = []
        for layer_name, group_info in group_dict.items():
            self.in_channels.append(group_info['feat_dim'])
            self.num_head_list.append(group_info['num_head'])
            self.attn_dim_list.append(group_info['attn_dim'])
            self.mlp_ratio_list.append(group_info['mlp_ratio'])
            self.w.append(group_info['W'])
            self.h.append(group_info['H'])
        self.out_channels = out_channels

        self.merge_feats = nn.ModuleList()
        for i in range(1, len(group_dict)):
            merge_feat = MGTBlock(
                in_channels=self.in_channels[i-1]+self.in_channels[i],
                out_channels=self.in_channels[i],
                attn_dim=self.attn_dim_list[i],
                num_heads=self.num_head_list[i],
                mlp_ratio=self.mlp_ratio_list[i]
            )
            self.merge_feats.append(merge_feat)

        self.merge_assigns = nn.ModuleList()
        for i in range(1, len(group_dict)):
            if (self.w[i-1] == self.w[i]) and (self.h[i-1] == self.h[i]):
                pooling = nn.Identity()
            else:
                pooling = nn.AvgPool2d(2, stride=2)
            merge_assign = MGABlock(
                pooling=pooling,
                in_dim=self.in_channels[i-1],
                score_dim=self.in_channels[i]
            )
            self.merge_assigns.append(merge_assign)

    def forward(self, inputs):
        for idx, (layer_name, group_info) in enumerate(inputs.items()):
            if idx == 0:
                layer1_feat = group_info['iregion_feature'].permute(0, 2, 1)
                layer_assign = group_info['iassign']
            else:
                layer2_feat = group_info['iregion_feature'].permute(0, 2, 1)
                layer2_assign = group_info['iassign']
                concat_feat = torch.cat((layer1_feat, layer2_feat), dim=-1)
                layer_feat = self.merge_feats[idx-1](concat_feat)

                merge_res = self.merge_assigns[idx-1](
                    [layer1_feat, layer2_feat, layer_feat],
                    [layer_assign, layer2_assign]
                )
                layer_assign = merge_res['layer_assign']
                q1_weight = merge_res['q1_weight']
                q2_weight = merge_res['q2_weight']
                layer1_feat = layer_feat.clone()
        return layer_feat, layer_assign

# Multiple Grouping Transformer Aggregation No Assign
class MGTANA(torch.nn.Module):
    def __init__(self, group_dict, out_channels=2048, **kwargs):
        super().__init__()
        self.in_channels = []
        self.num_head_list = []
        self.attn_dim_list = []
        self.mlp_ratio_list = []
        self.w = []
        self.h = []
        for layer_name, group_info in group_dict.items():
            self.in_channels.append(group_info['feat_dim'])
            self.num_head_list.append(group_info['num_head'])
            self.attn_dim_list.append(group_info['attn_dim'])
            self.mlp_ratio_list.append(group_info['mlp_ratio'])
            self.w.append(group_info['W'])
            self.h.append(group_info['H'])
        self.out_channels = out_channels

        self.merge_feats = nn.ModuleList()
        for i in range(1, len(group_dict)):
            merge_feat = MGTBlock(
                in_channels=self.in_channels[i-1]+self.in_channels[i],
                out_channels=self.in_channels[i],
                attn_dim=self.attn_dim_list[i],
                num_heads=self.num_head_list[i],
                mlp_ratio=self.mlp_ratio_list[i]
            )
            self.merge_feats.append(merge_feat)

    def forward(self, inputs):
        for idx, (layer_name, group_info) in enumerate(inputs.items()):
            assign_list = []
            if idx == 0:
                layer1_feat = group_info['iregion_feature'].permute(0, 2, 1)
                layer_assign = group_info['iassign']
            else:
                layer2_feat = group_info['iregion_feature'].permute(0, 2, 1)
                layer_assign = group_info['iassign']
                concat_feat = torch.cat((layer1_feat, layer2_feat), dim=-1)
                layer_feat = self.merge_feats[idx-1](concat_feat)
            assign_list.append(layer_assign)

        return layer_feat, assign_list

# ==================================================================================== #

class MultiGrouping(torch.nn.Module):
    def __init__(self, group_dict, part_dim=1280, grouping_dim=2048, start_merge=3, **kwargs):
        super().__init__()
        self.merge_assign = kwargs.get('merge_assign', True)
        self.start_merge = start_merge
        self.groupings = nn.ModuleList()
        build_info = {}
        for idx, (name, group_info) in enumerate(group_dict.items()):
            if start_merge <= idx:
                igroup_modeule = GroupingUnit(group_info['feat_dim'], group_info['num_parts'])
                igroup_modeule.reset_parameters(init_weight=None, init_smooth_factor=None)
                self.groupings.append(igroup_modeule)
                build_info[name] = group_info

        if self.merge_assign:
            self.stage_merge = MGTA(build_info)
        else:
            self.stage_merge = MGTANA(build_info)

        self.post_block = nn.Sequential(
            Bottleneck1x1(grouping_dim, 512, stride=1, downsample=nn.Sequential(
                MetaConv2d(grouping_dim, 2048, kernel_size=1, stride=1, bias=False),
                MetaBatchNorm2d(2048))),
            Bottleneck1x1(grouping_dim, 512, stride=1),
            Bottleneck1x1(grouping_dim, 512, stride=1),
            Bottleneck1x1(grouping_dim, 512, stride=1),
        )
        self.decrease_dim_block = nn.Sequential(  # pcb
            MetaConv2d(grouping_dim, part_dim, 1, 1, bias=False),  #
            MetaBatchNorm2d(part_dim),
            nn.ReLU(inplace=True))



    def forward(self, backbone_dict):
        # dim list bs-> [64 64 32; 256 64 32; 512 32 16; 1024 16 8; 2048 16 8]
        grouping_result = {}
        for idx, (name, x) in enumerate(backbone_dict.items()):
            if self.start_merge <= idx:
                iregion_feature, iassign, igrouping_centers = self.groupings[idx-self.start_merge](x)
                # region_feature: BS, feat_dim, partnum
                # assign: BS, partnum, height, width
                # grouping_centers: BS, partnum, feat_dim
                grouping_result[name] = {
                    'iregion_feature': iregion_feature,
                    'iassign': iassign,
                    'igrouping_centers': igrouping_centers
                }
        region_feature, assign = self.stage_merge(grouping_result)

        region_feature = region_feature.permute(0,2,1).unsqueeze(3).contiguous()
        # non-linear layers over the region features -- GNN
        region_feature = self.post_block(region_feature)
        part_feat = self.decrease_dim_block(region_feature)
        bn_part_feat = part_feat.contiguous().view(region_feature.size(0), -1)
        return bn_part_feat, assign, None