# import libs
import math
import numpy as np
import torch
import torch.nn as nn
from .transformer import DropPath, trunc_normal_, dot_attention, index_points
from torch.nn.parameter import Parameter

from reid.models.layers import MetaModule
from reid.models.attention import ResidualAttention, DAModule
import torch.nn.functional as F
from reid.utils.my_tools import do_kmeans, simple_transform

class GroupingUnit(nn.Module):

    def __init__(self, in_channels, num_parts):
        super(GroupingUnit, self).__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels
        # params
        self.weight = nn.Parameter(torch.FloatTensor(num_parts, in_channels, 1, 1))  # n * 1024 * 1*1
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))

    def reset_parameters(self, init_weight=None, init_smooth_factor=None):
        if init_weight is None:
            # msra init
            nn.init.kaiming_normal_(self.weight)
            self.weight.data.clamp_(min=1e-5)
        else:
            # init weight based on clustering
            assert init_weight.shape == (self.num_parts, self.in_channels)
            with torch.no_grad():
                self.weight.copy_(init_weight.unsqueeze(2).unsqueeze(3))

        # set smooth factor to 0 (before sigmoid)
        if init_smooth_factor is None:
            nn.init.constant_(self.smooth_factor, 0)
        else:
            # init smooth factor based on clustering
            assert init_smooth_factor.shape == (self.num_parts,)
            with torch.no_grad():
                self.smooth_factor.copy_(init_smooth_factor)

    def forward(self, inputs):
        assert inputs.dim() == 4

        # 0. store input size
        batch_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels == self.in_channels

        # 1. generate the grouping centers  # 5 1024 1 1 --> 1 5 1024 --> B 5 1024  # 因为
        grouping_centers = self.weight.view(1, self.num_parts, self.in_channels).expand(
            batch_size, self.num_parts, self.in_channels
        ).contiguous()

        # 2. compute assignment matrix
        # - d = -\|X - C\|_2 = - X^2 - C^2 + 2 * C^T X
        # C^T X (N * K * H * W)
        inputs_cx = inputs.contiguous().view(batch_size, self.in_channels, input_h * input_w)
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        # X^2 (N * C * H * W) -> (N * 1 * H * W) -> (N * K * H * W)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1)
        # C^2 (K * C * 1 * 1) -> 1 * K * 1 * 1
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3)
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        # expand the smooth term
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(batch_size, -1, input_h, input_w)
        # assignment = softmax(-d/s) (-d must be negative)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1)  # default dim = 1

        # 3. compute residual coding
        # NCHW -> N * C * HW
        x = inputs.contiguous().view(batch_size, self.in_channels, -1)
        # permute the inputs -> N * HW * C
        x = x.permute(0, 2, 1)

        # compute weighted feats N * K * C
        assign = assign.contiguous().view(batch_size, self.num_parts, -1)
        qx = torch.bmm(assign, x)

        # repeat the graph_weights (K * C) -> (N * K * C)
        c = grouping_centers

        # sum of assignment (N * K * 1) -> (N * K * K)
        sum_ass = torch.sum(assign, dim=2, keepdim=True)

        # residual coding N * K * C
        sum_ass = sum_ass.expand(-1, -1, self.in_channels).clamp(min=1e-5)
        sigma = (beta / 2).sqrt()
        out = ((qx / sum_ass) - c) / sigma.unsqueeze(0).unsqueeze(2)

        # 4. prepare outputs
        # we need to memorize the assignment (N * K * H * W)
        assign = assign.contiguous().view(
            batch_size, self.num_parts, input_h, input_w)

        # output features has the size of N * K * C
        outputs = nn.functional.normalize(out, dim=2)  # b 5 1024
        outputs_t = outputs.permute(0, 2, 1)  # b 1024 5

        # generate assignment map for basis for visualization
        return outputs_t, assign, grouping_centers

    # name
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.num_parts) + ')'




class NoisyGroupingUnit(nn.Module):

    def __init__(self, in_channels, num_parts, proposal_parts=15):
        super(NoisyGroupingUnit, self).__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels
        self.proposal_parts = proposal_parts
        # params
        self.noisy_info = nn.AdaptiveAvgPool2d((1, 1))
        self.weight = nn.Parameter(torch.FloatTensor(num_parts, in_channels, 1, 1))  # n * 1024 * 1*1
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))

    def reset_parameters(self, init_weight=None, init_smooth_factor=None):
        if init_weight is None:
            # msra init
            nn.init.kaiming_normal_(self.weight)
            self.weight.data.clamp_(min=1e-5)
        else:
            # init weight based on clustering
            assert init_weight.shape == (self.num_parts, self.in_channels)
            with torch.no_grad():
                self.weight.copy_(init_weight.unsqueeze(2).unsqueeze(3))

        # set smooth factor to 0 (before sigmoid)
        if init_smooth_factor is None:
            nn.init.constant_(self.smooth_factor, 0)
        else:
            # init smooth factor based on clustering
            assert init_smooth_factor.shape == (self.num_parts,)
            with torch.no_grad():
                self.smooth_factor.copy_(init_smooth_factor)

    def forward(self, inputs):
        assert inputs.dim() == 4

        # 0. store input size
        batch_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels == self.in_channels

        # 1. generate the grouping centers  # 5 1024 1 1 --> 1 5 1024 --> B 5 1024  # 因为
        noisy_gc = self.noisy_info(inputs)[:, :, 0, 0].unsqueeze(1).expand(
            batch_size, self.num_parts, self.in_channels
        ).contiguous()
        grouping_centers = self.weight.view(1, self.num_parts, self.in_channels).expand(
            batch_size, self.num_parts, self.in_channels
        ).contiguous() + noisy_gc * 0.01

        # 2. compute assignment matrix
        # - d = -\|X - C\|_2 = - X^2 - C^2 + 2 * C^T X
        # C^T X (N * K * H * W)
        inputs_cx = inputs.contiguous().view(batch_size, self.in_channels, input_h * input_w)
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        # X^2 (N * C * H * W) -> (N * 1 * H * W) -> (N * K * H * W)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1)
        # C^2 (K * C * 1 * 1) -> 1 * K * 1 * 1
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3)
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        # expand the smooth term
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(batch_size, -1, input_h, input_w)
        # assignment = softmax(-d/s) (-d must be negative)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1)  # default dim = 1

        # 3. compute residual coding
        # NCHW -> N * C * HW
        x = inputs.contiguous().view(batch_size, self.in_channels, -1)
        # permute the inputs -> N * HW * C
        x = x.permute(0, 2, 1)

        # compute weighted feats N * K * C
        assign = assign.contiguous().view(batch_size, self.num_parts, -1)
        qx = torch.bmm(assign, x)

        # repeat the graph_weights (K * C) -> (N * K * C)
        c = grouping_centers

        # sum of assignment (N * K * 1) -> (N * K * K)
        sum_ass = torch.sum(assign, dim=2, keepdim=True)

        # residual coding N * K * C
        sum_ass = sum_ass.expand(-1, -1, self.in_channels).clamp(min=1e-5)
        sigma = (beta / 2).sqrt()
        out = ((qx / sum_ass) - c) / sigma.unsqueeze(0).unsqueeze(2)

        # 4. prepare outputs
        # we need to memorize the assignment (N * K * H * W)
        assign = assign.contiguous().view(
            batch_size, self.num_parts, input_h, input_w)

        # output features has the size of N * K * C
        outputs = nn.functional.normalize(out, dim=2)  # b 5 1024
        outputs_t = outputs.permute(0, 2, 1)  # b 1024 5

        # generate assignment map for basis for visualization
        return outputs_t, assign, grouping_centers

    # name
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.num_parts) + ')'
# ==================================== #

class MeanGroupingUnit(nn.Module):
    def __init__(self, in_channels, num_parts):
        super(MeanGroupingUnit, self).__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels

        self.select_part = DAModule(d_model=in_channels, kernel_size=3, H=16, W=8)
        # params
        self.grouping_area = torch.nn.AdaptiveAvgPool2d((num_parts, num_parts))
        self.grouping_center = torch.nn.Linear(num_parts*num_parts, num_parts)
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))

    def reset_parameters(self, init_weight=None, init_smooth_factor=None):
        if init_weight is None:
            # msra init
            nn.init.kaiming_normal_(self.weight)
            self.grouping_center.weight.data.clamp_(min=1e-5)
        else:
            # init weight based on clustering
            assert init_weight.shape == (self.num_parts, self.in_channels)
            with torch.no_grad():
                self.weight.copy_(init_weight.unsqueeze(2).unsqueeze(3))

        # set smooth factor to 0 (before sigmoid)
        if init_smooth_factor is None:
            nn.init.constant_(self.smooth_factor, 0)
        else:
            # init smooth factor based on clustering
            assert init_smooth_factor.shape == (self.num_parts,)
            with torch.no_grad():
                self.smooth_factor.copy_(init_smooth_factor)

    def forward(self, inputs, iteration):
        assert inputs.dim() == 4

        # 0. store input size
        batch_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels == self.in_channels

        # 1. generate the grouping centers  # 5 1024 1 1 --> 1 5 1024 --> B 5 1024  # 因为
        grouping_area = self.grouping_area(inputs).view(inputs.size(0), inputs.size(1), -1)
        grouping_centers = self.grouping_center(grouping_area).permute(0, 2, 1) * 0.01

        # attention map
        inputs = self.select_part(inputs)

        # 2. compute assignment matrix
        # - d = -\|X - C\|_2 = - X^2 - C^2 + 2 * C^T X
        # C^T X (N * K * H * W)
        inputs_cx = inputs.contiguous().view(batch_size, self.in_channels, input_h * input_w)
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        # X^2 (N * C * H * W) -> (N * 1 * H * W) -> (N * K * H * W)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1)
        # C^2 (K * C * 1 * 1) -> 1 * K * 1 * 1
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3)
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        # expand the smooth term
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(batch_size, -1, input_h, input_w)
        # assignment = softmax(-d/s) (-d must be negative)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1)  # default dim = 1

        # 3. compute residual coding
        # NCHW -> N * C * HW
        x = inputs.contiguous().view(batch_size, self.in_channels, -1)
        # permute the inputs -> N * HW * C
        x = x.permute(0, 2, 1)

        # compute weighted feats N * K * C
        assign = assign.contiguous().view(batch_size, self.num_parts, -1)
        qx = torch.bmm(assign, x)

        # repeat the graph_weights (K * C) -> (N * K * C)
        c = grouping_centers

        # sum of assignment (N * K * 1) -> (N * K * K)
        sum_ass = torch.sum(assign, dim=2, keepdim=True)

        # residual coding N * K * C
        sum_ass = sum_ass.expand(-1, -1, self.in_channels).clamp(min=1e-5)
        sigma = (beta / 2).sqrt()
        out = ((qx / sum_ass) - c) / sigma.unsqueeze(0).unsqueeze(2)

        # 4. prepare outputs
        # we need to memorize the assignment (N * K * H * W)
        assign = assign.contiguous().view(
            batch_size, self.num_parts, input_h, input_w)

        # output features has the size of N * K * C
        outputs = nn.functional.normalize(out, dim=2)  # b 5 1024
        outputs_t = outputs.permute(0, 2, 1)  # b 1024 5

        # generate assignment map for basis for visualization
        return outputs_t, assign, grouping_centers

    # name
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.num_parts) + ')'

# ==================================== #

class MatchGroupingUnit(nn.Module):
    def __init__(self, in_channels, num_parts):
        super(MatchGroupingUnit, self).__init__()
        self.num_parts = num_parts
        self.in_channels = in_channels

        self.select_part = DAModule(d_model=in_channels, kernel_size=1, H=16, W=8)
        # params
        self.weight = nn.Parameter(torch.FloatTensor(num_parts, in_channels, 1, 1))  # n * 1024 * 1*1
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))

    def reset_parameters(self, init_weight=None, init_smooth_factor=None):
        if init_weight is None:
            # msra init
            nn.init.kaiming_normal_(self.weight)
            self.weight.data.clamp_(min=1e-5)
        else:
            # init weight based on clustering
            assert init_weight.shape == (self.num_parts, self.in_channels)
            with torch.no_grad():
                self.weight.copy_(init_weight.unsqueeze(2).unsqueeze(3))

        # set smooth factor to 0 (before sigmoid)
        if init_smooth_factor is None:
            nn.init.constant_(self.smooth_factor, 0)
        else:
            # init smooth factor based on clustering
            assert init_smooth_factor.shape == (self.num_parts,)
            with torch.no_grad():
                self.smooth_factor.copy_(init_smooth_factor)

    def forward(self, inputs, iteration):
        assert inputs.dim() == 4

        # 0. store input size
        batch_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels == self.in_channels

        # attention map
        inputs = self.select_part(inputs)
        # 1. generate the grouping centers  # 5 1024 1 1 --> 1 5 1024 --> B 5 1024  # 因为
        grouping_centers = self.weight.view(1, self.num_parts, self.in_channels).expand(
            batch_size, self.num_parts, self.in_channels
        ).contiguous()
        # 2. compute assignment matrix
        # - d = -\|X - C\|_2 = - X^2 - C^2 + 2 * C^T X
        # C^T X (N * K * H * W)
        inputs_cx = inputs.contiguous().view(batch_size, self.in_channels, input_h * input_w)
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        # X^2 (N * C * H * W) -> (N * 1 * H * W) -> (N * K * H * W)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1)
        # C^2 (K * C * 1 * 1) -> 1 * K * 1 * 1
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3)
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        # expand the smooth term
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(batch_size, -1, input_h, input_w)
        # assignment = softmax(-d/s) (-d must be negative)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1)  # default dim = 1

        # 3. compute residual coding
        # NCHW -> N * C * HW
        x = inputs.contiguous().view(batch_size, self.in_channels, -1)
        # permute the inputs -> N * HW * C
        x = x.permute(0, 2, 1)

        # compute weighted feats N * K * C
        assign = assign.contiguous().view(batch_size, self.num_parts, -1)
        qx = torch.bmm(assign, x)

        # repeat the graph_weights (K * C) -> (N * K * C)
        c = grouping_centers

        # sum of assignment (N * K * 1) -> (N * K * K)
        sum_ass = torch.sum(assign, dim=2, keepdim=True)

        # residual coding N * K * C
        sum_ass = sum_ass.expand(-1, -1, self.in_channels).clamp(min=1e-5)
        sigma = (beta / 2).sqrt()
        out = ((qx / sum_ass) - c) / sigma.unsqueeze(0).unsqueeze(2)

        # 4. prepare outputs
        # we need to memorize the assignment (N * K * H * W)
        assign = assign.contiguous().view(
            batch_size, self.num_parts, input_h, input_w)

        # output features has the size of N * K * C
        outputs = nn.functional.normalize(out, dim=2)  # b 5 1024
        outputs_t = outputs.permute(0, 2, 1)  # b 1024 5

        # generate assignment map for basis for visualization
        return outputs_t, assign, grouping_centers

    # name
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.num_parts) + ')'
# ==================================== #

class KMeansGroupingUnit(MetaModule):
    def __init__(
            self, in_channels, num_parts, proposal_parts=15,
            kmeans_n_iter=30, mask_method='max', start_epoch=10, alpha=0.1
    ):
        super(KMeansGroupingUnit, self).__init__()
        self.alpha = alpha
        self.in_channels   = in_channels
        self.kmeans_n_iter   = kmeans_n_iter
        self.num_parts = num_parts
        self.proposal_parts = proposal_parts
        self.mask_method = mask_method
        self.start_epoch = start_epoch
        # self.kmeans = faiss.Kmeans(in_channels, proposal_parts, niter=kmeans_n_iter)
        # params
        self.weight = nn.Parameter(torch.FloatTensor(num_parts, in_channels, 1, 1))  # n * 1024 * 1*1
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))
        self.part_weighted = nn.Parameter(torch.FloatTensor(num_parts, proposal_parts))


        # self.feature_lists = torch.empty((0, in_channels), dtype=torch.float)

        self.register_buffer(
            'proposal_centroid',
            torch.zeros((self.proposal_parts, in_channels), dtype=torch.float)
        )
        self.register_buffer(
            'data_count',
            torch.zeros(self.proposal_parts, dtype=torch.long)
        )
        self.smooth_factor = nn.Parameter(torch.FloatTensor(self.num_parts))
        nn.init.constant_(self.smooth_factor, 0)

    def reset_parameters(self):
        # msra init
        nn.init.kaiming_normal_(self.weight)
        self.weight.data.clamp_(min=1e-5)

        nn.init.kaiming_normal_(self.part_weighted)
        self.part_weighted.data.clamp_(min=1e-5)

    # def do_kmeans(self, batch_size, features, iteration, current_epoch_feat, train_iters):
    #     if self.training:
    #         if iteration == 0:
    #             self.proposal_centroid = torch.zeros((self.proposal_parts, self.in_channels), dtype=torch.float)
    #             _, error  = self.kmeans.fit_predict(current_epoch_feat)
    #             self.proposal_centroid = self.kmeans.centroids
    #             labels = self.kmeans.predict(features, self.proposal_centroid)
    #             error = 0
    #         else:
    #             labels, error = self.kmeans.fit_predict(
    #                 features, self.proposal_centroid, train_iters=train_iters
    #             )
    #             self.proposal_centroid = self.kmeans.centroids
    #     else:
    #         labels = self.kmeans.predict(features, self.proposal_centroid)
    #         error = 0
    #
    #     all_labels = labels.reshape(batch_size, -1)
    #     batch_labels = {}
    #     for batch_idx in range(all_labels.shape[0]):
    #         batch_labels[batch_idx] = all_labels[batch_idx].unique(return_counts=True)
    #     return batch_labels, error

    def select_parts(self, cluster_info, inputs, grouping_centers):
        emb_org_norm = torch.norm(self.proposal_centroid, 2, 1, True).clamp(min=1e-12)
        emb_org_norm = torch.div(self.proposal_centroid, emb_org_norm)
        selected_grouping_centers = []

        for ibatch in cluster_info.keys():
            cluster_index, cluster_counts = cluster_info[ibatch]
            proposal_center = emb_org_norm[cluster_index]
            # sigma = cluster_counts.float()/cluster_counts.sum()
            # proposal_center = sigma.unsqueeze(1) * proposal_center
            selected_grouping_center = torch.mm(self.part_weighted[:, cluster_index], proposal_center)

            selected_grouping_centers.append(selected_grouping_center.unsqueeze(0))

        return self.alpha * torch.cat(selected_grouping_centers, dim=0).to(inputs.device) + (1-self.alpha)*grouping_centers

    def forward(self, inputs, km_obj, iteration):
        assert inputs.dim() == 4

        # 0. store input size
        batch_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels == self.in_channels

        # 1. generate the grouping centers  # 5 1024 1 1 --> 1 5 1024 --> B 5 1024  # 因为
        grouping_centers = self.weight.view(1, self.num_parts, self.in_channels).expand(
            batch_size, self.num_parts, self.in_channels
        ).contiguous()

        # 1. gather features from each gpu
        # gathered_features = gather(inputs)
        error = torch.zeros(1).to(inputs.device)
        featslist = inputs.view(-1, inputs.size(1))
        if km_obj is not None and self.training:
            # current_epoch_feat is None: epoch no. smaller than predefine in first domain
            # other iterations in the same epoch instead of first iteration

            with torch.no_grad():
                I, error = do_kmeans(
                    True,
                    featslist, km_obj,
                    iteration,
                    self.proposal_centroid
                )
                # Update centroids.

                if iteration > 0:
                    # print(km_obj.centroids.shape)
                    # print(self.proposal_centroid.shape)
                    # print(torch.unique(I))
                    for k in torch.unique(I):
                        idx_k = torch.where(I == k)[0]
                        self.data_count[k] += len(idx_k)
                        centroid_lr    = len(idx_k) / (self.data_count[k] + 1e-6)
                        self.proposal_centroid[k]   = (1 - centroid_lr) * self.proposal_centroid[k] + \
                                                 centroid_lr * featslist[idx_k].mean(0)
                batch_labels = {}
                all_labels = I.reshape(batch_size, -1).to(device=inputs.device)
                for batch_idx in range(all_labels.shape[0]):
                    batch_labels[batch_idx] = all_labels[batch_idx].unique(return_counts=True)
        if not self.training:
            assert torch.nonzero(self.proposal_centroid).size(0) != 0
            I, error = do_kmeans(
                False,
                featslist, km_obj,
                iteration,
                self.proposal_centroid
            )
            batch_labels = {}
            all_labels = I.reshape(batch_size, -1).to(device=inputs.device)
            for batch_idx in range(all_labels.shape[0]):
                batch_labels[batch_idx] = all_labels[batch_idx].unique(return_counts=True)
        grouping_centers = self.select_parts(
            batch_labels, inputs, grouping_centers
        )
        # 2. compute assignment matrix
        # - d = -\|X - C\|_2 = - X^2 - C^2 + 2 * C^T X
        # C^T X (N * K * H * W)
        inputs_cx = inputs.contiguous().view(batch_size, self.in_channels, input_h * input_w)
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        # X^2 (N * C * H * W) -> (N * 1 * H * W) -> (N * K * H * W)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1)
        # C^2 (K * C * 1 * 1) -> 1 * K * 1 * 1
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3)
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        # expand the smooth term
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(batch_size, -1, input_h, input_w)
        # assignment = softmax(-d/s) (-d must be negative)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1)  # default dim = 1

        # 3. compute residual coding
        # NCHW -> N * C * HW
        x = inputs.contiguous().view(batch_size, self.in_channels, -1)
        # permute the inputs -> N * HW * C
        x = x.permute(0, 2, 1)

        # compute weighted feats N * K * C
        assign = assign.contiguous().view(batch_size, self.num_parts, -1)
        qx = torch.bmm(assign, x)

        # repeat the graph_weights (K * C) -> (N * K * C)
        c = grouping_centers

        # sum of assignment (N * K * 1) -> (N * K * K)
        sum_ass = torch.sum(assign, dim=2, keepdim=True)

        # residual coding N * K * C
        sum_ass = sum_ass.expand(-1, -1, self.in_channels).clamp(min=1e-5)
        sigma = (beta / 2).sqrt()
        out = ((qx / sum_ass) - c) / sigma.unsqueeze(0).unsqueeze(2)

        # 4. prepare outputs
        # we need to memorize the assignment (N * K * H * W)
        assign = assign.contiguous().view(
            batch_size, self.num_parts, input_h, input_w)

        # output features has the size of N * K * C
        outputs = nn.functional.normalize(out, dim=2)  # b 5 1024
        outputs_t = outputs.permute(0, 2, 1)  # b 1024 5

        # generate assignment map for basis for visualization
        return outputs_t, assign, grouping_centers, error

class KMeansPesudoGroupingUnit(MetaModule):
    def __init__(
            self, in_channels, num_parts, proposal_parts=15,
            kmeans_n_iter=30, mask_method='max', start_epoch=10, alpha=0.1
    ):
        super(KMeansPesudoGroupingUnit, self).__init__()
        self.alpha = alpha
        self.in_channels   = in_channels
        self.kmeans_n_iter   = kmeans_n_iter
        self.num_parts = num_parts
        self.proposal_parts = proposal_parts
        self.mask_method = mask_method
        self.start_epoch = start_epoch
        # self.kmeans = faiss.Kmeans(in_channels, proposal_parts, niter=kmeans_n_iter)
        # params
        self.weight = nn.Parameter(torch.FloatTensor(num_parts, in_channels, 1, 1))  # n * 1024 * 1*1
        self.smooth_factor = nn.Parameter(torch.FloatTensor(num_parts))
        self.part_weighted = nn.Parameter(torch.FloatTensor(num_parts, proposal_parts))


        # self.feature_lists = torch.empty((0, in_channels), dtype=torch.float)

        self.register_buffer(
            'proposal_centroid',
            torch.zeros((self.proposal_parts, in_channels), dtype=torch.float)
        )
        self.register_buffer(
            'data_count',
            torch.zeros(self.proposal_parts, dtype=torch.long)
        )
        self.smooth_factor = nn.Parameter(torch.FloatTensor(self.num_parts))
        nn.init.constant_(self.smooth_factor, 0)

    def reset_parameters(self):
        # msra init
        nn.init.kaiming_normal_(self.weight)
        self.weight.data.clamp_(min=1e-5)

        nn.init.kaiming_normal_(self.part_weighted)
        self.part_weighted.data.clamp_(min=1e-5)

    def select_parts(self, cluster_info, inputs, grouping_centers):
        emb_org_norm = torch.norm(self.proposal_centroid, 2, 1, True).clamp(min=1e-12)
        emb_org_norm = torch.div(self.proposal_centroid, emb_org_norm)
        selected_grouping_centers = []

        for ibatch in cluster_info.keys():
            cluster_index, cluster_counts = cluster_info[ibatch]
            proposal_center = emb_org_norm[cluster_index]
            # sigma = cluster_counts.float()/cluster_counts.sum()
            # proposal_center = sigma.unsqueeze(1) * proposal_center
            selected_grouping_center = torch.mm(self.part_weighted[:, cluster_index], proposal_center)

            selected_grouping_centers.append(selected_grouping_center.unsqueeze(0))

        return self.alpha * torch.cat(selected_grouping_centers, dim=0).to(inputs.device) + (1-self.alpha)*grouping_centers

    def forward(self, inputs, km_obj, iteration):
        assert inputs.dim() == 4

        # 0. store input size
        batch_size = inputs.size(0)
        in_channels = inputs.size(1)
        input_h = inputs.size(2)
        input_w = inputs.size(3)
        assert in_channels == self.in_channels

        # 1. generate the grouping centers  # 5 1024 1 1 --> 1 5 1024 --> B 5 1024  # 因为
        grouping_centers = self.weight.view(1, self.num_parts, self.in_channels).expand(
            batch_size, self.num_parts, self.in_channels
        ).contiguous()

        # 1. gather features from each gpu
        # gathered_features = gather(inputs)
        error = torch.zeros(1).to(inputs.device)
        I = None
        if km_obj is not None and self.training:
            # current_epoch_feat is None: epoch no. smaller than predefine in first domain
            # other iterations in the same epoch instead of first iteration

            with torch.no_grad():
                featslist = inputs.view(-1, inputs.size(1))
                I, error = do_kmeans(
                    self.training,
                    featslist, km_obj,
                    iteration,
                    self.proposal_centroid
                )
                # Update centroids.

                if iteration > 0:
                    # print(km_obj.centroids.shape)
                    # print(self.proposal_centroid.shape)
                    # print(torch.unique(I))
                    for k in torch.unique(I):
                        idx_k = torch.where(I == k)[0]
                        self.data_count[k] += len(idx_k)
                        centroid_lr    = len(idx_k) / (self.data_count[k] + 1e-6)
                        self.proposal_centroid[k]   = (1 - centroid_lr) * self.proposal_centroid[k] + \
                                                      centroid_lr * featslist[idx_k].mean(0)
                batch_labels = {}
                all_labels = I.reshape(batch_size, -1).to(device=inputs.device)
                for batch_idx in range(all_labels.shape[0]):
                    batch_labels[batch_idx] = all_labels[batch_idx].unique(return_counts=True)

            grouping_centers = self.select_parts(
                batch_labels, inputs, grouping_centers
            )
        # 2. compute assignment matrix
        # - d = -\|X - C\|_2 = - X^2 - C^2 + 2 * C^T X
        # C^T X (N * K * H * W)
        inputs_cx = inputs.contiguous().view(batch_size, self.in_channels, input_h * input_w)
        cx_ = torch.bmm(grouping_centers, inputs_cx)
        cx = cx_.contiguous().view(batch_size, self.num_parts, input_h, input_w)
        # X^2 (N * C * H * W) -> (N * 1 * H * W) -> (N * K * H * W)
        x_sq = inputs.pow(2).sum(1, keepdim=True)
        x_sq = x_sq.expand(-1, self.num_parts, -1, -1)
        # C^2 (K * C * 1 * 1) -> 1 * K * 1 * 1
        c_sq = grouping_centers.pow(2).sum(2).unsqueeze(2).unsqueeze(3)
        c_sq = c_sq.expand(-1, -1, input_h, input_w)
        # expand the smooth term
        beta = torch.sigmoid(self.smooth_factor)
        beta_batch = beta.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        beta_batch = beta_batch.expand(batch_size, -1, input_h, input_w)
        # assignment = softmax(-d/s) (-d must be negative)
        assign = (2 * cx - x_sq - c_sq).clamp(max=0.0) / beta_batch
        assign = nn.functional.softmax(assign, dim=1)  # default dim = 1

        # 3. compute residual coding
        # NCHW -> N * C * HW
        x = inputs.contiguous().view(batch_size, self.in_channels, -1)
        # permute the inputs -> N * HW * C
        x = x.permute(0, 2, 1)

        # compute weighted feats N * K * C
        assign = assign.contiguous().view(batch_size, self.num_parts, -1)
        qx = torch.bmm(assign, x)

        # repeat the graph_weights (K * C) -> (N * K * C)
        c = grouping_centers

        # sum of assignment (N * K * 1) -> (N * K * K)
        sum_ass = torch.sum(assign, dim=2, keepdim=True)

        # residual coding N * K * C
        sum_ass = sum_ass.expand(-1, -1, self.in_channels).clamp(min=1e-5)
        sigma = (beta / 2).sqrt()
        out = ((qx / sum_ass) - c) / sigma.unsqueeze(0).unsqueeze(2)

        # 4. prepare outputs
        # we need to memorize the assignment (N * K * H * W)
        assign = assign.contiguous().view(
            batch_size, self.num_parts, input_h, input_w)

        # output features has the size of N * K * C
        outputs = nn.functional.normalize(out, dim=2)  # b 5 1024
        outputs_t = outputs.permute(0, 2, 1)  # b 1024 5

        # generate assignment map for basis for visualization
        return outputs_t, assign, grouping_centers, error, I
# ==================================== #

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# Attention for dynamic tokens
class MGAttention(nn.Module):
    def __init__(self, dim, attn_dim=512, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim_q = attn_dim
        self.dim_k = attn_dim
        self.dim_v = attn_dim
        self.num_heads = num_heads

        self.q = nn.Linear(dim, self.dim_q)
        self.k = nn.Linear(dim, self.dim_k)
        self.v = nn.Linear(dim, self.dim_v)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.num_heads)
        K = self.k(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.num_heads)
        V = self.v(x).reshape(-1, x.shape[0], x.shape[1], self.dim_v // self.num_heads)
        scale = (K.size(-1) // self.num_heads) ** -0.5

        atten = torch.matmul(Q, K.permute(0, 1, 3, 2)) * scale
        atten = self.softmax(atten)  # Q * K.T() # batch_size * seq_len * seq_len
        atten = self.attn_drop(atten)

        output = torch.matmul(atten, V).reshape(
            x.shape[0], x.shape[1], -1
        )  # Q * K.T() * V # batch_size * seq_len * dim_v
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class MGTBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, attn_dim=512,
            num_heads=4, mlp_ratio=2., attn_drop=0., drop_path=0., drop=0.,
    ):
        super(MGTBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = MGAttention(
            dim=in_channels, attn_dim=attn_dim, num_heads=num_heads,
            attn_drop=attn_drop, proj_drop=0.1
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(in_channels)
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(
            in_features=in_channels, hidden_features=mlp_hidden_dim,
            out_features=out_channels, act_layer=nn.GELU, drop=drop
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, concat_feat):
        x = concat_feat
        # norm1
        concat_feat = self.norm1(concat_feat)
        # attn
        attn_x = self.attn(concat_feat)
        x = x + self.drop_path(attn_x)
        x = self.norm1(x)
        x = self.drop_path(self.mlp(x))
        return x


# ==================================== #

class MGABlock(nn.Module):
    def __init__(self, pooling, in_dim, score_dim):
        super(MGABlock, self).__init__()
        self.pooling = pooling
        self.in_proj = nn.Linear(in_dim, score_dim, bias=False)
        self.FeatScore = dot_attention()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feat_list, assign_list):
        layerq1_feat, layerq2_feat, layer_feat = feat_list
        q_assign, k_assign = assign_list
        layerq1_feat = self.in_proj(layerq1_feat)
        q1_weight = self.FeatScore(layerq1_feat, layer_feat).unsqueeze(1)
        q2_weight = self.FeatScore(layerq2_feat, layer_feat).unsqueeze(1)
        q_weight = self.softmax(torch.cat((q1_weight, q2_weight), dim=1))
        q1_weight = q_weight[:, 0].reshape(q_assign.shape[:2]).unsqueeze(-1).unsqueeze(-1)
        q2_weight = q_weight[:, 1].reshape(q_assign.shape[:2]).unsqueeze(-1).unsqueeze(-1)
        return {
            'layer_assign': self.pooling(q_assign) * q1_weight + k_assign * q2_weight,
            'q1_weight': q1_weight,
            'q2_weight': q2_weight
        }



