from __future__ import print_function, absolute_import
import time
import math

import copy

import torch
import torch.nn as nn
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, \
    CrossEntropyLabelSmooth_weighted, SoftTripletLoss_weight, NonlapLoss, cross_entropy_loss, triplet_loss
from .utils.meters import AverageMeter
from .utils.my_tools import *
from reid.metric_learning.distance import cosine_similarity, cosine_distance
from reid.grad_cam import GradCAM

class TrainerCam(object):
    def __init__(self, model, num_classes, loss_items, margin=0.0, use_TSBN=False, header='part'):
        super(TrainerCam, self).__init__()
        self.use_TSBN = use_TSBN
        self.header = header

        self.loss_items = loss_items

        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_ce_weight = CrossEntropyLabelSmooth_weighted(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.criterion_triple_weight = SoftTripletLoss_weight(margin=margin).cuda()
        self.trip_hard = TripletLoss(margin=margin).cuda()
        if header in part_header_list:
            self.criterion_shaping = NonlapLoss(
                radius=2, std=0.2, alpha=1,
                beta=0.001, num_parts=5, eps=1e-5
            ).cuda()

    def train(self, epoch, data_loader_train, data_loader_replay, optimizer, training_phase,
              train_iters=200, add_num=0, old_model=None, km_obj=None):

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_acc = AverageMeter()
        train_part_acc = AverageMeter()
        pid_acc = AverageMeter()
        old_model_train_acc = AverageMeter()

        losses = {}
        for loss_name in self.loss_items:
            losses[loss_name] = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            train_inputs = data_loader_train.next()
            data_time.update(time.time() - end)
            s_inputs, targets, cids, domains = self._parse_data(train_inputs)
            targets += add_num

            # global_feat[..., 0, 0], bn_feat,        cls_outputs, fake_globalfeat_list
            # s_features,             bn_features,    s_cls_out,   fake_feat_list

            model_output = self.model(
                s_inputs, targets=targets, domains=domains,
                training_phase=training_phase, iteration=i, epoch=epoch,
                km_obj=km_obj,
                train_iters=train_iters
            )

            cls_outputs = model_output['cls_outputs']
            bn_feat = model_output['bn_feat']
            global_feat = model_output['global_feat']

            ############################# accuracy for training phase #############################
            pred_class_logits = model_output['cls_outputs'].detach()
            train_acc_val = log_accuracy(pred_class_logits, targets)
            train_acc.update(train_acc_val.item())


            if old_model is not None:
                with torch.no_grad():
                    old_model_new_outputs = old_model(
                        s_inputs, targets=targets, domains=domains, training_phase=training_phase,
                        fkd=False, iteration=i, epoch=epoch, current_epoch_feat=None
                    )
                ############################# accuracy for old model #############################
                old_model_pred_class_logits = old_model_new_outputs['cls_outputs'].detach()
                old_model_train_acc_val = log_accuracy(old_model_pred_class_logits, targets)
                old_model_train_acc.update(old_model_train_acc_val.item())


            ############################# Loss for TSBN #############################
            if self.use_TSBN:
                ############################# We-ID： global + part #############################
                if old_model is not None:
                    weight_list = []
                    for j in range(training_phase - 1):
                        statistics_mean = old_model.module.task_specific_batch_norm[j].running_mean.unsqueeze(0)
                        weight_list.append(cosine_similarity(global_feat, statistics_mean).view(-1))
                    temp = torch.mean(torch.stack(weight_list, dim=0), dim=0)
                    weights = F.softmax(temp * 2, dim=0)
                    loss_ce, loss_tp = self.forward_weight(global_feat, cls_outputs, targets, weights)
                else:
                    loss_ce, loss_tp = self._forward(global_feat, cls_outputs, targets)

                loss = loss_ce + loss_tp
                losses['ce'].update(loss_ce.item())
                losses['tr'].update(loss_tp.item())

                # ############################# DCL #############################
                if 'DCL' in self.loss_items:
                    if epoch >= 10:
                        fake_feat_list = model_output['fake_feat_list']
                        DCL = self.DCL(bn_feat, fake_feat_list, targets)
                        loss += DCL
                        losses['DCL'].update(DCL.item())

                if old_model is not None:
                    if 'kd_new' in self.loss_items:
                        old_bn_feat_new = old_model_new_outputs['bn_feat']
                        old_cls_outputs_new = old_model_new_outputs['cls_outputs']

                        KD_loss_new = self.loss_kd_old(
                            bn_feat, old_bn_feat_new,
                            cls_outputs, old_cls_outputs_new
                        )
                        losses['kd_new'].update(KD_loss_new.item())
                        loss += KD_loss_new
                    if "contrast_new" in self.loss_items:
                        old_bn_feat_new = old_model_new_outputs['bn_feat']
                        ctr_label = torch.cat((targets, targets))
                        ctr_feat = torch.cat((bn_feat, old_bn_feat_new), dim=0)

                        loss_ctr = self.loss_part_contrast(ctr_feat, ctr_label)
                        loss += loss_ctr
                        losses['contrast_new'].update(loss_ctr.item())

                if data_loader_replay is not None:
                    imgs_r, fnames_r, pid_r, cid_r, domain_r = next(iter(data_loader_replay))
                    imgs_r = imgs_r.cuda()
                    pid_r = pid_r.cuda()

                    model_output_r = self.model(
                        imgs_r, targets=pid_r,
                        domains=domain_r, training_phase=training_phase, fkd=True
                    )

                    # ############################# PT_ID #############################
                    if 'PT_ID' in self.loss_items:
                        # PT-ID
                        fake_feat_list = model_output['fake_feat_list']
                        loss_PT_ID = self.PT_ID(fake_feat_list, bn_feat, targets)
                        losses['PT_ID'].update(loss_PT_ID.item())
                        loss += loss_PT_ID

                    # ############################# triplet loss for old data #############################
                    if 'replay_tr' in self.loss_items:
                        global_feat_r = model_output_r['global_feat']
                        loss_tr_r = self.trip_hard(global_feat_r, pid_r)[0]
                        losses['replay_tr'].update(loss_tr_r.item())
                        loss += loss_tr_r

                    if old_model is not None:
                        with torch.no_grad():
                            old_model_outputs = old_model(
                                imgs_r, targets=pid_r,
                                domains=domain_r, training_phase=training_phase, fkd=True
                            )



                        # ############################# sce loos for old data #############################
                        if 'sce' in self.loss_items:
                            bn_feat_r = model_output_r['bn_feat']
                            old_bn_feat = old_model_outputs['bn_feat']

                            cls_outputs_r = model_output_r['cls_outputs']
                            old_cls_outputs = old_model_outputs['cls_outputs']

                            KD_loss_r = self.loss_kd_old(
                                bn_feat_r, old_bn_feat,
                                cls_outputs_r, old_cls_outputs
                            )
                            losses['sce'].update(KD_loss_r.item())
                            loss += KD_loss_r

                        # PT-KD
                        if 'PT_KD' in self.loss_items:
                            fake_feat_list_r = model_output_r['fake_feat_list']
                            old_fake_feat_list_r = old_model_outputs['fake_feat_list']

                            loss_PT_KD = self.PT_KD(
                                old_fake_feat_list_r[:(training_phase-1)],
                                fake_feat_list_r[:(training_phase-1)]
                            )
                            losses['PT_KD'].update(loss_PT_KD.item())
                            loss += loss_PT_KD

            else:
                loss_ce, loss_tp = self._forward(global_feat, cls_outputs, targets)

                loss = loss_ce + loss_tp
                losses['ce'].update(loss_ce.item())
                losses['tr'].update(loss_tp.item())

            ############################# Loss for knowledge distillation #############################

            if self.header in part_header_list:
                cls_outputs_part = model_output['cls_outputs_part']
                train_part_acc_val = log_accuracy(cls_outputs_part.detach(), targets)
                train_part_acc.update(train_part_acc_val.item())

                bn_feat_part = model_output['bn_feat_part']

                if ('part_tr' in self.loss_items) and ('part_ce' in self.loss_items):
                    loss_part_ce = cross_entropy_loss(cls_outputs_part, targets, eps=0.1)
                    loss_part_tp = self.trip_hard(bn_feat_part, targets, emb_=bn_feat_part)

                    loss += loss_part_ce + loss_part_tp
                    losses['part_ce'].update(loss_part_ce.item())
                    losses['part_tr'].update(loss_part_tp.item())

                ############################# Loss for non over lap #############################
                if 'nonlap' in self.loss_items:
                    # 让各个部分尽量不重叠的损失
                    #  assign: BS, partnum, height, width
                    soft_assign = model_output['soft_assign']
                    if isinstance(soft_assign, list):
                        loss_shape = 0
                        for idx, iassign in enumerate(soft_assign):
                            loss_shape += self.criterion_shaping(iassign)
                        losses[f'nonlap'].update(loss_shape.item())
                        loss += loss_shape
                    else:
                        loss_shape = self.criterion_shaping(soft_assign)
                        losses['nonlap'].update(loss_shape.item())
                        loss += loss_shape

                if "gc_pesudo_loss" in self.loss_items:
                    cluster_label = model_output['cluster_label']
                    bn_pesudo_gc_outputs = model_output['bn_pesudo_gc_outputs']
                    gc_pesudo_loss = self.criterion_ce(bn_pesudo_gc_outputs, cluster_label.squeeze())
                    losses['gc_pesudo_loss'].update(gc_pesudo_loss.item())
                    loss += gc_pesudo_loss

                if old_model is not None:
                    if 'kd_part_new' in self.loss_items:
                        old_bn_feat_part_new = old_model_new_outputs['bn_feat_part']

                        old_cls_outputs_part_new = old_model_new_outputs['cls_outputs_part']

                        part_KD_loss_new = self.loss_kd_old(
                            bn_feat_part, old_bn_feat_part_new,
                            cls_outputs_part, old_cls_outputs_part_new
                        )
                        losses['kd_part_new'].update(part_KD_loss_new.item())
                        loss += part_KD_loss_new

                    if "part_contrast_new" in self.loss_items:
                        old_bn_feat_part_new = old_model_new_outputs['bn_feat_part']
                        ctr_label_part = torch.cat((targets, targets), dim=0)
                        ctr_feat_part = torch.cat((bn_feat_part, old_bn_feat_part_new), dim=0)

                        loss_ctr_part = self.loss_part_contrast(ctr_feat_part, ctr_label_part)
                        loss += loss_ctr_part
                        losses['part_contrast_new'].update(loss_ctr_part.item())

                    if data_loader_replay is not None:
                        if 'kd_part' in self.loss_items:

                            bn_feat_part_r = model_output_r['bn_feat_part']
                            old_bn_feat_part = old_model_outputs['bn_feat_part']

                            cls_outputs_part_r = model_output_r['cls_outputs_part']
                            old_cls_outputs_part = old_model_outputs['cls_outputs_part']

                            part_KD_loss_r = self.loss_kd_old(
                                bn_feat_part_r, old_bn_feat_part,
                                cls_outputs_part_r, old_cls_outputs_part
                            )
                            losses['kd_part'].update(part_KD_loss_r.item())
                            loss += part_KD_loss_r

                        if "kd_assign" in self.loss_items:
                            soft_assign_r = model_output_r['soft_assign']
                            old_soft_assign = old_model_outputs['soft_assign']
                            bs = soft_assign_r.size(0)
                            kd_assign_loss = self.loss_kd_assign(
                                soft_assign_r.reshape(bs, -1),
                                old_soft_assign.reshape(bs, -1)
                            )
                            losses['kd_assign'].update(kd_assign_loss.item())
                            loss += kd_assign_loss


            if "pair_loss" in self.loss_items:
                score = model_output['score']
                pair_label = model_output['pair_label']
                pair_loss, pidacc = self.loss_edge(score, pair_label)
                pid_acc.update(pidacc)
                losses['pair_loss'].update(pair_loss.item())
                loss += pair_loss

            if "cluster_error" in self.loss_items:
                cluster_error = model_output['cluster_error']
                losses['cluster_error'].update(cluster_error.mean().item())


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.use_TSBN:
                for bn in self.model.module.task_specific_batch_norm:
                    bn.weight.data.copy_(self.model.module.bottleneck.weight.data)


            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) == train_iters:
                print('Epoch: [{}][{}/{}]\t'
                      'train accuracy {:.5f} ({:.5f})\t'
                      'old model train accuracy {:.5f} ({:.5f})\t'
                      'train part accuracy {:.5f} ({:.5f})\t'
                      'train pid accuracy {:.5f} ({:.5f})\t'
                      'Time {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, train_iters,
                              train_acc.val, train_acc.avg,
                              old_model_train_acc.val, old_model_train_acc.avg,
                              train_part_acc.val, train_part_acc.avg,
                              pid_acc.val, pid_acc.avg,
                              batch_time.val, batch_time.avg,
                              )
                      )
                ret_loss = {}
                for iloss_name, iloss_val in losses.items():
                    print("{}: {:.3f} ({:.3f})\t".format(
                        iloss_name,
                        iloss_val.val,
                        iloss_val.avg,
                    ))
                    ret_loss[iloss_name] = iloss_val.avg

        return ret_loss

    def _parse_data(self, inputs):
        imgs, _, pids, cids, domains = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets, cids, domains

    def _forward(self, global_feat, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        # loss_tr = self.criterion_triple(global_feat, global_feat, targets)
        loss_tr = self.trip_hard(global_feat, targets, emb_=global_feat)
        # cross_entropy_loss(s_outputs, targets, 0.1, 0.2)
        # loss_tr = triplet_loss(global_feat, targets, 0, False, True)
        return loss_ce, loss_tr

    def forward_weight(self, global_feat, s_outputs, targets, weigths):
        loss_ce = self.criterion_ce_weight(s_outputs, targets, weigths)
        loss_tr = self.criterion_triple_weight(global_feat, global_feat, targets, weigths)
        return loss_ce, loss_tr

    def DCL(self, features, feature_list_bn, pids):
        loss = []
        uniq_pid = torch.unique(pids)
        for pid in uniq_pid:
            pid_index = torch.where(pid == pids)[0]
            global_bn_feat_single = features[pid_index]
            for feats in feature_list_bn:
                speci_bn_feat_single = feats[pid_index]
                distance_matrix = -torch.mm(F.normalize(global_bn_feat_single, p=2, dim=1),
                                            F.normalize(speci_bn_feat_single, p=2, dim=1).t().detach())
                loss.append(torch.mean(distance_matrix))
        loss = torch.mean(torch.stack(loss))
        return loss


    def loss_kd_L1(self, new_features, old_features):

        L1 = torch.nn.L1Loss()

        old_simi_matrix = cosine_distance(old_features, old_features)
        new_simi_matrix = cosine_distance(new_features, new_features)

        simi_loss = L1(old_simi_matrix, new_simi_matrix)

        return simi_loss

    def PT_KD(self, fake_feat_list_old, fake_feat_list_new):
        loss_cross = []
        for i in range(len(fake_feat_list_old)):
            for j in range(i, len(fake_feat_list_old)):
                loss_cross.append(self.loss_kd_L1(fake_feat_list_old[i], fake_feat_list_new[j]))
        loss_cross = torch.mean(torch.stack(loss_cross))
        return loss_cross

    def loss_kd_old(self, new_features, old_features, new_logits, old_logits):

        logsoftmax = nn.LogSoftmax(dim=1).cuda()

        L1 = torch.nn.L1Loss()

        old_simi_matrix = cosine_distance(old_features, old_features)
        new_simi_matrix = cosine_distance(new_features, new_features)

        simi_loss = L1(old_simi_matrix, new_simi_matrix)
        loss_ke_ce = (- F.softmax(old_logits, dim=1).detach() * logsoftmax(new_logits)).mean(0).sum()

        return loss_ke_ce + simi_loss

    def loss_kd_assign(self, soft_assign_r, old_soft_assign):
        L1 = torch.nn.L1Loss()

        simi_loss = L1(old_soft_assign, soft_assign_r)
        loss_ke_ce = (- old_soft_assign * soft_assign_r.log()).mean(0).sum()

        return 0.05 * loss_ke_ce + simi_loss

    def PT_ID(self, feature_list_bn, bn_feat, pids):

        loss = []
        for features in feature_list_bn:
            loss.append(self.trip_hard(features, pids)[0])
        loss.append(self.trip_hard(bn_feat, pids)[0])
        loss = torch.mean(torch.stack(loss))

        loss_cross = []
        for i in range(len(feature_list_bn)):
            for j in range(i + 1, len(feature_list_bn)):
                loss_cross.append(self.trip_hard(feature_list_bn[i], pids, feature_list_bn[j]))
        loss_cross = torch.mean(torch.stack(loss_cross))
        loss = 0.5 * (loss + loss_cross)

        return loss


    def pair_loss(self, sub_score, sub_pair_labels):
        pair_loss = 0
        pidacc = 0

        split_nums = len(self.model.device_ids)

        each_feat_dim = sub_score.size(0) // split_nums
        reize_dim = int(math.sqrt(each_feat_dim))

        for i in range(split_nums):
            score_ = sub_score[i*each_feat_dim: (i+1)*each_feat_dim].reshape(reize_dim, reize_dim)
            pair_labels_ = sub_pair_labels[i*each_feat_dim: (i+1)*each_feat_dim].reshape(reize_dim, reize_dim)
            pair_loss_ = F.binary_cross_entropy_with_logits(
                score_, pair_labels_, reduction='none'
            ).sum(-1)

            finite_mask = pair_loss_.isfinite()
            with torch.no_grad():
                min_pos = torch.min(score_ * pair_labels_ +
                                    (1 - pair_labels_ + torch.eye(score_.size(0), device=score_.device)) * 1e15, dim=1)[0]
                max_neg = torch.max(score_ * (1 - pair_labels_) - pair_labels_ * 1e15, dim=1)[0]
                pidacc_ = (min_pos > max_neg).float()

            if finite_mask.any():
                pair_loss_ = pair_loss_[finite_mask].mean()
                pidacc_ = pidacc_[finite_mask].mean()
                pair_loss += pair_loss_
                pidacc += pidacc_
        return 0.1 * pair_loss/len(self.model.device_ids), pidacc / len(self.model.device_ids)

    def loss_part_contrast(self, features, labels, temperature=0.5):
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        if labels is not None: # 如果给出了labels, mask根据label得到，两个样本i,j的label相等时，mask_{i,j}=1
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(features.device)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        # 构建mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(features.device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask
        num_positives_per_row  = torch.sum(positives_mask, axis=1) # 除了自己之外，正样本的个数  [2 0 2 2]
        # denominator = torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)
        denominator = 0.2*torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) + 0.8 * torch.sum(exp_logits * positives_mask, axis=1, keepdims=True)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")


        log_probs = torch.sum(log_probs*positives_mask , axis=1)[num_positives_per_row > 0] / num_positives_per_row[num_positives_per_row > 0]
        '''
        计算正样本平均的log-likelihood
        考虑到一个类别可能只有一个样本，就没有正样本了 比如我们labels的第二个类别 labels[1,2,1,1]
        所以这里只计算正样本个数>0的    
        '''
        # loss
        loss = -log_probs
        loss *= temperature
        loss = loss.mean()
        return loss