from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import numpy as np
import torch

from .evaluation_metrics import cmc, mean_ap, mean_ap_cuhk03, market1501_torch, cuhk03_torch
from .feature_extraction import extract_cnn_feature, extract_pretain_feature
from .utils.meters import AverageMeter, CatMeter

from tqdm import tqdm
import torch.nn.functional as F

import warnings

try:
    from reid.CEval.rank_cylib.rank_cy import evaluate_cy
    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )

def extract_extra_features(model, data_loader, args, use_pretrain_feat=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    extras = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domians) in tqdm(enumerate(data_loader), total=len(data_loader)):
            data_time.update(time.time() - end)
            if use_pretrain_feat:
                model_outputs = extract_pretain_feature(model, imgs)
            else:
                model_outputs = extract_cnn_feature(model, imgs, args, middle_feat=True)
            if 'kmeans' in args.header:
                outputs = model_outputs['outputs'].data.cpu()
                region_outs = model_outputs[args.match_feat].data.cpu()

                for fname, output, region_out, pid in zip(fnames, outputs, region_outs, pids):
                    features[fname] = output
                    labels[fname] = pid
                    extras[fname] = region_out
            else:
                outputs = model_outputs['outputs']
                for fname, output, pid in zip(fnames, outputs, pids):
                    features[fname] = output
                    labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

    return features, labels, extras

# def extract_features(model, data_loader, args):
#     model.eval()
#     batch_time = AverageMeter()
#     data_time = AverageMeter()
#
#     features = OrderedDict()
#     labels = OrderedDict()
#
#     end = time.time()
#     with torch.no_grad():
#         for i, (imgs, fnames, pids, cids, domians) in tqdm(enumerate(data_loader), total=len(data_loader)):
#             data_time.update(time.time() - end)
#
#             outputs = extract_cnn_feature(model, imgs, args)
#             for fname, output, pid in zip(fnames, outputs, pids):
#                 features[fname] = output
#                 labels[fname] = pid
#
#             batch_time.update(time.time() - end)
#             end = time.time()
#
#     return features, labels


def pairwise_distance(features, query=None, gallery=None, metric=None, measure='L2'):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    if measure == 'L2':
        x = x.view(m, -1)
        y = y.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
            y = metric.transform(y)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                 torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        dist_m.addmm_(1, -2, x, y.t())
        return dist_m, x.numpy(), y.numpy()
    else:
        input1_normed = F.normalize(x, p=2, dim=1)
        input2_normed = F.normalize(y, p=2, dim=1)
        dist_m = -torch.mm(input1_normed, input2_normed.t())
        return dist_m, x.numpy(), y.numpy()

# def evaluate_all(query_features, gallery_features, distmat, query=None, gallery=None,
#                  query_ids=None, gallery_ids=None,
#                  query_cams=None, gallery_cams=None,
#                  cmc_topk=(1, 5, 10), cmc_flag=False, cuhk03=False, use_gpu=False, use_cython=False, use_distmat=False, **kwarg):
#     if query is not None and gallery is not None:
#         query_ids = [pid for _, pid, _, _ in query]
#         gallery_ids = [pid for _, pid, _, _ in gallery]
#         query_cams = [cam for _, _, cam, _ in query]
#         gallery_cams = [cam for _, _, cam, _ in gallery]
#     else:
#         assert (query_ids is not None and gallery_ids is not None
#                 and query_cams is not None and gallery_cams is not None)
#
#     # Compute mean AP
#     # if use_cython and IS_CYTHON_AVAI:
#     if use_gpu:
#         print('Using torch to evaluate')
#         if cuhk03:
#             cmc_scores, mAP = cuhk03_torch(
#                 distmat, query_features, gallery_features, query_ids, gallery_ids, query_cams, gallery_cams, 100,
#                 use_distmat, **kwarg
#             )
#         else:
#             cmc_scores, mAP = market1501_torch(
#                 distmat, query_features, gallery_features, query_ids, gallery_ids, query_cams, gallery_cams, 100,
#                 use_distmat, **kwarg
#             )
#         print('Mean AP: {:4.1%}'.format(mAP))
#         if (not cmc_flag):
#             return mAP
#
#         print("cython's cmc")
#         for k in cmc_topk:
#             print('  top-{:<4}{:12.1%}'
#                   .format(k, cmc_scores[k - 1]))
#         return cmc_scores, mAP
#     else:
#         print('Using numpy to evaluate')
#         if cuhk03:
#             mAP = mean_ap_cuhk03(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
#         else:
#             mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams, use_distmat=True)
#         print('Mean AP: {:4.1%}'.format(mAP))
#
#         if (not cmc_flag):
#             return mAP
#
#         cmc_configs = {
#             'market1501': dict(separate_camera_set=False,
#                                single_gallery_shot=False,
#                                first_match_break=True),
#             'cuhk03': dict(separate_camera_set=True,
#                            single_gallery_shot=True,
#                            first_match_break=False)
#         }
#         cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
#                                 query_cams, gallery_cams, **params)
#                       for name, params in cmc_configs.items()}
#
#         if cuhk03:
#             print('CUHK03 CMC Scores:')
#             for k in cmc_topk:
#                 print('  top-{:<4}{:12.1%}'
#                       .format(k,
#                               cmc_scores['cuhk03'][k - 1]))
#             return cmc_scores['cuhk03'], mAP
#
#         else:
#             print('CMC Scores:')
#             for k in cmc_topk:
#                 print('  top-{:<4}{:12.1%}'
#                       .format(k,
#                               cmc_scores['market1501'][k - 1]))
#             return cmc_scores['market1501'], mAP

def evaluate_datasets(evaluator, data_info, evaluate_name_list=()):

    test_result = {}
    for name in evaluate_name_list:
        print(f'Results on {name}')
        test_info = data_info[name]
        if name in ['cuhk03']:
            cuhk03 = True
        else:
            cuhk03 = False
        _R1, _mAP = evaluator.evaluate(
            test_info['test_loader'],  test_info['dataset'].query,  test_info['dataset'].gallery,
            cmc_flag=True, cuhk03=cuhk03
        )
        test_result[name] = {
            'R1': _R1[0],
            'mAP': _mAP,
        }
    return test_result

def evalute_all_generalization(evaluator, query_loader, gallery_loader):
    test_result = {}
    _R1, _mAP = evaluator.evaluate_all(
        query_loader, gallery_loader
    )
    test_result['all_generalization'] = {
        'R1': _R1[0],
        'mAP': _mAP,
    }
    return test_result

def format_evalute_info(test_result):
    print_info = {}
    for iname, evalute_index in test_result.items():
        print_info[f"{iname}_mAP"] = evalute_index['mAP']
        print_info[f"{iname}_R1"] = evalute_index['R1']
    return print_info


class Evaluator(object):
    def __init__(self, model, args):
        super(Evaluator, self).__init__()
        self.model = model
        self.args = args
        self.use_gpu = self.args['use_gpu']

    def evaluate_all(self, query_loader, gallery_loader):
        from reid.CEval import compute_distance_matrix, fast_evaluate_rank
        print("... extract all generalization features")
        query_features_meter, query_pids_meter, query_cids_meter = CatMeter(), CatMeter(), CatMeter()
        gallery_features_meter, gallery_pids_meter, gallery_cids_meter = CatMeter(), CatMeter(), CatMeter()

        for data in query_loader:
            images, pids, cids = data[0:3]
            model_outputs = extract_cnn_feature(self.model, images, self.args, middle_feat=True)
            features = model_outputs['outputs']
            query_features_meter.update(features)
            query_pids_meter.update(pids)
            query_cids_meter.update(cids)

        for data in gallery_loader:
            images, pids, cids = data[0:3]
            model_outputs = extract_cnn_feature(self.model, images, self.args, middle_feat=True)
            features = model_outputs['outputs']
            gallery_features_meter.update(features)
            gallery_pids_meter.update(pids)
            gallery_cids_meter.update(cids)

        print("calculating the distance...")

        distance_matrix = compute_distance_matrix(
            query_features_meter.get_val(), gallery_features_meter.get_val(), 'euclidean'
        )
        distance_matrix = distance_matrix.data.cpu().numpy()

        CMC, mAP = fast_evaluate_rank(distance_matrix,
                                      query_pids_meter.get_val_numpy(),
                                      gallery_pids_meter.get_val_numpy(),
                                      query_cids_meter.get_val_numpy(),
                                      gallery_cids_meter.get_val_numpy(),
                                      max_rank=50,
                                      use_metric_cuhk03=False,
                                      use_cython=True)
        print('Mean AP: {:4.1%}'.format(mAP))
        print('Rank 1: {:4.1%}'.format(CMC[0]))
        return [CMC[0]], mAP

    # def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False,
    #              rerank=False, pre_features=None, cuhk03=False):
    #     print("... extract features")
    #     if (pre_features is None):
    #         features, _, other_feat = extract_extra_features(
    #             self.model, data_loader,
    #             self.args
    #         )
    #     else:
    #         features = pre_features
    #         other_feat = None
    #     print("calculating the distance...")
    #     distmat, query_features, gallery_features = pairwise_distance(features, query, gallery, metric=metric)
    #     print("starting evaluate query and gallery...")
    #     kwargs = set_postprocess_input(self.args, self.model, query, gallery, other_feat)
    #     results = evaluate_all(query_features, gallery_features, distmat, query=query, gallery=gallery,
    #                            cmc_flag=cmc_flag, cuhk03=cuhk03, use_gpu=self.use_gpu, **kwargs)
    #     if (not rerank):
    #         return results
    #
    #     print('Applying person re-ranking ...')
    #     distmat_qq = pairwise_distance(features, query, query, metric=metric)
    #     distmat_gg = pairwise_distance(features, gallery, gallery, metric=metric)
    #     distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy())
    #     return evaluate_all(
    #         query_features, gallery_features, distmat,
    #         query=query, gallery=gallery, cmc_flag=cmc_flag
    #     )



    def evaluate(self, data_loader, query, gallery, metric=None, cmc_flag=False,
                 rerank=False, pre_features=None, cuhk03=False):
        from reid.CEval import compute_distance_matrix, fast_evaluate_rank

        print("... extract features")
        features, _, other_feat = extract_extra_features(
            self.model, data_loader,
            self.args
        )

        print("calculating the distance...")
        query_features = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in query], 0)
        gallery_features = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in gallery], 0)

        distance_matrix = compute_distance_matrix(
            query_features, gallery_features, 'euclidean'
        )
        distance_matrix = distance_matrix.data.cpu().numpy()

        query_ids = [pid for _, pid, _, _ in query]
        gallery_ids = [pid for _, pid, _, _ in gallery]
        query_cams = [cam for _, _, cam, _ in query]
        gallery_cams = [cam for _, _, cam, _ in gallery]

        CMC, mAP = fast_evaluate_rank(distance_matrix,
                                      np.array(query_ids),
                                      np.array(gallery_ids),
                                      np.array(query_cams),
                                      np.array(gallery_cams),
                                      max_rank=50,
                                      use_metric_cuhk03=False,
                                      use_cython=True)
        print('Mean AP: {:4.1%}'.format(mAP))
        print('Rank 1: {:4.1%}'.format(CMC[0]))
        return [CMC[0]], mAP
