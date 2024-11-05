from __future__ import absolute_import
from collections import defaultdict, OrderedDict

from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score

from ..utils import to_numpy, to_torch

import torch

def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


def cmc(distmat, query_ids=None, gallery_ids=None,
        query_cams=None, gallery_cams=None, max_rank=100,
        separate_camera_set=False,
        single_gallery_shot=False,
        first_match_break=False):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(max_rank)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
            #valid &= ((gallery_cams[indices[i]]//2 == query_cams[i]//2) & gallery_cams[indices[i]]%2 != query_cams[i]%2)
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= max_rank: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries


def mean_ap(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None, use_distmat=True):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))

        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)


def mean_ap_cuhk03(distmat, query_ids=None, gallery_ids=None,
            query_cams=None, gallery_cams=None, use_distmat=True):
    distmat = to_numpy(distmat)
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute AP for each query
    aps = []
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))

        y_true = matches[i, valid]
        y_score = -distmat[i][indices[i]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))
    if len(aps) == 0:
        raise RuntimeError("No valid query")
    return np.mean(aps)


def market1501_torch(
        distmat, query_features, gallery_features, query_ids, gallery_ids, query_cams, gallery_cams, max_rank,
        use_distmat, **kwargs
):

    post_process = kwargs.get('post_process', False)
    device = kwargs.get('device', None)
    if device is None:
        device = torch.device('cpu')


    _results = OrderedDict()

    q_feats = to_torch(query_features).to(device)
    g_feats = to_torch(gallery_features).to(device)
    query_ids = np.array(query_ids)
    gallery_ids = np.array(gallery_ids)
    query_cams = np.array(query_cams)
    gallery_cams = np.array(gallery_cams)
    distmat = to_numpy(distmat)

    if post_process:
        simi_topk = kwargs['simi_topk']
        simi_query_feats = kwargs.get('simi_query_feats', q_feats)
        simi_gallery_feats = kwargs.get('simi_gallery_feats', g_feats)
        self_match = kwargs.get('self_match', True)
        post_process_function = kwargs.get('post_process_function', None)

    n_q, n_g = distmat.shape
    if n_g < max_rank:
        max_rank = n_g
        print('Note: number of gallery samples is quite small, got {}'.format(n_g))

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # compute cmc curve for each query

    num_valid_queries = 0.  # number of valid query
    ret = np.zeros(max_rank)
    aps = []
    for q_idx in tqdm(range(n_q)):
        qemb = q_feats[q_idx].reshape((1, -1))
        iindices = indices[q_idx]

        # get query pid and camid
        q_pid = query_ids[q_idx]
        q_camid = query_cams[q_idx]

        # remove gallery samples that have the same pid and camid with query
        remove = (gallery_ids[iindices] == q_pid) & (gallery_cams[iindices] == q_camid)
        keep = np.invert(remove)
        idistmat_keep = distmat[q_idx][iindices][keep]
        imatch_keep = matches[q_idx][keep]
        #===================================================================#
        if post_process:
            new_dist = do_post_process(
                post_process_function,
                qemb, g_feats,
                simi_query_feats[q_idx], simi_gallery_feats,
                iindices, keep,
                simi_topk, self_match, device
            )

            new_order = np.argsort(new_dist, axis=1)
            imatch_keep[:simi_topk] = imatch_keep[:simi_topk][new_order]
            # print("new order", new_order)
            # print("old idist", idistmat_keep[:25])
            # print("old idist new order", idistmat_keep[:simi_topk][new_order])
            # print("new dist", new_dist)
            idistmat_keep[:simi_topk] = new_dist
            # print("after update", idistmat_keep[:25])
        #===================================================================#
        y_true = imatch_keep
        y_score = -idistmat_keep

        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))

        repeat = 1
        for _ in range(repeat):
            index = np.nonzero(imatch_keep)[0]
            for j, k in enumerate(index):
                if k - j >= max_rank:
                    break
                ret[k - j] += 1
                break
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries, np.mean(aps)


# def market1501_torch(
#         distmat, query_features, gallery_features, query_ids, gallery_ids, query_cams, gallery_cams, max_rank,
#         use_distmat, **kwargs
# ):
#
#     post_process = kwargs.get('post_process', False)
#     device = kwargs.get('device', None)
#     if device is None:
#         device = torch.device('cpu')
#
#
#     _results = OrderedDict()
#
#     q_feats = to_torch(query_features).to(device)
#     g_feats = to_torch(gallery_features).to(device)
#     query_ids = np.array(query_ids)
#     gallery_ids = np.array(gallery_ids)
#     query_cams = np.array(query_cams)
#     gallery_cams = np.array(gallery_cams)
#
#     if post_process:
#         simi_topk = kwargs['simi_topk']
#         simi_query_feats = kwargs.get('simi_query_feats', q_feats)
#         simi_gallery_feats = kwargs.get('simi_query_feats', g_feats)
#         self_match = kwargs.get('self_match', True)
#         post_process_function = kwargs.get('post_process_function', None)
#
#     n_q, n_g = distmat.shape
#     if n_g < max_rank:
#         max_rank = n_g
#         print('Note: number of gallery samples is quite small, got {}'.format(n_g))
#
#     indices = np.argsort(distmat.cpu().numpy(), axis=1)
#     # compute cmc curve for each query
#     all_cmc = []
#     all_AP = []
#     all_INP = []
#     num_valid_q = 0.  # number of valid query
#
#     for q_idx in tqdm(range(n_q)):
#         qemb = q_feats[q_idx].reshape((1, -1))
#         # get query pid and camid
#         q_pid = query_ids[q_idx]
#         q_camid = query_cams[q_idx]
#
#         # remove gallery samples that have the same pid and camid with query
#         order = indices[q_idx]
#         remove = (gallery_ids[order] == q_pid) & (gallery_cams[order] == q_camid)
#         keep = np.invert(remove)
#
#         matches = (gallery_ids[order] == q_pid).astype(np.int32)
#         raw_cmc = matches[keep]  # binary vector, positions with value 1 are correct matches
#         if not np.any(raw_cmc):
#             # this condition is true when query identity does not appear in gallery
#             continue
#
#         #===================================================================#
#         if post_process:
#             new_dist = do_post_process(
#                 post_process_function,
#                 qemb,
#                 simi_query_feats[q_idx], simi_gallery_feats,
#                 order, keep,
#                 simi_topk, self_match, device
#             )
#
#             new_order = torch.argsort(new_dist, axis=1).cpu().numpy()
#             raw_cmc[:max_rank] = raw_cmc[:max_rank][new_order]
#         #===================================================================#
#         cmc = raw_cmc.cumsum()
#
#         pos_idx = np.where(raw_cmc == 1)
#         max_pos_idx = np.max(pos_idx)
#         inp = cmc[max_pos_idx] / (max_pos_idx + 1.0)
#         all_INP.append(inp)
#
#         cmc[cmc > 1] = 1
#
#         all_cmc.append(cmc[:max_rank])
#         num_valid_q += 1.
#
#         # compute average precision
#         # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#         num_rel = raw_cmc.sum()
#         tmp_cmc = raw_cmc.cumsum()
#         tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
#         tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)
#
#     assert num_valid_q > 0, 'Error: all query identities do not appear in gallery'
#
#     all_cmc = np.asarray(all_cmc).astype(np.float32)
#     all_cmc = all_cmc.sum(0) / num_valid_q
#
#     mAP = np.mean(all_AP)
#     # mINP = np.mean(all_INP)
#
#     return all_cmc, mAP




def cuhk03_torch(
        distmat, query_features, gallery_features, query_ids, gallery_ids, query_cams, gallery_cams, max_rank,
        use_distmat, **kwargs
):
    post_process = kwargs.get('post_process', False)
    device = kwargs.get('device', None)
    if device is None:
        device = torch.device('cpu')

    _results = OrderedDict()

    q_feats = to_torch(query_features).to(device)
    g_feats = to_torch(gallery_features).to(device)
    query_ids = np.array(query_ids)
    gallery_ids = np.array(gallery_ids)
    query_cams = np.array(query_cams)
    gallery_cams = np.array(gallery_cams)

    if post_process:
        simi_topk = kwargs['simi_topk']
        simi_query_feats = kwargs.get('simi_query_feats', q_feats)
        simi_gallery_feats = kwargs.get('simi_gallery_feats', g_feats)
        self_match = kwargs.get('self_match', True)
        post_process_function = kwargs.get('post_process_function', None)

    n_q, n_g = distmat.shape
    if n_g < max_rank:
        max_rank = n_g
        print('Note: number of gallery samples is quite small, got {}'.format(n_g))

    indices = np.argsort(distmat.cpu().numpy(), axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # compute cmc curve for each query
    ret = np.zeros(max_rank)
    num_valid_queries = 0

    aps = []
    for q_idx in tqdm(range(n_q)):
        qemb = q_feats[q_idx].reshape((1, -1))
        imatch = matches[q_idx]
        iindices = indices[q_idx]
        idistmat = distmat[q_idx]
        #===================================================================#

        valid = ((gallery_ids[indices[q_idx]] != query_ids[q_idx]) |
                 (gallery_cams[indices[q_idx]] != query_cams[q_idx]))
        # Filter out samples from same camera
        cmc_valid = valid & (gallery_cams[indices[q_idx]] != query_cams[q_idx])

        #===================================================================#
        if post_process:
            new_dist = do_post_process(
                post_process_function,
                qemb, g_feats,
                simi_query_feats[q_idx], simi_gallery_feats,
                indices[q_idx], valid,
                simi_topk, self_match, device
            )
            new_order = np.argsort(new_dist, axis=1)

            imatch[:simi_topk] = imatch[:simi_topk][new_order.squeeze()]
            iindices[:simi_topk] = iindices[:simi_topk][new_order.squeeze()]
            idistmat[:simi_topk] = idistmat[:simi_topk][new_order.squeeze()]

        #===================================================================#

        y_true = matches[q_idx, valid]
        y_score = -distmat[q_idx][indices[q_idx]][valid]
        if not np.any(y_true): continue
        aps.append(average_precision_score(y_true, y_score))

        if post_process:
            if not (valid == cmc_valid).all():
                new_dist = do_post_process(
                    post_process_function,
                    qemb, g_feats,
                    simi_query_feats[q_idx], simi_gallery_feats,
                    indices[q_idx], cmc_valid,
                    simi_topk, self_match, device
                )
                new_order = np.argsort(new_dist, axis=1)
                iindices[:simi_topk] = iindices[:simi_topk][new_order.squeeze()]
                imatch[:simi_topk] = imatch[:simi_topk][new_order.squeeze()]
        repeat = 10
        gids = gallery_ids[iindices[cmc_valid]]
        inds = np.where(cmc_valid)[0]
        ids_dict = defaultdict(list)
        for j, x in zip(inds, gids):
            ids_dict[x].append(j)

        for _ in range(repeat):
            # Randomly choose one instance for each id
            sampled = (cmc_valid & _unique_sample(ids_dict, len(cmc_valid)))
            index = np.nonzero(imatch[sampled])[0]

            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= max_rank: break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries, np.mean(aps)


def do_post_process(
        post_process_function,
        qemb, g_feats,
        simi_query_feat, simi_gallery_feats,
        old_order, valid_mask,
        simi_topk, self_match, device, measure='cos'
):
    simi_gallery_top_n = simi_gallery_feats[old_order][valid_mask][:simi_topk, :]
    old_g_feats = g_feats[old_order][valid_mask][:simi_topk,:]
    if self_match:
        simi_query_feat = simi_gallery_top_n

    with torch.no_grad():
        score, _ = post_process_function(
            simi_query_feat.to(device),
            simi_gallery_top_n.to(device)
        )
        new_g_feats = process_feat(score, old_g_feats, act='softmax')
    m, n = qemb.size(0), new_g_feats.size(0)
    if measure == "L2":
        new_dist = torch.pow(qemb, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(new_g_feats, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        new_dist.addmm_(1, -2, qemb, new_g_feats.t())
    else:
        input1_normed = F.normalize(qemb, p=2, dim=1)
        input2_normed = F.normalize(new_g_feats, p=2, dim=1)
        new_dist = -torch.mm(input1_normed, input2_normed.t())
    return new_dist.cpu().numpy()