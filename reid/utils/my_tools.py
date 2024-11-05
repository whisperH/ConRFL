import torch
import torch.nn.functional as F

from reid.utils.data.preprocessor import Preprocessor
from reid.utils.data.transforms import transforms as T
from torch.utils.data import DataLoader
from .data.sampler import RandomIdentitySampler, MultiDomainRandomIdentitySampler
from sklearn.cluster import KMeans
import collections
import numpy as np
# import faiss

post_header_list = []
part_header_list = [
    'part', 'partsft', 'partTransMatch', 'forgetpart',
    'partDomainCTR', 'partPMatch', 'ncforgetpart',
    'midpart', 'midGMP_part', 'ncforgetmidpart', 'midsftpart', 'midcosegpart'
]
def set_postprocess_input(args, model, query, gallery, other_feat):
    header_name = args.header
    if header_name == 'sft':
        return {
            'simi_topk': 10,
            'self_match': True,
            'post_process_function': model.module.feat_compute,
            "post_process": args.post_process,
            "device": model.src_device_obj
        }
    elif header_name in ['partsft', 'partl2ft']:
        return {
            'simi_topk': 10,
            'self_match': True,
            'post_process_function': model.module.feat_compute,
            "post_process": args.post_process,
            "device": model.src_device_obj
        }
    # elif header_name == 'partTransMatch':
    #     if args.match_feat in ['region_feature', 'backbone_feat']:
    #         if args.post_process:
    #             simi_query_feats = torch.cat([other_feat[f].unsqueeze(0) for f, _, _, _ in query], 0)
    #             simi_gallery_feats = torch.cat([other_feat[f].unsqueeze(0) for f, _, _, _ in gallery], 0)
    #             return {
    #                 'simi_topk': 10,
    #                 'simi_query_feats': simi_query_feats,
    #                 'simi_gallery_feats': simi_gallery_feats,
    #                 'self_match': True,
    #                 'post_process_function': model.module.feat_compute,
    #                 "post_process": args.post_process,
    #                 "device": model.src_device_obj
    #             }
    #
    #     return {}
    else:
        print("no post process")
        return {}

def extract_features(model, data_loader, header):
    features_all = []
    global_feature_all = []
    part_features_all = []
    labels_all = []
    fnames_all = []
    camids_all = []
    model.eval()
    with torch.no_grad():
        for i, (imgs, fnames, pids, cids, domains) in enumerate(data_loader):
            model_outputs = model(imgs)
            if header in part_header_list:
                bn_feat = model_outputs['bn_feat']
                bn_feat_part = model_outputs['bn_feat_part']
                features = torch.cat((bn_feat, bn_feat_part), 1)
                global_features = bn_feat
                part_features = bn_feat_part
                for fname, feature, global_feature, part_feature, pid, cid in zip(fnames, features, global_features, part_features, pids, cids):
                    features_all.append(feature)
                    global_feature_all.append(global_feature)
                    part_features_all.append(part_feature)
                    labels_all.append(int(pid))
                    fnames_all.append(fname)
                    camids_all.append(cid)
            elif header == 'embedding':
                features = model_outputs['bn_feat']
                for fname, feature, pid, cid in zip(fnames, features, pids, cids):
                    features_all.append(feature)
                    labels_all.append(int(pid))
                    fnames_all.append(fname)
                    camids_all.append(cid)
            elif header == 'sft':
                features = model_outputs['bn_feat']
                for fname, feature, pid, cid in zip(fnames, features, pids, cids):
                    features_all.append(feature)
                    labels_all.append(int(pid))
                    fnames_all.append(fname)
                    camids_all.append(cid)
    model.train()
    return {
               "features_all": features_all,
               "global_feature_all": global_feature_all,
               "part_features_all": part_features_all
           }, labels_all, fnames_all, camids_all

def get_faiss_module(in_dim):
    pass
    # cpu_index = faiss.IndexFlatL2(in_dim)  # build a flat (CPU) index
    # ngpus = faiss.get_num_gpus()
    # if ngpus == 1:
    #     res = faiss.StandardGpuResources()
    #     gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    # else:
    #     gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    #         cpu_index
    #     )
    # return gpu_index

def get_kmeans(in_dim, args, featlist, proposal_centroid):
    kmeans_n_iter, proposal_parts = args.kmeans_n_iter, args.proposal_parts
    # clus = faiss.Clustering(in_dim, K)

    # if torch.nonzero(proposal_centroid).size(0) != 0:
    #     clus.int_centroids = proposal_centroid.cpu().numpy().astype('float32')
    #
    # clus.seed  = np.random.randint(1)
    # clus.niter = kmeans_n_iter
    # clus.max_points_per_centroid = 10000000
    # clus.train(featlist, index)
    # ################################################################################### #
    # if not inference:
    #     km = KMeans(n_clusters=proposal_parts, random_state=args.seed).fit(featlist)
    # ################################################################################### #
    pass
    # km = faiss.Kmeans(in_dim, proposal_parts, niter=kmeans_n_iter)
    # km.cp.spherical = False
    # faiss.normalize_L2(featlist)
    # if torch.nonzero(proposal_centroid).size(0) != 0:
    #     km.train(featlist, init_centroids=proposal_centroid.cpu().numpy().astype('float32'))
    # else:
    #     km.train(featlist)
    # D, I = km.index.search(featlist, 1)
    #
    # data_count = np.zeros(proposal_parts)
    # for k in np.unique(I):
    #     idx_k = np.where(I == k)[0]
    #     data_count[k] += len(idx_k)
    # return km, data_count


def do_kmeans(training, featslist, km_obj, iteration, proposal_centroid):
    # https://github.com/facebookresearch/faiss/wiki/Brute-force-search-without-an-index
    pass
    # if training:
    #     if iteration == 0 :
    #         search_v = featslist.cpu().numpy().astype('float32')
    #         faiss.normalize_L2(search_v)
    #         D, I = km_obj.index.search(search_v, 1)
    #         D = torch.from_numpy(D).to(featslist.device)
    #         I = torch.from_numpy(I).to(featslist.device)
    #     else:
    #         featslist = F.normalize(featslist)
    #         norms_xq = (featslist ** 2).sum(axis=1)
    #         norms_xb = (proposal_centroid ** 2).sum(axis=1)
    #         distances = norms_xq.reshape(-1, 1) + norms_xb -2 * featslist @ proposal_centroid.T
    #         D, I = torch.topk(distances, 1, largest=False)
    #     # km_obj.index.reset()
    #     # km_obj.index.add(
    #     #     proposal_centroid
    #     # )
    #     # D, I = km_obj.index.search(featslist, 1)
    # else:
    #     featslist = F.normalize(featslist)
    #     norms_xq = (featslist ** 2).sum(axis=1)
    #     norms_xb = (proposal_centroid ** 2).sum(axis=1)
    #     distances = norms_xq.reshape(-1, 1) + norms_xb -2 * featslist @ proposal_centroid.T
    #     D, I = torch.topk(distances, 1, largest=False)
    # return I, D

def initial_classifier(model, data_loader, method='center', header='global'):
    if method == 'center':
        pid2features = collections.defaultdict(list)
        pid2part_feature = collections.defaultdict(list)
        features, labels_all, fnames_all, camids_all = extract_features(model, data_loader, header)

        features_all = features['features_all']
        global_feature_all = features['global_feature_all']
        part_features_all = features['part_features_all']

        if header in part_header_list:
            for global_feature, part_feature, pid in zip(global_feature_all, part_features_all, labels_all):
                pid2features[pid].append(global_feature)
                pid2part_feature[pid].append(part_feature)
        elif header in ['embedding', 'sft']:
            for feature_all, pid in zip(features_all, labels_all):
                pid2features[pid].append(feature_all)

        if header in part_header_list:
            class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
            class_centers = torch.stack(class_centers)

            class_part_centers = [torch.stack(pid2part_feature[pid]).mean(0) for pid in sorted(pid2part_feature.keys())]
            class_part_centers = torch.stack(class_part_centers)
            return [
                F.normalize(class_centers, dim=1).float().cuda(),
                F.normalize(class_part_centers, dim=1).float().cuda()
            ]
        elif header in ['embedding', 'sft']:
            class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
            class_centers = torch.stack(class_centers)
            return [
                F.normalize(class_centers, dim=1).float().cuda()
            ]

    elif method == 'SVD':
        if header=='part':
            class_weight = SVD_weight(model.module.heads.classifier)
            class_part_weight = SVD_weight(model.module.heads.classifier_part)
            model.train()
            return [
                class_weight, class_part_weight
            ]
        else:
            class_weight = SVD_weight(model.module.heads.classifier)
            model.train()
            return [class_weight]

def SVD_weight(layer):
    with torch.no_grad():
        # NECESSARY! The weight of Linear layer has been transposed!
        A = layer.weight.t()
        M, N = A.size()
        M: 2048
        N: 1024
        U, S, V = torch.svd(A, some=False)
        W = A @ V
        W: '2048 x 1024 = M x N'

        NW = torch.zeros_like(A)

        for i in range(N):

            curr_N = W.size(1)

            W_norm = torch.norm(W, p=2, dim=0)
            W_norm: 'curr_N'

            index = i
            vec_i = A[:, i]
            vec_i_norm = torch.norm(vec_i)

            co = (A[:, i].view(M, 1).t() @ W).view(curr_N)
            co: 'curr_N'
            co = co / vec_i_norm
            absco = abs(co / W_norm)
            maxco_index = torch.max(absco, 0)[1].item()

            NW[:, index] = W[:, maxco_index] * torch.sign(co[maxco_index])

            # Remove selected column vector from W
            W = W[:, sorted({x for x in range(curr_N) if x != maxco_index})]

        layer.weight.copy_(NW.t())

    return layer.weight

def select_replay_samples(
        model, dataset, training_phase=0, add_num=0,
        old_datas=None, select_samples=2, batch_size=128, num_workers=4,
        header='embedding'
):
    replay_data = []
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    train_transformer = T.Compose([
        T.Resize((256, 128), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_loader = DataLoader(Preprocessor(dataset.train, root=dataset.images_dir,transform=transformer),
                              batch_size=batch_size, num_workers=num_workers,
                              shuffle=True, pin_memory=True, drop_last=False)

    features, labels_all, fnames_all, camids_all = extract_features(model, train_loader, header)
    features_all = features['features_all']

    pid2features = collections.defaultdict(list)
    pid2fnames = collections.defaultdict(list)
    pid2cids = collections.defaultdict(list)

    for feature, pid, fname, cid in zip(features_all, labels_all, fnames_all, camids_all):
        pid2features[pid].append(feature)
        pid2fnames[pid].append(fname)
        pid2cids[pid].append(cid)

    labels_all = list(set(labels_all))

    class_centers = [torch.stack(pid2features[pid]).mean(0) for pid in sorted(pid2features.keys())]
    class_centers = F.normalize(torch.stack(class_centers), dim=1)
    select_pids = np.random.choice(labels_all, 250, replace=False)
    for pid in select_pids:
        feautures_single_pid = F.normalize(torch.stack(pid2features[pid]), dim=1, p=2)
        center_single_pid = class_centers[pid]
        simi = torch.mm(feautures_single_pid, center_single_pid.unsqueeze(0).t())
        simi_sort_inx = torch.sort(simi, dim=0)[1][:2]
        for id in simi_sort_inx:
            replay_data.append((pid2fnames[pid][id], pid+add_num, pid2cids[pid][id], training_phase-1))

    if old_datas is None:
        data_loader_replay = DataLoader(Preprocessor(replay_data, dataset.images_dir, train_transformer),
                             batch_size=batch_size, num_workers=num_workers, sampler=RandomIdentitySampler(replay_data, select_samples),
                             pin_memory=True, drop_last=True)
    else:
        replay_data.extend(old_datas)
        data_loader_replay = DataLoader(Preprocessor(replay_data, dataset.images_dir, train_transformer),
                             batch_size=training_phase*batch_size, num_workers=num_workers,
                             sampler=MultiDomainRandomIdentitySampler(replay_data, select_samples),
                             pin_memory=True, drop_last=True)
    return data_loader_replay, replay_data


def get_pseudo_features(data_specific_batch_norm, training_phase, x, domain, unchange=False):
    fake_feat_list = []
    if unchange is False:
        for i in range(training_phase):
            if int(domain[0]) == i:
                data_specific_batch_norm[i].train()
                fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
            else:
                data_specific_batch_norm[i].eval()
                fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])
                data_specific_batch_norm[i].train()
    else:
        for i in range(training_phase):
            data_specific_batch_norm[i].eval()
            fake_feat_list.append(data_specific_batch_norm[i](x)[..., 0, 0])

    return fake_feat_list


def log_accuracy(pred_class_logits, gt_classes, topk=(1,)):
    """
    Log the accuracy metrics to EventStorage.
    """
    bsz = pred_class_logits.size(0)
    maxk = max(topk)
    _, pred_class = pred_class_logits.topk(maxk, 1, True, True)
    pred_class = pred_class.t()
    correct = pred_class.eq(gt_classes.view(1, -1).expand_as(pred_class))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / bsz))
    return ret[0]


def domain_class_map(each_domain_class_num):
    map_list = {}
    for i in range(len(each_domain_class_num)-1):
        if i == 0:
            start_pos = 0
        else:
            start_pos = start_pos + each_domain_class_num[i-1]

        class_num = each_domain_class_num[i] + each_domain_class_num[i+1]
        map_list[i] = [start_pos, start_pos+class_num]

    return map_list


def simple_transform(x, beta):
    x = 1/torch.pow(torch.log(1/x+1),beta)
    return x

def extended_simple_transform(x, beta):
    zero_tensor = torch.zeros_like(x)
    x_pos = torch.maximum(x, zero_tensor)
    x_neg = torch.minimum(x, zero_tensor)
    x_pos = 1/torch.pow(torch.log(1/(x_pos+1e-5)+1),beta)
    x_neg = -1/torch.pow(torch.log(1/(-x_neg+1e-5)+1),beta)
    return x_pos+x_neg


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)