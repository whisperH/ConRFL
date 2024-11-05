from __future__ import absolute_import
from collections import defaultdict
import os
import time

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)
# from torch.utils.data import DataLoader
# from reid.utils.data.preprocessor import Preprocessor
# from reid.evaluators import extract_extra_features, pairwise_distance



def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _, _) in enumerate(data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(self.num_samples).tolist()
        ret = []
        for i in indices:
            pid = self.pids[i]
            t = self.index_dic[pid]
            if len(t) >= self.num_instances:
                t = np.random.choice(t, size=self.num_instances, replace=False)
            else:
                t = np.random.choice(t, size=self.num_instances, replace=True)
            ret.extend(t)
        return iter(ret)

class MultiDomainRandomIdentitySampler(Sampler):
    def __init__(self, data_source, num_instances):
        self.data_source = data_source
        self.num_instances = num_instances

        self.domain2pids = defaultdict(list)
        self.pid2index = defaultdict(list)

        for index, (_, pid, _, domain) in enumerate(data_source):
            if pid not in self.domain2pids[domain]:
                self.domain2pids[domain].append(pid)
            self.pid2index[pid].append(index)

        self.pids = list(self.pid2index.keys())
        self.domains = list(sorted(self.domain2pids.keys()))

        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):

        ret = []
        domain2pids = copy.deepcopy(self.domain2pids)
        for _ in range(8):
            for domain in self.domains:
                pids = np.random.choice(domain2pids[domain], size=8, replace=False)
                for pid in pids:
                    idxs = copy.deepcopy(self.pid2index[pid])
                    idxs = np.random.choice(idxs, size=2, replace=False)
                    ret.extend(idxs)
        return iter(ret)

class RandomMultipleGallerySampler(Sampler):
    def __init__(self, data_source, num_instances=4):
        self.data_source = data_source
        self.index_pid = defaultdict(int)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (_, pid, cam, frame) in enumerate(data_source):
            self.index_pid[index] = pid
            self.pid_cam[pid].append(cam)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_samples = len(self.pids)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.pids)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.pid_index[self.pids[kid]])

            _, i_pid, i_cam, _ = self.data_source[i]
            #_, i_pid, i_cam = self.data_source[i]

            ret.append(i)

            pid_i = self.index_pid[i]
            cams = self.pid_cam[pid_i]
            index = self.pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if (not select_indexes): continue
                if len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])


        return iter(ret)

class ClassUniformlySampler4Incremental(Sampler):
    '''
    random sample according to class label
    Arguments:
        data_source (Dataset): data_loader to sample from
        class_position (int): which one is used as class
        k (int): sample k images of each class
    '''

    def __init__(self, data_source, pid_list, class_position=1, k=4):

        self.data_source = data_source
        self.class_position = class_position
        self.k = k

        self.samples = self.data_source
        class_dict = self._tuple2dict(self.samples)
        self.class_dict = self.filter_current_step_pids(class_dict, pid_list)

    def __iter__(self):
        self.sample_list = self._generate_list(self.class_dict)
        return iter(self.sample_list)

    def __len__(self):
        return len(self.sample_list)

    def _tuple2dict(self, inputs):
        '''
        :param inputs: list with tuple elemnts, [(image_path1, class_index_1), (imagespath_2, class_index_2), ...]
        :return: dict, {class_index_i: [samples_index1, samples_index2, ...]}
        '''
        dict = {}
        for index, each_input in enumerate(inputs):
            class_index = each_input[self.class_position]
            if class_index not in list(dict.keys()):
                dict[class_index] = [index]
            else:
                dict[class_index].append(index)
        return dict


    def _generate_list(self, dict):
        '''
        :param dict: dict, whose values are list
        :return:
        '''

        sample_list = []

        dict_copy = dict.copy()
        keys = list(dict_copy.keys())
        random.shuffle(keys)
        for key in keys:
            value = dict_copy[key]
            if len(value) >= self.k:
                random.shuffle(value)
                sample_list.extend(value[0: self.k])
            else:
                value = value * self.k
                random.shuffle(value)
                sample_list.extend(value[0: self.k])

        return sample_list

    def filter_current_step_pids(self, class_dict, pid_list):
        update_dict = {}
        for pid in pid_list:
            assert pid in class_dict.keys()
            update_dict[pid] = class_dict[pid]

        return update_dict


# class GraphSampler(Sampler):
#     def __init__(self, data_source, img_path, transformer, model, args, batch_size=64, num_instance=4,
#                  gal_batch_size=256, prob_batch_size=256, verbose=False):
#         super(GraphSampler, self).__init__(data_source)
#         self.data_source = data_source
#         self.img_path = img_path
#         self.transformer = transformer
#         self.args = args
#         self.model = model
#         self.batch_size = batch_size
#         self.num_instance = num_instance
#         self.gal_batch_size = gal_batch_size
#         self.prob_batch_size = prob_batch_size
#         self.verbose = verbose
#
#         self.index_dic = defaultdict(list)
#
#
#
#         for index, (_, pid, _, _) in enumerate(data_source):
#             self.index_dic[pid].append(index)
#         self.pids = list(self.index_dic.keys())
#         self.num_pids = len(self.pids)
#         for pid in self.pids:
#             random.shuffle(self.index_dic[pid])
#
#         self.sam_index = None
#         self.sam_pointer = [0] * self.num_pids
#
#     def make_index(self):
#         start = time.time()
#         self.graph_index()
#         if self.verbose:
#             print('\nTotal GS time: %.3f seconds.\n' % (time.time() - start))
#
#     def calc_distance(self, dataset):
#         data_loader = DataLoader(
#             Preprocessor(dataset, self.img_path, transform=self.transformer),
#             batch_size=64, num_workers=8,
#             shuffle=False, pin_memory=True)
#
#         if self.verbose:
#             print('\t GraphSampler: ', end='\t')
#         features, labels, _ = extract_extra_features(self.model, data_loader, self.args)
#         features = torch.cat([features[fname].unsqueeze(0) for fname, _, _, _ in dataset], 0)
#
#         if self.verbose:
#             print('\t GraphSampler: \tCompute distance...', end='\t')
#         start = time.time()
#         dist = pairwise_distance(features)
#
#         if self.verbose:
#             print('Time: %.3f seconds.' % (time.time() - start))
#
#         return dist
#
#     def graph_index(self):
#         sam_index = []
#         for pid in self.pids:
#             index = np.random.choice(self.index_dic[pid], size=1)[0]
#             sam_index.append(index)
#
#         dataset = [self.data_source[i] for i in sam_index]
#         dist = self.calc_distance(dataset)
#
#         with torch.no_grad():
#             dist = dist + torch.eye(self.num_pids, device=dist.device) * 1e15
#             topk = self.batch_size // self.num_instance - 1
#             _, topk_index = torch.topk(dist.cuda(), topk, largest=False)
#             topk_index = topk_index.cpu().numpy()
#
#         sam_index = []
#         for i in range(self.num_pids):
#             id_index = topk_index[i, :].tolist()
#             id_index.append(i)
#             index = []
#             for j in id_index:
#                 pid = self.pids[j]
#                 img_index = self.index_dic[pid]
#                 len_p = len(img_index)
#                 index_p = []
#                 remain = self.num_instance
#                 while remain > 0:
#                     end = self.sam_pointer[j] + remain
#                     idx = img_index[self.sam_pointer[j] : end]
#                     index_p.extend(idx)
#                     remain -= len(idx)
#                     self.sam_pointer[j] = end
#                     if end >= len_p:
#                         random.shuffle(img_index)
#                         self.sam_pointer[j] = 0
#                 assert(len(index_p) == self.num_instance)
#                 index.extend(index_p)
#             sam_index.extend(index)
#
#         sam_index = np.array(sam_index)
#         sam_index = sam_index.reshape((-1, self.batch_size))
#         np.random.shuffle(sam_index)
#         sam_index = list(sam_index.flatten())
#         self.sam_index = sam_index
#
#     def __len__(self):
#         if self.sam_index is None:
#             return self.num_pids
#         else:
#             return len(self.sam_index)
#
#     def __iter__(self):
#         self.make_index()
#         return iter(self.sam_index)