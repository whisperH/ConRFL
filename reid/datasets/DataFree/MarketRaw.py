from __future__ import division, print_function, absolute_import
import os
import copy
from .data_loader import IncrementalPersonReIDSamples
import re
import glob
import os.path as osp
import warnings

class Market1501(IncrementalPersonReIDSamples):
    '''
    Market Dataset
    '''
    _junk_pids = [0, -1]
    dataset_dir = 'market1501/Market-1501-v15.09.15/'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    def __init__(self, datasets_root, relabel=True, combineall=False, **kwargs):
        self.relabel = relabel
        self.domain_id = kwargs['domain_id']
        self.combineall = combineall
        root = osp.join(datasets_root.replace("_DF", ""))
        self.train_dir = osp.join(root, 'bounding_box_train')
        self.query_dir = osp.join(root, 'query')
        self.gallery_dir = osp.join(root, 'bounding_box_test')
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)
        super(Market1501, self).__init__(train, query, gallery, **kwargs)


    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid, self.domain_id))

        return data
