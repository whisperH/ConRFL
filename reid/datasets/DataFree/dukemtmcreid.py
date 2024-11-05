from __future__ import division, print_function, absolute_import
import os
import copy
import os.path as osp
from reid.datasets.DataFree.data_loader import IncrementalPersonReIDSamples
from reid.utils.data.dataset1 import ImageDataset
import re
import glob

class DukeMTMCreID(IncrementalPersonReIDSamples):
    '''
    Duke dataset
    '''
    duke_path = 'dukemtmc-reid/DukeMTMC-reID/'
    def __init__(self, datasets_root, relabel=True, combineall=False, **kwargs):
        self.domain_id = kwargs['domain_id']
        self.relabel = relabel
        self.combineall = combineall
        root = osp.join(datasets_root.replace("_DF", ""))
        self.train_dir = osp.join(
            root, 'bounding_box_train'
        )
        self.query_dir = osp.join(root, 'query')
        self.gallery_dir = osp.join(
            root, 'bounding_box_test'
        )

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(train, query, gallery)
        super(DukeMTMCreID, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 1 <= camid <= 8
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid, self.domain_id))

        return data


