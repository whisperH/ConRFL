from __future__ import division, print_function, absolute_import
import os.path as osp

from reid.utils.data.dataset1 import ImageDataset
from .data_loader import IncrementalPersonReIDSamples
# Log
# 22.01.2019
# - add v2
# - v1 and v2 differ in dir names
# - note that faces in v2 are blurred
TRAIN_DIR_KEY = 'train_dir'
TEST_DIR_KEY = 'test_dir'
VERSION_DICT = {
    'MSMT17_V1': {
        TRAIN_DIR_KEY: 'train',
        TEST_DIR_KEY: 'test',
    },
    'MSMT17_V2': {
        TRAIN_DIR_KEY: 'mask_train_v2',
        TEST_DIR_KEY: 'mask_test_v2',
    }
}

class MSMT17(IncrementalPersonReIDSamples):
    '''
    Market Dataset
    '''
    dataset_dir = 'msmt17'
    def __init__(self, datasets_root, relabel=True, combineall=False, **kwargs):
        self.domain_id = kwargs['domain_id']
        self.version = kwargs['version']
        self.relabel = relabel
        self.combineall = combineall

        self.dataset_dir = osp.join(datasets_root.replace("_DF", ""))

        # has_main_dir = False
        # for main_dir in VERSION_DICT:
        #     if osp.exists(osp.join(self.dataset_dir, main_dir)):
        #         train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
        #         test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
        #         has_main_dir = True
        #         break
        # assert has_main_dir, 'Dataset folder not found'
        main_dir = "MSMT17_V2"
        train_dir = VERSION_DICT[main_dir][TRAIN_DIR_KEY]
        test_dir = VERSION_DICT[main_dir][TEST_DIR_KEY]
        self.train_dir = osp.join(self.dataset_dir, main_dir, train_dir)
        self.test_dir = osp.join(self.dataset_dir, main_dir, test_dir)
        self.list_train_path = osp.join(
            self.dataset_dir, main_dir, 'list_train.txt'
        )
        self.list_val_path = osp.join(
            self.dataset_dir, main_dir, 'list_val.txt'
        )
        self.list_query_path = osp.join(
            self.dataset_dir, main_dir, 'list_query.txt'
        )
        self.list_gallery_path = osp.join(
            self.dataset_dir, main_dir, 'list_gallery.txt'
        )

        train = self.process_dir(self.train_dir, self.list_train_path)
        val = self.process_dir(self.train_dir, self.list_val_path)
        query = self.process_dir(self.test_dir, self.list_query_path)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path)

        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        if self.combineall:
            train += val
        self.train, self.query, self.gallery = train, query, gallery
        self._show_info(self.train, self.query, self.gallery)
        super(MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2]) - 1  # index starts from 0
            img_path = osp.join(dir_path, img_path)
            data.append((img_path, pid, camid, self.domain_id))

        return data