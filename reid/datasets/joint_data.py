from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

class JointTrain(BaseImageDataset):
    """
    """

    def __init__(self, root, verbose=True, **kwargs):
        super(JointTrain, self).__init__()
        self.name_list = kwargs['name_list']
        self.dataset_dir = None
        self.train_dir = []
        self.train = []
        self.num_train_imgs = 0
        self.num_train_pids = 0
        self.camid = 0
        self.pid_list = []


