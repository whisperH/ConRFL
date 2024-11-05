from __future__ import division, print_function, absolute_import
import copy
import numpy as np
import os.path as osp
import tarfile
import zipfile
import torch
import os
import errno
import time
import sys
from PIL import Image
from prettytable import PrettyTable

def download_url(url, dst):
    """Downloads file from a url to a destination.

    Args:
        url (str): url to download file.
        dst (str): destination path.
    """
    from six.moves import urllib
    print('* url="{}"'.format(url))
    print('* destination="{}"'.format(dst))

    def _reporthook(count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024*duration))
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(
            '\r...%d%%, %d MB, %d KB/s, %d seconds passed' %
            (percent, progress_size / (1024*1024), speed, duration)
        )
        sys.stdout.flush()

    urllib.request.urlretrieve(url, dst, _reporthook)
    sys.stdout.write('\n')


def read_image(path):
    """Reads image from path using ``PIL.Image``.

    Args:
        path (str): path to an image.

    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img

def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset`` and ``VideoDataset``.

    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        k_tfm (int): number of times to apply augmentation to an image
            independently. If k_tfm > 1, the transform function will be
            applied k_tfm times to an image. This variable will only be
            useful for training and is currently valid for image datasets only.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """

    # junk_pids contains useless person IDs, e.g. background,
    # false detections, distractors. These IDs will be ignored
    # when combining all images in a dataset for training, i.e.
    # combineall=True
    _junk_pids = []

    def __init__(
        self,
        train,
        query,
        gallery,
        transform=None,
        k_tfm=1,
        mode='train',
        combineall=False,
        verbose=True,
        **kwargs
    ):
        # extend 3-tuple (img_path(s), pid, camid) to
        # 4-tuple (img_path(s), pid, camid, dsetid) by
        # adding a dataset indicator "dsetid"
        if len(train[0]) == 3:
            train = [(*items, 0) for items in train]
        if len(query[0]) == 3:
            query = [(*items, 0) for items in query]
        if len(gallery[0]) == 3:
            gallery = [(*items, 0) for items in gallery]

        self.train = train
        self.query = query
        self.gallery = gallery
        self.transform = transform
        self.k_tfm = k_tfm
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)
        self.num_datasets = self.get_num_datasets(self.train)

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError(
                'Invalid mode. Got {}, but expected to be '
                'one of [train | query | gallery]'.format(self.mode)
            )

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        train = copy.deepcopy(self.train)

        for img_path, pid, camid, dsetid in other.train:
            pid += self.num_train_pids
            camid += self.num_train_cams
            dsetid += self.num_datasets
            train.append((img_path, pid, camid, dsetid))

        ###################################
        # Note that
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset; setting it to True will
        #    create new IDs that should have already been included
        ###################################
        if isinstance(train[0][0], str):
            return ImageDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False
            )
        else:
            return VideoDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False,
                seq_len=self.seq_len,
                sample_method=self.sample_method
            )

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def get_num_pids(self, data):
        """Returns the number of training person identities.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        pids = set()
        for items in data:
            pid = items[1]
            pids.add(pid)
        return len(pids)

    def get_num_cams(self, data):
        """Returns the number of training cameras.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        cams = set()
        for items in data:
            camid = items[2]
            cams.add(camid)
        return len(cams)

    def get_num_datasets(self, data):
        """Returns the number of datasets included.

        Each tuple in data contains (img_path(s), pid, camid, dsetid).
        """
        dsets = set()
        for items in data:
            dsetid = items[3]
            dsets.add(dsetid)
        return len(dsets)

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        for items in self.gallery:
            pid = items[1]
            if pid in self._junk_pids:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            for img_path, pid, camid, dsetid in data:
                if pid in self._junk_pids:
                    continue
                pid = pid2label[pid] + self.num_train_pids
                combined.append((img_path, pid, camid, dsetid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def download_dataset(self, dataset_dir, dataset_url):
        """Downloads and extracts dataset.

        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please follow the '
                'document to prepare this dataset'.format(
                    self.__class__.__name__
                )
            )

        print('Creating directory "{}"'.format(dataset_dir))
        mkdir_if_missing(dataset_dir)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))

        print(
            'Downloading {} dataset to "{}"'.format(
                self.__class__.__name__, dataset_dir
            )
        )
        download_url(dataset_url, fpath)

        print('Extracting "{}"'.format(fpath))
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dataset_dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  train    | {:5d} | {:7d} | {:9d}\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image/video dataset\n'.format(
                  num_train_pids, len(self.train), num_train_cams,
                  num_query_pids, len(self.query), num_query_cams,
                  num_gallery_pids, len(self.gallery), num_gallery_cams
              )

        return msg

    def _transform_image(self, tfm, k_tfm, img0):
        """Transforms a raw image (img0) k_tfm times with
        the transform function tfm.
        """
        img_list = []

        for k in range(k_tfm):
            img_list.append(tfm(img0))

        img = img_list
        if len(img) == 1:
            img = img[0]

        return img

    @property
    def images_dir(self):
        return osp.join(self.root, 'images')


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):
        img_path, pid, camid, dsetid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self._transform_image(self.transform, self.k_tfm, img)
        item = {
            'img': img,
            'pid': pid,
            'camid': camid,
            'impath': img_path,
            'dsetid': dsetid
        }
        return item

    def show_summary(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print(
            '  train    | {:5d} | {:8d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:8d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:8d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  ----------------------------------------')


class VideoDataset(Dataset):
    """A base class representing VideoDataset.

    All other video datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``imgs``, ``pid`` and ``camid``
    where ``imgs`` has shape (seq_len, channel, height, width). As a result,
    data in each batch has shape (batch_size, seq_len, channel, height, width).
    """

    def __init__(
        self,
        train,
        query,
        gallery,
        seq_len=15,
        sample_method='evenly',
        **kwargs
    ):
        super(VideoDataset, self).__init__(train, query, gallery, **kwargs)
        self.seq_len = seq_len
        self.sample_method = sample_method

        if self.transform is None:
            raise RuntimeError('transform must not be None')

    def __getitem__(self, index):
        img_paths, pid, camid, dsetid = self.data[index]
        num_imgs = len(img_paths)

        if self.sample_method == 'random':
            # Randomly samples seq_len images from a tracklet of length num_imgs,
            # if num_imgs is smaller than seq_len, then replicates images
            indices = np.arange(num_imgs)
            replace = False if num_imgs >= self.seq_len else True
            indices = np.random.choice(
                indices, size=self.seq_len, replace=replace
            )
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)

        elif self.sample_method == 'evenly':
            # Evenly samples seq_len images from a tracklet
            if num_imgs >= self.seq_len:
                num_imgs -= num_imgs % self.seq_len
                indices = np.arange(0, num_imgs, num_imgs / self.seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_imgs)
                num_pads = self.seq_len - num_imgs
                indices = np.concatenate(
                    [
                        indices,
                        np.ones(num_pads).astype(np.int32) * (num_imgs-1)
                    ]
                )
            assert len(indices) == self.seq_len

        elif self.sample_method == 'all':
            # Samples all images in a tracklet. batch_size must be set to 1
            indices = np.arange(num_imgs)

        else:
            raise ValueError(
                'Unknown sample method: {}'.format(self.sample_method)
            )

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0) # img must be torch.Tensor
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        item = {'img': imgs, 'pid': pid, 'camid': camid, 'dsetid': dsetid}

        return item

    def show_summary(self):
        num_train_pids = self.get_num_pids(self.train)
        num_train_cams = self.get_num_cams(self.train)

        num_query_pids = self.get_num_pids(self.query)
        num_query_cams = self.get_num_cams(self.query)

        num_gallery_pids = self.get_num_pids(self.gallery)
        num_gallery_cams = self.get_num_cams(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  -------------------------------------------')
        print('  subset   | # ids | # tracklets | # cameras')
        print('  -------------------------------------------')
        print(
            '  train    | {:5d} | {:11d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:11d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:11d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  -------------------------------------------')



def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files

class IncrementalPersonReIDSamples:

    def _relabels_incremental(self, samples, label_index, is_mix=False):
        '''
        reorder labels
        map labels [1, 3, 5, 7] to [0,1,2,3]
        '''
        ids = []
        pid2label = {}
        for sample in samples:
            ids.append(sample[label_index])
        # delete repetitive elments and order
        ids = list(set(ids))
        ids.sort()

        # reorder
        for sample in samples:
            sample = list(sample)
            pid2label[sample[label_index]] = ids.index(sample[label_index])
        new_samples = copy.deepcopy(samples)
        for i, sample in enumerate(samples):
            new_samples[i] = list(new_samples[i])
            new_samples[i][label_index] = pid2label[sample[label_index]]
        if is_mix:
            return samples, pid2label
        else:
            return new_samples

    def _load_images_path(self, folder_dir, domain_name='market', is_mix=False):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_file_name(file_name, is_mix=is_mix)
                samples.append([root_path + file_name, identi_id, camera_id, domain_name])
        return samples

    def _analysis_file_name(self, file_name, is_mix=False):
        '''
        :param file_name: format like 0844_c3s2_107328_01.jpg
        :return: 0844, 3
        '''

        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        if is_mix:
            identi_id, camera_id = int(split_list[0]), int(split_list[2])
        else:
            identi_id, camera_id = int(split_list[0]), int(split_list[1])

        return identi_id, camera_id

    def _show_info(self, train, query, gallery, name=None, if_show=True):
        if if_show:
            def analyze(samples):
                pid_num = len(set([sample[1] for sample in samples]))
                cid_num = len(set([sample[2] for sample in samples]))
                sample_num = len(samples)
                return sample_num, pid_num, cid_num

            train_info = analyze(train)
            query_info = analyze(query)
            gallery_info = analyze(gallery)

            # please kindly install prettytable: ```pip install prettyrable```
            table = PrettyTable(['set', 'images', 'identities', 'cameras'])
            table.add_row([self.__class__.__name__ if name is None else name, '', '', ''])
            table.add_row(['train', str(train_info[0]), str(train_info[1]), str(train_info[2])])
            table.add_row(['query', str(query_info[0]), str(query_info[1]), str(query_info[2])])
            table.add_row(['gallery', str(gallery_info[0]), str(gallery_info[1]), str(gallery_info[2])])
            print(table)
        else:
            pass

def Incremental_combine_test_samples(samples_list):
    '''combine more than one samples (e.g. market.train and duke.train) as a samples'''

    all_gallery, all_query = [], []

    def _generate_relabel_dict(s_list):
        pids_in_list, pid2relabel_dict = [], {}
        for new_label, samples in enumerate(s_list):
            if str(samples[1]) + str(samples[3]) not in pids_in_list:
                pids_in_list.append(str(samples[1]) + str(samples[3]))
        for i, pid in enumerate(sorted(pids_in_list)):
            pid2relabel_dict[pid] = i
        return pid2relabel_dict
    def _replace_pid2relabel(s_list, pid2relabel_dict, pid_dimension=1):
        new_list = copy.deepcopy(s_list)
        for i, sample in enumerate(s_list):
            new_list[i] = list(new_list[i])
            new_list[i][pid_dimension] = pid2relabel_dict[str(sample[pid_dimension])+str(sample[pid_dimension + 2])]
        return new_list

    for samples_class in samples_list:
        all_gallery.extend(samples_class.gallery)
        all_query.extend(samples_class.query)
    pid2relabel_dict = _generate_relabel_dict(all_gallery)
    # pid2relabel_dict2 = _generate_relabel_dict(all_query)

    # assert len(list(pid2relabel_dict2.keys())) == sum([1 for query_key in pid2relabel_dict2.keys() if query_key in pid2relabel_dict.keys()])
    #print(pid2relabel_dict)
    #print(pid2relabel_dict2)
    # assert operator.eq(pid2relabel_dict, _generate_relabel_dict(all_query))
    gallery = _replace_pid2relabel(all_gallery, pid2relabel_dict, pid_dimension=1)
    query = _replace_pid2relabel(all_query, pid2relabel_dict, pid_dimension=1)


    return query, gallery