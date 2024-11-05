from __future__ import print_function, absolute_import
import os.path as osp

from reid.utils.data import IterLoader
from reid.utils.data.sampler import RandomMultipleGallerySampler, ClassUniformlySampler4Incremental
from reid import datasets
from reid.utils.my_tools import *
from reid.utils.data.transforms import build_transforms
import copy
from PIL import Image

def get_data(name, args, **kwargs):

    data_dir = args.data_dir
    batch_size = args.batch_size
    workers = args.workers
    num_instances = args.num_instances

    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, **kwargs)

    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = build_transforms(args, is_train=True)
    #     T.Compose([
    #     T.Resize((height, width), interpolation=3),
    #     T.RandomHorizontalFlip(p=0.5),
    #     T.Pad(10),
    #     T.RandomCrop((height, width)),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225]),
    #     T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    # ])

    test_transformer = build_transforms(args, is_train=False)
    #     # T.Compose([
    #     T.Resize((height, width), interpolation=3),
    #     T.ToTensor(),
    #     T.Normalize(mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225])
    # ])

    rmgs_flag = num_instances > 0
    if args.use_replay:
        if rmgs_flag:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
        else:
            sampler = None
    else:
        domain_id = kwargs.get("domain_id", -1)
        if domain_id == -1:
            sampler = None
        else:
            pid_list = sorted(list(dataset.pid_list))[0:args.num_identities_per_domain]
            sampler = ClassUniformlySampler4Incremental(train_set, pid_list, class_position=1, k=4)

    train_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True)
    train_IterLoader = IterLoader(train_loader, length=iters)

    test_loader = DataLoader(
        Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    init_loader = DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=test_transformer),
                             batch_size=128, num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)

    return dataset, num_classes, train_IterLoader, train_loader, test_loader, init_loader

def get_joint_train_data(name_list, args, **kwargs):
    data_dir = args.data_dir
    batch_size = args.batch_size
    workers = args.workers
    num_instances = args.num_instances

    dataset = datasets.create('joint_train', root='', name_list=name_list)
    dataset_camid = 0
    pid_list = []

    for id, data_name in enumerate(name_list):
        root = osp.join(data_dir, data_name)
        idataset = datasets.create(data_name, root=root, domain_id=0, **kwargs)
        max_camid = 0
        if hasattr(idataset, 'pid_list'):
            pid_list.extend(
                [_+dataset.num_train_pids for _ in list(idataset.pid_list)[0:args.num_identities_per_domain]]
            )

        for iele in idataset.train:
            fname, pid, camid, domain = iele
            if root not in fname:
                if data_name == 'msmt17' and args.use_replay:
                    fname = osp.join(root, 'MSMT17_V1',fname)
                elif data_name == 'msmt17' and not args.use_replay:
                    fname = osp.join(root, 'MSMT17_V2',fname)
                else:
                    fname = osp.join(root, fname)
            if camid > max_camid:
                # cam id start from 0
                max_camid += 1
            dataset.train.append((fname, pid+dataset.num_train_pids, camid+dataset_camid+1, domain))

        dataset.camid += max_camid
        dataset.num_train_pids += idataset.num_train_pids

    dataset.pid_list = pid_list
    train_set = sorted(dataset.train)

    iters = int(len(train_set) / batch_size)
    num_classes = dataset.num_train_pids

    train_transformer = build_transforms(args, is_train=True)

    test_transformer = build_transforms(args, is_train=False)

    rmgs_flag = num_instances > 0
    if args.use_replay:
        if rmgs_flag:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
        else:
            sampler = None
    else:
        domain_id = kwargs.get("domain_id", 0)
        if domain_id == -1:
            sampler = None
        else:
            sampler = ClassUniformlySampler4Incremental(train_set, pid_list, class_position=1, k=4)

    train_loader = DataLoader(Preprocessor(train_set, root=None, transform=train_transformer),
                              batch_size=batch_size, num_workers=workers, sampler=sampler,
                              shuffle=not rmgs_flag, pin_memory=True, drop_last=True)
    train_IterLoader = IterLoader(train_loader, length=iters)



    init_loader = DataLoader(Preprocessor(train_set, root=None, transform=test_transformer),
                             batch_size=128, num_workers=workers, shuffle=False, pin_memory=True, drop_last=False)

    return dataset, num_classes, train_IterLoader, train_loader, None, init_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def load_lreid_data(args):
    # Create data loaders
    dataset_market, num_classes_market, train_IterLoader_market, train_loader_market, test_loader_market, init_loader_market = \
        get_data('market1501', args, domain_id=0)

    dataset_duke, num_classes_duke, train_IterLoader_duke, train_loader_duke, test_loader_duke, init_loader_duke = \
        get_data('dukemtmc', args, domain_id=1)

    dataset_cuhksysu, num_classes_cuhksysu, train_IterLoader_cuhksysu, train_loader_cuhksysu, test_loader_cuhksysu, init_loader_chuksysu = \
        get_data('cuhk-sysu', args, domain_id=2)

    dataset_msmt17, num_classes_msmt17, train_IterLoader_msmt17, train_loader_msmt17, test_loader_msmt17, init_loader_msmt17 = \
        get_data('msmt17', args, domain_id=3)

    # Data loaders for test only
    dataset_cuhk03, _, _, _, test_loader_cuhk03, _ = get_data('cuhk03', args)
    # dataset_cuhk01, _, _, _, test_loader_cuhk01, _ = get_data('cuhk01', args)
    # dataset_grid, _, _, _, test_loader_grid, _ = get_data('grid', args)
    # dataset_sense, _, _, _, test_loader_sense, _ = get_data('sense', args)
    dataset_VIPeR, _, _, _, test_loader_VIPeR, _ = get_data('viper', args)
    dataset_PRID, _, _, _, test_loader_PRID, _ = get_data('prid', args)
    dataset_grid, _, _, _, test_loader_grid, _ = get_data('grid', args)
    dataset_i_LIDS, _, _, _, test_loader_i_LIDS, _ = get_data('ilids', args)
    dataset_cuhk01, _, _, _, test_loader_cuhk01, _ = get_data('cuhk01', args)
    dataset_cuhk02, _, _, _, test_loader_cuhk02, _ = get_data('cuhk02', args)
    dataset_sense, _, _, _, test_loader_sense, _ = get_data('sense', args)

    seen_domain = {
        'market1501': {
            'milestones': [40, 70],
            'lr': args.lr,
            'domain_class_num': num_classes_market,
            # 'train_epoch': args.epochs,
            'train_epoch': args.epochs + 20,
            'train_Iterloader': train_IterLoader_market,
            'train_loader': train_loader_market,
            'test_loader': test_loader_market,
            'init_loader': init_loader_market,
            'dataset': dataset_market,
            'replay_dataset_name': 'market1501'
        },
        'duke': {
            'milestones': [30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_duke,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_duke,
            'train_loader': train_loader_duke,
            'test_loader': test_loader_duke,
            'init_loader': init_loader_duke,
            'dataset': dataset_duke,
            'replay_dataset_name': 'market1501'
        },
        'cuhk': {
            'milestones': [30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_cuhksysu,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_cuhksysu,
            'train_loader': train_loader_cuhksysu,
            'test_loader': test_loader_cuhksysu,
            'init_loader': init_loader_chuksysu,
            'dataset': dataset_cuhksysu,
            'replay_dataset_name': 'duke'
        },
        'msmt17': {
            'milestones': [20, 30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_msmt17,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_msmt17,
            'train_loader': train_loader_msmt17,
            'test_loader': test_loader_msmt17,
            'init_loader': init_loader_msmt17,
            'dataset': dataset_msmt17,
            'replay_dataset_name': 'duke'
        }

    }
    seen_list = ['market1501', 'duke', 'cuhk', 'msmt17']
    # unseen_domain = {
    #     "cuhk01": {
    #         'test_loader': test_loader_cuhk01,
    #         'dataset': dataset_cuhk01,
    #     },
    #     "cuhk03": {
    #         'test_loader': test_loader_cuhk03,
    #         'dataset': dataset_cuhk03,
    #     },
    #     "grid": {
    #         'test_loader': test_loader_grid,
    #         'dataset': dataset_grid,
    #     },
    #     "SenseReID": {
    #         'test_loader': test_loader_sense,
    #         'dataset': dataset_sense,
    #     }
    # }
    # unseen_list = ['cuhk01', 'cuhk03', 'grid', 'SenseReID']
    unseen_domain = {
        "viper": {
            'test_loader': test_loader_VIPeR,
            'dataset': dataset_VIPeR,
        },
        "prid2011": {
            'test_loader': test_loader_PRID,
            'dataset': dataset_PRID,
        },
        "grid": {
            'test_loader': test_loader_grid,
            'dataset': dataset_grid,
        },
        "ilids": {
            'test_loader': test_loader_i_LIDS,
            'dataset': dataset_i_LIDS,
        },
        "cuhk01": {
            'test_loader': test_loader_cuhk01,
            'dataset': dataset_cuhk01,
        },
        "cuhk02": {
            'test_loader': test_loader_cuhk02,
            'dataset': dataset_cuhk02,
        },
        "cuhk03": {
            'test_loader': test_loader_cuhk03,
            'dataset': dataset_cuhk03,
        },
        "SenseReID": {
            'test_loader': test_loader_sense,
            'dataset': dataset_sense,
        }
    }
    unseen_list = ['viper', 'prid2011', 'grid', 'ilids', 'cuhk01', 'cuhk02', 'cuhk03','SenseReID']
    return seen_domain, seen_list, unseen_domain, unseen_list

def load_lreid_no_replaydata_1(args):
    # Create data loaders
    dataset_market, num_classes_market, train_IterLoader_market, train_loader_market, test_loader_market, init_loader_market = \
        get_data('market1501_DF', args, domain_id=0)

    dataset_cuhksysu, num_classes_cuhksysu, train_IterLoader_cuhksysu, train_loader_cuhksysu, test_loader_cuhksysu, init_loader_chuksysu = \
        get_data('cuhk-sysu_DF', args, domain_id=1)

    dataset_duke, num_classes_duke, train_IterLoader_duke, train_loader_duke, test_loader_duke, init_loader_duke = \
        get_data('dukemtmc_DF', args, domain_id=2)

    dataset_msmt17, num_classes_msmt17, train_IterLoader_msmt17, train_loader_msmt17, test_loader_msmt17, init_loader_msmt17 = \
        get_data('msmt17_DF', args, domain_id=3, version="V2")


    dataset_cuhk03, num_classes_cuhk03, train_IterLoader_cuhk03, train_loader_cuhk03, test_loader_cuhk03, init_loader_cuhk03 = \
        get_data('cuhk03_DF', args, domain_id=4)


    dataset_VIPeR, _, _, _, test_loader_VIPeR, _ = get_data('viper', args)
    dataset_PRID, _, _, _, test_loader_PRID, _ = get_data('prid', args)
    dataset_grid, _, _, _, test_loader_grid, _ = get_data('grid', args)
    dataset_i_LIDS, _, _, _, test_loader_i_LIDS, _ = get_data('ilids', args)
    dataset_cuhk01, _, _, _, test_loader_cuhk01, _ = get_data('cuhk01', args)
    dataset_cuhk02, _, _, _, test_loader_cuhk02, _ = get_data('cuhk02', args)
    dataset_sense, _, _, _, test_loader_sense, _ = get_data('sense', args)

    seen_domain = {
        'market1501': {
            'milestones': [40, 70],
            'lr': args.lr,
            'domain_class_num': num_classes_market,
            # 'train_epoch': args.epochs,
            'train_epoch': args.epochs + 20,
            'train_Iterloader': train_IterLoader_market,
            'train_loader': train_loader_market,
            'test_loader': test_loader_market,
            'init_loader': init_loader_market,
            'dataset': dataset_market,
            'replay_dataset_name': 'market1501'
        },
        'cuhk': {
            'milestones': [30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_cuhksysu,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_cuhksysu,
            'train_loader': train_loader_cuhksysu,
            'test_loader': test_loader_cuhksysu,
            'init_loader': init_loader_chuksysu,
            'dataset': dataset_cuhksysu,
            'replay_dataset_name': 'duke'
        },
        'duke': {
            'milestones': [30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_duke,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_duke,
            'train_loader': train_loader_duke,
            'test_loader': test_loader_duke,
            'init_loader': init_loader_duke,
            'dataset': dataset_duke,
            'replay_dataset_name': 'market1501'
        },
        'msmt17': {
            'milestones': [20, 30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_msmt17,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_msmt17,
            'train_loader': train_loader_msmt17,
            'test_loader': test_loader_msmt17,
            'init_loader': init_loader_msmt17,
            'dataset': dataset_msmt17,
            'replay_dataset_name': 'duke'
        },
        'cuhk03': {
            'milestones': [20, 30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_cuhk03,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_cuhk03,
            'train_loader': train_loader_cuhk03,
            'test_loader': test_loader_cuhk03,
            'init_loader': init_loader_cuhk03,
            'dataset': dataset_cuhk03,
            'replay_dataset_name': 'msmt17'
        }

    }
    seen_list = ['market1501', 'cuhk', 'duke', 'msmt17', 'cuhk03']
    unseen_domain = {
        "viper": {
            'test_loader': test_loader_VIPeR,
            'dataset': dataset_VIPeR,
        },
        "prid2011": {
            'test_loader': test_loader_PRID,
            'dataset': dataset_PRID,
        },
        "grid": {
            'test_loader': test_loader_grid,
            'dataset': dataset_grid,
        },
        "ilids": {
            'test_loader': test_loader_i_LIDS,
            'dataset': dataset_i_LIDS,
        },
        "cuhk01": {
            'test_loader': test_loader_cuhk01,
            'dataset': dataset_cuhk01,
        },
        "cuhk02": {
            'test_loader': test_loader_cuhk02,
            'dataset': dataset_cuhk02,
        },
        "SenseReID": {
            'test_loader': test_loader_sense,
            'dataset': dataset_sense,
        }
    }
    unseen_list = ['viper', 'prid2011', 'grid', 'ilids', 'cuhk01', 'cuhk02','SenseReID']
    return seen_domain, seen_list, unseen_domain, unseen_list

def load_lreid_no_replaydata_2(args):
    # Create data loaders

    dataset_duke, num_classes_duke, train_IterLoader_duke, train_loader_duke, test_loader_duke, init_loader_duke = \
        get_data('dukemtmc_DF', args, domain_id=0)

    dataset_msmt17, num_classes_msmt17, train_IterLoader_msmt17, train_loader_msmt17, test_loader_msmt17, init_loader_msmt17 = \
        get_data('msmt17_DF', args, domain_id=1, version="V2")

    dataset_market, num_classes_market, train_IterLoader_market, train_loader_market, test_loader_market, init_loader_market = \
        get_data('market1501_DF', args, domain_id=2)

    dataset_cuhksysu, num_classes_cuhksysu, train_IterLoader_cuhksysu, train_loader_cuhksysu, test_loader_cuhksysu, init_loader_chuksysu = \
        get_data('cuhk-sysu_DF', args, domain_id=3)
    # Data loaders for test only
    dataset_cuhk03, num_classes_cuhk03, train_IterLoader_cuhk03, train_loader_cuhk03, test_loader_cuhk03, init_loader_cuhk03 = \
        get_data('cuhk03_DF', args, domain_id=4)

    dataset_VIPeR, _, _, _, test_loader_VIPeR, _ = get_data('viper', args)
    dataset_PRID, _, _, _, test_loader_PRID, _ = get_data('prid', args)
    dataset_grid, _, _, _, test_loader_grid, _ = get_data('grid', args)
    dataset_i_LIDS, _, _, _, test_loader_i_LIDS, _ = get_data('ilids', args)
    dataset_cuhk01, _, _, _, test_loader_cuhk01, _ = get_data('cuhk01', args)
    dataset_cuhk02, _, _, _, test_loader_cuhk02, _ = get_data('cuhk02', args)
    dataset_sense, _, _, _, test_loader_sense, _ = get_data('sense', args)

    seen_domain = {

        'duke': {
            'milestones': [30, 55, 70],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_duke,
            'train_epoch': args.epochs+20,
            'train_Iterloader': train_IterLoader_duke,
            'train_loader': train_loader_duke,
            'test_loader': test_loader_duke,
            'init_loader': init_loader_duke,
            'dataset': dataset_duke,
            'replay_dataset_name': 'market1501'
        },
        'msmt17': {
            'milestones': [20, 30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_msmt17,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_msmt17,
            'train_loader': train_loader_msmt17,
            'test_loader': test_loader_msmt17,
            'init_loader': init_loader_msmt17,
            'dataset': dataset_msmt17,
            'replay_dataset_name': 'duke'
        },
        'market1501': {
            'milestones': [40, 70],
            'lr': args.lr,
            'domain_class_num': num_classes_market,
            # 'train_epoch': args.epochs,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_market,
            'train_loader': train_loader_market,
            'test_loader': test_loader_market,
            'init_loader': init_loader_market,
            'dataset': dataset_market,
            'replay_dataset_name': 'market1501'
        },
        'cuhk': {
            'milestones': [30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_cuhksysu,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_cuhksysu,
            'train_loader': train_loader_cuhksysu,
            'test_loader': test_loader_cuhksysu,
            'init_loader': init_loader_chuksysu,
            'dataset': dataset_cuhksysu,
            'replay_dataset_name': 'duke'
        },
        'cuhk03': {
            'milestones': [20, 30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_cuhk03,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_cuhk03,
            'train_loader': train_loader_cuhk03,
            'test_loader': test_loader_cuhk03,
            'init_loader': init_loader_cuhk03,
            'dataset': dataset_cuhk03,
            'replay_dataset_name': 'msmt17'
        }

    }
    seen_list = ['duke', 'msmt17', 'market1501', 'cuhk', 'cuhk03']
    unseen_domain = {
        "viper": {
            'test_loader': test_loader_VIPeR,
            'dataset': dataset_VIPeR,
        },
        "prid2011": {
            'test_loader': test_loader_PRID,
            'dataset': dataset_PRID,
        },
        "grid": {
            'test_loader': test_loader_grid,
            'dataset': dataset_grid,
        },
        "ilids": {
            'test_loader': test_loader_i_LIDS,
            'dataset': dataset_i_LIDS,
        },
        "cuhk01": {
            'test_loader': test_loader_cuhk01,
            'dataset': dataset_cuhk01,
        },
        "cuhk02": {
            'test_loader': test_loader_cuhk02,
            'dataset': dataset_cuhk02,
        },
        "SenseReID": {
            'test_loader': test_loader_sense,
            'dataset': dataset_sense,
        }
    }
    unseen_list = ['viper', 'prid2011', 'grid', 'ilids', 'cuhk01', 'cuhk02','SenseReID']
    return seen_domain, seen_list, unseen_domain, unseen_list

def load_lreid_no_replaydata_3(args):
    # Create data loaders

    dataset_duke, num_classes_duke, train_IterLoader_duke, train_loader_duke, test_loader_duke, init_loader_duke = \
        get_data('dukemtmc_DF', args, domain_id=0)

    dataset_msmt17, num_classes_msmt17, train_IterLoader_msmt17, train_loader_msmt17, test_loader_msmt17, init_loader_msmt17 = \
        get_data('msmt17_DF', args, domain_id=1, version="V2")

    dataset_market, num_classes_market, train_IterLoader_market, train_loader_market, test_loader_market, init_loader_market = \
        get_data('market1501_DF', args, domain_id=2)

    dataset_cuhksysu, num_classes_cuhksysu, train_IterLoader_cuhksysu, train_loader_cuhksysu, test_loader_cuhksysu, init_loader_chuksysu = \
        get_data('cuhk-sysu_DF', args, domain_id=3)
    # Data loaders for test only
    dataset_cuhk03, num_classes_cuhk03, train_IterLoader_cuhk03, train_loader_cuhk03, test_loader_cuhk03, init_loader_cuhk03 = \
        get_data('cuhk03_DF', args, domain_id=4)

    dataset_VIPeR, _, _, _, test_loader_VIPeR, _ = get_data('viper', args)
    dataset_PRID, _, _, _, test_loader_PRID, _ = get_data('prid', args)
    dataset_grid, _, _, _, test_loader_grid, _ = get_data('grid', args)
    dataset_i_LIDS, _, _, _, test_loader_i_LIDS, _ = get_data('ilids', args)
    dataset_cuhk01, _, _, _, test_loader_cuhk01, _ = get_data('cuhk01', args)
    dataset_cuhk02, _, _, _, test_loader_cuhk02, _ = get_data('cuhk02', args)
    dataset_sense, _, _, _, test_loader_sense, _ = get_data('sense', args)

    seen_domain = {
        'market1501': {
            'milestones': [40, 70],
            'lr': args.lr,
            'domain_class_num': num_classes_market,
            # 'train_epoch': args.epochs,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_market,
            'train_loader': train_loader_market,
            'test_loader': test_loader_market,
            'init_loader': init_loader_market,
            'dataset': dataset_market,
            'replay_dataset_name': 'market1501'
        },
        'duke': {
            'milestones': [30, 55, 70],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_duke,
            'train_epoch': args.epochs+20,
            'train_Iterloader': train_IterLoader_duke,
            'train_loader': train_loader_duke,
            'test_loader': test_loader_duke,
            'init_loader': init_loader_duke,
            'dataset': dataset_duke,
            'replay_dataset_name': 'cuhk'
        },
        'cuhk03': {
            'milestones': [20, 30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_cuhk03,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_cuhk03,
            'train_loader': train_loader_cuhk03,
            'test_loader': test_loader_cuhk03,
            'init_loader': init_loader_cuhk03,
            'dataset': dataset_cuhk03,
            'replay_dataset_name': 'msmt17'
        },
        'cuhk': {
            'milestones': [30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_cuhksysu,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_cuhksysu,
            'train_loader': train_loader_cuhksysu,
            'test_loader': test_loader_cuhksysu,
            'init_loader': init_loader_chuksysu,
            'dataset': dataset_cuhksysu,
            'replay_dataset_name': 'market1501'
        },

        'msmt17': {
            'milestones': [20, 30],
            'lr': args.lr * 0.1,
            'domain_class_num': num_classes_msmt17,
            'train_epoch': args.epochs,
            'train_Iterloader': train_IterLoader_msmt17,
            'train_loader': train_loader_msmt17,
            'test_loader': test_loader_msmt17,
            'init_loader': init_loader_msmt17,
            'dataset': dataset_msmt17,
            'replay_dataset_name': 'duke'
        },




    }
    print("33333333333333333333333333333333333333333333333333333333")
    seen_list = ['market1501', 'duke', 'cuhk03', 'cuhk', 'msmt17']

    unseen_domain = {
        "viper": {
            'test_loader': test_loader_VIPeR,
            'dataset': dataset_VIPeR,
        },
        "prid2011": {
            'test_loader': test_loader_PRID,
            'dataset': dataset_PRID,
        },
        "grid": {
            'test_loader': test_loader_grid,
            'dataset': dataset_grid,
        },
        "ilids": {
            'test_loader': test_loader_i_LIDS,
            'dataset': dataset_i_LIDS,
        },
        "cuhk01": {
            'test_loader': test_loader_cuhk01,
            'dataset': dataset_cuhk01,
        },
        "cuhk02": {
            'test_loader': test_loader_cuhk02,
            'dataset': dataset_cuhk02,
        },
        "SenseReID": {
            'test_loader': test_loader_sense,
            'dataset': dataset_sense,
        }
    }
    unseen_list = ['viper', 'prid2011', 'grid', 'ilids', 'cuhk01', 'cuhk02','SenseReID']
    return seen_domain, seen_list, unseen_domain, unseen_list


class IncrementalReIDDataSet:
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):

        this_sample = copy.deepcopy(self.samples[index])
        this_sample = list(this_sample)
        this_sample.append(this_sample[0])
        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])

        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')

def _get_all_loader(size_test, samples, batch_size):
    import torchvision
    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size_test, interpolation=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = IncrementalReIDDataSet(samples, transform=transform_test)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
    return loader


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

            new_list[i][pid_dimension] = pid2relabel_dict[
                str(sample[pid_dimension])+str(sample[pid_dimension + 2])
                ]
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

def _get_all_test_samples(args):
    from .datasets.sensereid import IncrementalSamples4sensereid
    from .datasets.cuhk01 import IncrementalSamples4cuhk01
    from .datasets.cuhk02 import IncrementalSamples4cuhk02
    from .datasets.viper import IncrementalSamples4viper
    from .datasets.ilids import IncrementalSamples4ilids
    from .datasets.prid import IncrementalSamples4prid
    from .datasets.grid import IncrementalSamples4grid
    samples4sensereid = IncrementalSamples4sensereid(args.data_dir, relabel=True,
                                                     combineall=False)

    samples4cuhk01 = IncrementalSamples4cuhk01(args.data_dir, relabel=True,
                                               combineall=False)

    samples4cuhk02 = IncrementalSamples4cuhk02(args.data_dir, relabel=True,
                                               combineall=False)

    samples4viper = IncrementalSamples4viper(args.data_dir, relabel=True,
                                             combineall=False)

    samples4ilids = IncrementalSamples4ilids(args.data_dir, relabel=True,
                                             combineall=False)

    samples4prid = IncrementalSamples4prid(args.data_dir, relabel=True,
                                           combineall=False)

    samples4grid = IncrementalSamples4grid(args.data_dir, relabel=True,
                                           combineall=False)
    query, gallery = Incremental_combine_test_samples(
        samples_list=[
            samples4viper, samples4ilids,
            samples4prid,
            samples4grid,
            samples4sensereid, samples4cuhk01,
            samples4cuhk02
        ])
    return query, gallery