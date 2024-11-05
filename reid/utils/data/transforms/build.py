# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""


from reid.utils.data.transforms import transforms as T
from .autoaugment import AutoAugment


def build_transforms(args, is_train=True):
    res = []
    data_aug_strategy = args.data_aug
    if is_train:
        size_train = args.height, args.width
        total_iter = args.iters

        if 'DO_AUTOAUG' in data_aug_strategy:
            res.append(AutoAugment(total_iter))

        # resize
        res.append(T.Resize(size_train, interpolation=3))
        # RandomHorizontalFlip
        if 'DO_FLIP' in data_aug_strategy:
            flip_prob = args.FLIP_PROB
            res.append(T.RandomHorizontalFlip(p=flip_prob))
        # Pad
        if 'DO_PAD' in data_aug_strategy:
            padding = args.PADDING
            padding_mode = args.PADDING_MODE
            res.extend([T.Pad(padding, padding_mode=padding_mode),
                        T.RandomCrop(size_train)
                        ])
        # if 'CJ_ENABLED' in data_aug_strategy:
        #     cj_prob = args.CJ_PROB
        #     cj_brightness = args.CJ_BRIGHTNESS
        #     cj_contrast = args.CJ_CONTRAST
        #     cj_saturation = args.CJ_SATURATION
        #     cj_hue = args.CJ_HUE
        #     res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))

        res.append(T.ToTensor())
        if 'DO_NORM' in data_aug_strategy:
            norm_mean = args.NORM_MEAN
            norm_std = args.NORM_STD
            res.append(T.Normalize(mean=norm_mean, std=norm_std))
        if 'REA_ENABLED' in data_aug_strategy:
            rea_prob = args.REA_PROB
            rea_mean = args.REA_MEAN
            res.append(T.RandomErasing(probability=rea_prob, mean=rea_mean))
    else:
        norm_mean = args.NORM_MEAN
        norm_std = args.NORM_STD

        size_test = args.height, args.width
        res.append(T.Resize(size_test, interpolation=3))
        res.append(T.ToTensor())
        res.append(T.Normalize(mean=norm_mean, std=norm_std))


    return T.Compose(res)
