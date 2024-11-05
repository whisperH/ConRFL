######################################

#
#
#
#                            _ooOoo_
#                           o8888888o
#                          o88.888.88o
#                           (| -_- |)
#                            O\ = /O
#                        ____/`---'\____
#                      .   ' \| |// `.
#                       / \||| : |||// \
#  //                     / _||||| -:- |||||- \
#                                    //                       | | \\ - /// | |
#                     | \_| ''\---/'' | |
#                      \ .-\__ `-` ___/-. /
#                   ___`. .' /--.--\ `. . __
#                . '< `.___\_<|>_/___.' >'.
#               | | : `- \`.;`\ _ /`;.`/ - ` : | |
#                 \ \ `-. \_ __\ /__ _/ .-` / /
#         ======`-.____`-.___\_____/___.-`____.-'======
#                            `=---='
#
#         .............................................
#                  佛祖镇楼             BUG辟易
#
#          佛曰:
#                  写字楼里写字间，写字间里程序员；
#                  程序人员写程序，又拿程序换酒钱。
#                  酒醒只在网上坐，酒醉还来网下眠；
#                  酒醉酒醒日复日，网上网下年复年。
#                  但愿老死电脑间，不愿鞠躬老板前；
#                  奔驰宝马贵者趣，公交自行程序员。
#                  别人笑我忒疯癫，我笑自己命太贱；
#                  不见满街漂亮妹，哪个归得程序员？
#
#
#
######################################
'''
this script remove the all aux loss function, and add the interpratable head
'''


from __future__ import print_function, absolute_import
import os.path as osp
import sys
import os

from torch.backends import cudnn
import random
import datetime
from parameters import get_hyper_para, setup

from reid.evaluators import Evaluator, evaluate_datasets, evalute_all_generalization

from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, load_ckpt
from reid.utils.my_tools import *
from reid.load_lreid_data import load_lreid_data,\
    load_lreid_no_replaydata_1, load_lreid_no_replaydata_2, load_lreid_no_replaydata_3, _get_all_loader, _get_all_test_samples
from reid.models.layers import DataParallel

# model config
from reid.models.TMT_forgetpart_TSBN import build_TMT_coseg_backbone as TSBNFGTMTPartBackbone





def check_config(args):
    loss_items = args.loss_items
    backbone = args.backbone
    header = args.header
    use_replay = args.use_replay

    not_support_loss = []
    if not args.use_TSBN:
        not_support_loss = ['PT_ID', 'PT_KD', 'DCL']
    if header not in part_header_list:
        not_support_loss = ['nonlap'] + [_ for _ in loss_items if 'part' in _]
    if not use_replay:
        not_support_loss = ['replay_tr', 'kd_r', 'PT_KD']

    res = list(set(not_support_loss) & set(loss_items))
    assert len(res) == 0, f"not_support_loss: {not_support_loss} in {backbone} with header: {header}"

    print(f"using backbone: {backbone} header: {header} use_replay: {use_replay}")
    print(f"loss items: {loss_items}")



def main():
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True


    log_name = f'{args.logfilename}_evaluate_all_generalization_log.txt'

    sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    check_config(args)
    main_worker(args)




def main_worker(args):
    cudnn.benchmark = True
    size_test = args.height, args.width
    all_query_samples, all_gallery_samples = _get_all_test_samples(args)
    all_query_loader = _get_all_loader(size_test, all_query_samples, args.batch_size)
    all_gallery_loader = _get_all_loader(size_test, all_gallery_samples, args.batch_size)

    # load reid data
    if args.use_replay:
        seen_domain, seen_list, unseen_domain, unseen_list = load_lreid_data(args)
        if args.header == 'ncforgetpart':
            seen_domain['msmt17']['replay_dataset_name'] = 'cuhk'
    else:
        if args.order == "list1":
            print('training order list 1')
            seen_domain, seen_list, unseen_domain, unseen_list = load_lreid_no_replaydata_1(args)
        elif args.order == "list2":
            print('training order list 2')
            seen_domain, seen_list, unseen_domain, unseen_list = load_lreid_no_replaydata_2(args)
        else:
            print('training order list 3')
            seen_domain, seen_list, unseen_domain, unseen_list = load_lreid_no_replaydata_3(args)
    # Create model
    if args.backbone == 'ResNet':
        if args.use_TSBN:
            # if args.header == 'part':
            #     model = TSBNTMTPartBackbone(
            #         num_class=seen_domain[seen_list[0]]['domain_class_num'],
            #         depth='50x', part_dim=args.part_dim, num_parts=args.num_parts, args=args
            #     )
            # el
            if args.header == 'ncforgetpart':
                model = TSBNFGTMTPartBackbone(
                    num_class=seen_domain[seen_list[0]]['domain_class_num'],
                    depth='50x', part_dim=args.part_dim, num_parts=args.num_parts, args=args
                )
            # elif args.header == 'forgetpart':
            #     model = TSBNFGTMTPartBackbone(
            #         num_class=seen_domain[seen_list[0]]['domain_class_num'],
            #         depth='50x', part_dim=args.part_dim, num_parts=args.num_parts, args=args
            #     )
            # elif args.header in ["midpart", "ncforgetmidpart"]:
            #     model = TSBNMidPartBackbone(
            #         num_class=seen_domain[seen_list[0]]['domain_class_num'],
            #         depth='50x', part_dim=args.part_dim, num_parts=args.num_parts, args=args
            #     )

            model.cuda()
            model = DataParallel(model)
        else:
            if args.header == 'ncforgetpart':
                model = TSBNFGTMTPartBackbone(
                    num_class=seen_domain[seen_list[0]]['domain_class_num'],
                    depth='50x', part_dim=args.part_dim, num_parts=args.num_parts, args=args
                )
            model.cuda()
            model = DataParallel(model)
    else:
        raise "Unknown backbone"

    # Evaluator
    evaluator = Evaluator(model, args)

    # Start training
    print('Continual training starts!')

    for idx, domain_name in enumerate(seen_domain):
        training_phase = idx + 1

        print(f"Train stage {training_phase}: {domain_name}")

        resume_file = osp.join(args.logs_dir, domain_name + "_checkpoint.pth.tar")

        if os.path.isfile(resume_file):
            print(f"loading checkpoint file from {resume_file}")
            checkpoint = load_checkpoint(resume_file)
            model = load_ckpt(checkpoint['state_dict'], model)
            # start_epoch = checkpoint['epoch']
            # best_mAP = checkpoint['mAP']
            # print("=> Start {}: epoch {} mAP {:.1%}".format(domain_name, start_epoch, best_mAP))

            print('=================Testing on seen tasks:=================')
            # val_seen_result = evaluate_datasets(
            #     evaluator, seen_domain,
            #     evaluate_name_list=seen_list
            # )
            # print(val_seen_result)

            print('=================Testing on allgeneralizable:=================')
            val_all_unseen_result = evalute_all_generalization(
                evaluator,
                all_query_loader, all_gallery_loader
            )
            print(val_all_unseen_result)
        else:
            print("{} not exist, exit...".format(resume_file))
            continue


    print('finished')


if __name__ == '__main__':
    command_line_args = get_hyper_para()
    print("Command Line Args:", command_line_args)
    args = setup(command_line_args)
    if args.visualize_train_by_visdom:
        from reid.utils.logging import VisdomPlotLogger, VisdomFeatureMapsLogger, plot_dict


    # args.logs_dir = args.logs_dir.replace("/home/huangjinze/code", "/root/autodl-tmp")
    # args.data_dir = "/root/autodl-nas/REID"
    # args.cache_file = "/root/autodl-tmp/PTKP/pretrain/resnet50-19c8e357.pth"
    main()

