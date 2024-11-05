'''
this script remove the all aux loss function, and add the interpratable head
'''


from __future__ import print_function, absolute_import
import os.path as osp
import sys
import os
from sklearn.preprocessing import normalize


from torch.backends import cudnn
import copy
import torch.nn as nn
import random
import datetime
from parameters import get_hyper_para, setup

from reid.evaluators import Evaluator, evaluate_datasets, format_evalute_info, extract_extra_features

from reid.utils.logging import Logger
from reid.visualization import visualization_assignment, visualize_tsne
from reid.utils.serialization import load_checkpoint, save_checkpoint, load_ckpt
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.my_tools import *
from reid.load_lreid_data import load_lreid_data, load_lreid_no_replaydata_1, load_lreid_no_replaydata_2, load_lreid_no_replaydata_3
from reid.models.layers import DataParallel

# model config
from reid.models.resnet_part_TSBN import build_resnet_coseg_backbone as TSBNPartBackbone
from reid.trainer_part import TrainerPart



def update_replay_info(replay_domain, replay_dataloader, replay_dataset, replay_datasetname):
    replay_domain[replay_datasetname] = {
        'replay_dataloader': replay_dataloader,
        'replay_dataset': replay_dataset
    }
    return replay_domain


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

    if args.evaluate:
        log_name = f'{args.logfilename}_evaluate_unseen_log.txt'
    else:
        log_name = f'{time_str}_{args.logfilename}_train_log.txt'
    sys.stdout = Logger(osp.join(args.logs_dir, log_name))
    print("==========\nArgs:{}\n==========".format(args))

    check_config(args)
    main_worker(args)


def main_worker(args):
    cudnn.benchmark = True

    if args.visualize_train_by_visdom:
        port = args.port
        visdom_dict = {}

    # load reid data
    if args.use_replay:
        seen_domain, seen_list, unseen_domain, unseen_list = load_lreid_data(args)
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
            if args.header == 'part':
                model = TSBNPartBackbone(
                    num_class=seen_domain[seen_list[0]]['domain_class_num'],
                    depth='50x', part_dim=args.part_dim, num_parts=args.num_parts, args=args
                )
            model.cuda()
            model = DataParallel(model)
        else:
            raise "Unknown header"

    else:
        raise "Unknown backbone"

    global_start_epoch = 0
    domain_num_class = 0
    old_model = None
    replay_dataset = None

    replay_domain = {_: {
        'replay_dataloader': None,
        'replay_dataset': None
    } for _ in seen_list}

    # Evaluator
    evaluator = Evaluator(model, args)

    # Start training
    print('Continual training starts!')

    for idx, domain_name in enumerate(seen_domain):
        training_phase = idx + 1

        print(f"Train stage {training_phase}: {domain_name}")

        domain_num_class += seen_domain[domain_name]['domain_class_num']
        old_num_class = domain_num_class - seen_domain[domain_name]['domain_class_num']

        resume_file = osp.join(args.logs_dir, domain_name + "_checkpoint.pth.tar")

        if args.evaluate:
            if os.path.isfile(resume_file):
                print(f"loading checkpoint file from {resume_file}")
                checkpoint = load_checkpoint(resume_file)
                model = load_ckpt(checkpoint['state_dict'], model)
                start_epoch = checkpoint['epoch']
                best_mAP = checkpoint['mAP']
                print("=> Start {}: epoch {} mAP {:.1%}".format(domain_name, start_epoch, best_mAP))

                if args.tsne:
                    print('=================t-SNE on Seen data:=================')
                    visualize_tsne(seen_domain, seen_list, model, domain_name, args)
                else:
                    print('=================Testing on seen tasks:=================')
                    val_seen_result = evaluate_datasets(
                        evaluator, seen_domain,
                        evaluate_name_list=seen_list
                    )
                    print(val_seen_result)

                    print('=================Testing on unseen tasks:=================')
                    test_unseen_dict = evaluate_datasets(evaluator, unseen_domain, evaluate_name_list=unseen_list)
                    test_unseen_result = format_evalute_info(test_unseen_dict)
                    print(test_unseen_result)


            else:
                print("{} not exist, exit...".format(resume_file))
                continue
        else:
            if args.resume and os.path.exists(resume_file):
                print(f"loading checkpoint file from {resume_file}")
                checkpoint = load_checkpoint(resume_file)
                model = load_ckpt(checkpoint['state_dict'], model)
                start_epoch = checkpoint['epoch']
                best_mAP = checkpoint['mAP']
                print("=> Start {}: epoch {} mAP {:.1%}".format(domain_name, start_epoch, best_mAP))

            else:
                params = []
                if args.use_TSBN:
                    for key, value in model.named_params(model):
                        if not value.requires_grad:
                            continue
                        if "attention" in key:
                            if args.grouping_arch == "MatchGrouping":
                                lr = seen_domain[domain_name]['lr'] * 0.1
                            elif args.grouping_arch == "MeanGrouping":
                                lr = seen_domain[domain_name]['lr'] * 0.1
                        else:
                            lr = seen_domain[domain_name]['lr']
                        params += [
                            {
                                "name": key,
                                "params": [value],
                                "lr": lr,
                                "weight_decay": args.weight_decay
                            }
                        ]
                else:
                    for key, value in model.named_params(model):
                    # for key, value in model.named_parameters():
                        if not value.requires_grad:
                            continue
                        params += [{
                            "name": key, "params": [value],
                            "lr": seen_domain[domain_name]['lr'],
                            "weight_decay": args.weight_decay
                        }]
                ####################### trainer initialization #######################
                start_epoch = 0
                optimizer = torch.optim.Adam(params)
                lr_scheduler = WarmupMultiStepLR(
                    optimizer, seen_domain[domain_name]['milestones'], gamma=0.1,
                    warmup_factor=0.01, warmup_iters=args.warmup_step
                )

                # if args.header == 'part':

                trainer = TrainerPart(
                    model, domain_num_class, margin=args.margin,
                    use_TSBN=args.use_TSBN, loss_items=args.loss_items,
                    header=args.header, args=args
                )

                for epoch in range(start_epoch, seen_domain[domain_name]['train_epoch']):
                    print(f"starting train stage: {training_phase} with {epoch} epoches")
                    seen_domain[domain_name]['train_Iterloader'].new_epoch()

                    use_domain_name = seen_domain[domain_name]['replay_dataset_name']
                    train_iters = len(seen_domain[domain_name]['train_Iterloader'])

                    loss_info = trainer.train(
                        epoch, seen_domain[domain_name]['train_Iterloader'],
                        replay_domain[use_domain_name]['replay_dataloader'],
                        optimizer, training_phase=training_phase,
                        train_iters=train_iters,
                        add_num=old_num_class, old_model=old_model,
                        km_obj=None,
                    )
                    lr_scheduler.step()

                    if (epoch + 1 == seen_domain[domain_name]['train_epoch']):
                        try:
                            val_seen_result = evaluate_datasets(evaluator, seen_domain,
                                                                evaluate_name_list=reversed(seen_list[:training_phase]))
                        except:
                            val_seen_result = {
                                domain_name: {
                                    'mAP': -1
                                }
                            }
                        save_checkpoint({
                            'state_dict': model.state_dict(),
                            'epoch': epoch + 1,
                            'mAP': val_seen_result[domain_name]['mAP'],
                        }, True, fpath=osp.join(args.logs_dir, f'{domain_name}_checkpoint.pth.tar'))

                        print('Finished epoch {:3d} {} mAP: {:5.1%} '.format(epoch, domain_name,
                                                                             val_seen_result[domain_name]['mAP']))

            # data and model replay setting
            if args.use_replay:
                replay_dataloader, replay_dataset = select_replay_samples(
                    model, seen_domain[domain_name]['dataset'], training_phase=training_phase,
                    add_num=old_num_class, old_datas=replay_dataset, batch_size=args.replay_batch_size,
                    num_workers=args.workers, header=args.header
                )

                replay_domain = update_replay_info(
                    replay_domain,
                    replay_dataloader=replay_dataloader,
                    replay_dataset=replay_dataset,
                    replay_datasetname=domain_name
                )


            if training_phase < len(seen_domain):
                next_domain_name = seen_list[idx + 1]
                next_domain_class = seen_domain[next_domain_name]['domain_class_num']
                # Expand the dimension of classifier
                org_classifier_params = model.module.classifier.weight.data
                if args.header in part_header_list:
                    org_classifier_part_params = model.module.classifier_part.weight.data

                model.module.classifier = nn.Linear(2048, domain_num_class + next_domain_class, bias=False)
                model.module.classifier.weight.data[:domain_num_class].copy_(org_classifier_params)
                if args.header in part_header_list:
                    model.module.classifier_part = nn.Linear(args.part_dim * args.num_parts,
                                                                   domain_num_class + next_domain_class, bias=False)
                    model.module.classifier_part.weight.data[:domain_num_class].copy_(org_classifier_part_params)
                model.cuda()

                # Initialize classifer with class centers
                init_weights = initial_classifier(
                    model, seen_domain[next_domain_name]['init_loader'],
                    method=args.init_new_header, header=args.header
                )
                model.module.classifier.weight.data[-init_weights[0].size(0):, :].copy_(init_weights[0])
                if args.header in part_header_list:
                    model.module.classifier_part.weight.data[-init_weights[1].size(0):, :].copy_(init_weights[1])
                model.cuda()

                # Create old frozen model
                old_model = copy.deepcopy(model)
                old_model = old_model.cuda()
                old_model.eval()

    print('finished')


if __name__ == '__main__':
    command_line_args = get_hyper_para()
    print("Command Line Args:", command_line_args)
    args = setup(command_line_args)
    main()

