import argparse
import os.path as osp
from reid.config import get_cfg
from reid.utils.file_io import PathManager

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # ================================================ #
    eval_mode = args.evaluate
    debug_mode = args.debug
    tsne = args.tsne
    # ================================================ #
    args = cfg.merge_from_file(args.config_file)
    # ================================================ #
    # evaluate reset by command line
    args.debug = debug_mode
    args.evaluate = eval_mode
    args.tsne = tsne
    # ================================================ #
    output_dir = args.logs_dir
    if output_dir:
        PathManager.mkdirs(output_dir)

    path = osp.join(output_dir, "config.yaml")
    with PathManager.open(path, "w") as f:
        f.write(args.dump())
    print("Full config saved to {}".format(osp.abspath(path)))
    return args

def get_hyper_para():
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")

    parser.add_argument('--tsne', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--evaluate', action='store_true')

    # ========================== path ==========================#

    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('-c', '--config_file', type=str, default='configs/Reset_Loss.yml')


    args = parser.parse_args()
    args.config_file = osp.join(working_dir, args.config_file)
    return args