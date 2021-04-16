# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import argparse

import nncore
from nncore.engine import Engine, set_random_seed

from consnet.models import build_dataloader, build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        help='config file',
        default='configs/consnet_5e_hico_det.py')
    parser.add_argument('--checkpoint', help='load a checkpoint')
    parser.add_argument('--resume', help='resume from a checkpoint')
    parser.add_argument('--eval', action='store_true', help='evaluation mode')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = nncore.Config.from_file(args.config)

    if not args.eval:
        timestamp = nncore.get_timestamp()
        work_dir = nncore.join('work_dirs', nncore.pure_name(args.config))
        log_file = nncore.join(work_dir, '{}.log'.format(timestamp))
    else:
        log_file = work_dir = None

    logger = nncore.get_logger(log_file=log_file)
    logger.info(f'Environment Info:\n{nncore.collect_env_info()}')
    logger.info(f'Config:\n{cfg.text}')

    seed = set_random_seed()
    logger.info(f'Using random seed: {seed}')

    model = build_model(cfg.model)
    logger.info(f'Model Architecture:\n{model.module}')

    data_loaders = {
        k: build_dataloader(v, seed=seed)
        for k, v in cfg.data.items()
    }

    engine = Engine(model, data_loaders, work_dir=work_dir)

    if checkpoint := args.checkpoint:
        engine.load_checkpoint(checkpoint)
    elif checkpoint := args.resume:
        engine.resume(checkpoint)

    engine.launch(eval_mode=args.eval)


if __name__ == '__main__':
    main()
