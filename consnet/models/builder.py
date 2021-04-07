# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

from nncore import Registry, build_object
from nncore.parallel import NNDataParallel, collate
from torch.utils.data import DataLoader

from consnet.datasets import build_dataset

try:
    from mmcv.parallel import MMDataParallel
    from mmdet.datasets import DATASETS
    from mmdet.datasets import build_dataloader as _build_dataloader
    from mmdet.datasets import build_dataset as _build_dataset
    from nncore.engine import load_checkpoint, set_random_seed
except ImportError:
    pass

DETECTORS = Registry('detector')
HEADS = Registry('head')
EMBEDDERS = Registry('embedder')
MODELS = Registry('model')
BLOCKS = Registry('block')
LOSSES = Registry('loss')


def build_dataloader(cfg, num_samples=1, num_workers=0, seed=None, **kwargs):
    if cfg['type'] in DATASETS:
        dataset = _build_dataset(cfg, **kwargs)
        data_loader = _build_dataloader(
            dataset, num_samples, num_workers, dist=False, shuffle=False)
    else:

        def _init_fn(worker_id):
            set_random_seed(seed=seed + worker_id)

        _cfg = cfg.pop('loader')
        dataset = build_dataset(cfg, **kwargs)
        data_loader = DataLoader(
            dataset, collate_fn=collate, worker_init_fn=_init_fn, **_cfg)

    return data_loader


def build_detector(cfg, checkpoint, **kwargs):
    model = build_object(cfg, DETECTORS, **kwargs)
    load_checkpoint(model, checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return model


def build_model(cfg, **kwargs):
    model = build_object(cfg, MODELS, **kwargs)
    model = NNDataParallel(model, device_ids=[0])
    return model


def build_head(cfg, **kwargs):
    return build_object(cfg, HEADS, **kwargs)


def build_embedder(cfg, **kwargs):
    return build_object(cfg, EMBEDDERS, **kwargs)


def build_block(cfg, **kwargs):
    return build_object(cfg, BLOCKS, **kwargs)


def build_loss(cfg, **kwargs):
    return build_object(cfg, LOSSES, **kwargs)
