# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

from nncore import Registry, build_object

DATASETS = Registry('dataset')


def build_dataset(cfg, **kwargs):
    return build_object(cfg, DATASETS, **kwargs)
