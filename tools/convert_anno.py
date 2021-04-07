# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import argparse

import nncore

from consnet.api import convert_anno


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--anno',
        help='annotation file',
        default='data/hico_20160224_det/anno_bbox.mat')
    parser.add_argument(
        '--out', help='output directory', default='data/hico_det/annotations')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    for split in ['train', 'test']:
        out_file = nncore.join(args.out, f'hico_det_{split}.json')
        convert_anno(args.anno, out_file, split)


if __name__ == '__main__':
    main()
