# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import nncore
import torch
from scipy.io import loadmat

from .static import (ACT_IDX_TO_ACT_NAME, HOI_IDX_TO_ACT_IDX,
                     HOI_IDX_TO_OBJ_IDX, OBJ_IDX_TO_COCO_ID,
                     OBJ_IDX_TO_OBJ_NAME, RARE_HOI_IDX, UA_HOI_IDX, UC_HOI_IDX,
                     UO_HOI_IDX)


def convert_anno(anno_file, out_file, split):
    """
    Convert annotations of HICO-DET [1] dataset to COCO [2] format.

    Args:
        anno_file (str): Path to the ``anno_bbox.mat`` file.
        out_file (str): Path to the output file. The filename must end with
            ``'.json'``.
        split (str): The dataset split to be converted. Expected values
            are ``'train'`` and ``'test'``.

    References:
        1. Chao et al. (https://arxiv.org/abs/1702.05448)
        2. Lin et al. (https://arxiv.org/abs/1405.0312)
    """
    assert isinstance(out_file, str) and out_file.endswith('.json')
    assert split in ('train', 'test')
    nncore.is_file(anno_file, raise_error=True)

    print(f'converting annotations of *{split}* split')

    mat_anno = loadmat(anno_file)[f'bbox_{split}'][0]
    coco_anno, img_id, anno_id = dict(
        images=[], annotations=[], categories=[]), 0, 0

    for idx in range(80):
        coco_anno['categories'].append(
            dict(
                id=obj_idx_to_coco_id(idx),
                name=get_obj_name(idx).replace('_', ' ')))

    prog_bar = nncore.ProgressBar(num_tasks=len(mat_anno))
    for img_anno in mat_anno:
        anno = []
        for ins in img_anno[2][0]:
            if bool(ins[4]):
                continue

            for i in range(1, 3):
                cat_id = 1 if i == 1 else hoi_idx_to_coco_id(int(ins[0]) - 1)
                for bbox in ins[i][0]:
                    x1, x2, y1, y2 = map(int, bbox)
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    ins_anno = dict(
                        id=anno_id,
                        image_id=img_id,
                        category_id=cat_id,
                        iscrowd=0,
                        bbox=[x, y, w, h],
                        area=w * h)
                    anno.append(ins_anno)
                    anno_id += 1

        if len(anno) > 0:
            img_info = dict(
                id=img_id,
                file_name=str(img_anno[0][0]),
                width=int(img_anno[1][0][0][0]),
                height=int(img_anno[1][0][0][1]))
            coco_anno['images'].append(img_info)
            coco_anno['annotations'] += anno
            img_id += 1

        prog_bar.update()

    print(f'saving results to {out_file}...')
    nncore.dump(coco_anno, out_file)


def load_anno(anno_file, split):
    """
    Load annotations of HICO-DET [1] dataset.

    The loaded annotations will be stored in an ``N * 10`` tensor whose rows
    are in ``(img_id, hoi_idx, human_bbox, object_bbox)`` format. The
    ``hoi_idx`` will be in the range of ``0 ~ 599`` and the bboxes will be in
    ``xyxy`` style. Invisible human-object pairs will be discarded.

    Args:
        anno_file (str): Path to the ``anno_bbox.mat`` file.
        split (str): The dataset split to be loaded. Expected values are
            ``'train'`` and ``'test'``.

    Returns:
        :obj:`Tensor[N, 10]`: The loaded annotations.

    Example:
        >>> anno = load_anno('<path-to-anno_bbox.mat>', 'train')
        >>> print(anno)
        ... tensor([[img_id_1, hoi_idx_1, human_bbox_1, object_bbox_1]
        ...         [img_id_2, hoi_idx_2, human_bbox_2, object_bbox_2]
        ...         [img_id_3, hoi_idx_3, human_bbox_3, object_bbox_3]])

    References:
        1. Chao et al. (https://arxiv.org/abs/1702.05448)
    """
    assert split in ('train', 'test')
    nncore.is_file(anno_file, raise_error=True)

    mat_anno = loadmat(anno_file)[f'bbox_{split}'][0]
    collected = []

    for img_anno in mat_anno:
        hoi_idx, h_bbox, o_bbox = [], [], []

        for ins in img_anno[2][0]:
            if bool(ins[4]):
                continue

            h_bboxes = [[float(c) for c in b] for b in ins[1][0]]
            o_bboxes = [[float(c) for c in b] for b in ins[2][0]]

            for conn in ins[3]:
                hoi_idx.append(int(ins[0]) - 1)
                h_bbox.append(h_bboxes[conn[0] - 1])
                o_bbox.append(o_bboxes[conn[1] - 1])

        if (num_anno := len(hoi_idx)) > 0:
            img_id = int(img_anno[0][0][-12:-4])
            img_id = torch.full((num_anno, 1), img_id, dtype=torch.float)
            hoi_idx = torch.Tensor([hoi_idx]).t()
            h_bbox = torch.Tensor(h_bbox)[:, [0, 2, 1, 3]]
            o_bbox = torch.Tensor(o_bbox)[:, [0, 2, 1, 3]]

            anno = torch.cat((img_id, hoi_idx, h_bbox, o_bbox), dim=1)
            collected.append(anno)

    return torch.cat(collected)


@nncore.recursive()
def obj_idx_to_coco_id(obj_idx):
    """
    Convert object index (0 ~ 79) to COCO id (90 classes).

    Args:
        obj_idx (list[int] or int): The object index or list of object indexes.

    Returns:
        list[int] or int: The converted COCO id or list of COCO ids.
    """
    return OBJ_IDX_TO_COCO_ID[obj_idx]


@nncore.recursive()
def hoi_idx_to_coco_id(hoi_idx):
    """
    Convert HOI index (0 ~ 599) to COCO id (90 classes).

    Args:
        hoi_idx (list[int] or int): The HOI index or list of HOI indexes.

    Returns:
        list[int] or int: The converted COCO id or list of COCO ids.
    """
    obj_idx = hoi_idx_to_obj_idx(hoi_idx)
    return obj_idx_to_coco_id(obj_idx)


@nncore.recursive()
def hoi_idx_to_act_idx(hoi_idx):
    """
    Convert HOI index (0 ~ 599) to action index (0 ~ 116).

    Args:
        hoi_idx (list[int] or int): The HOI index or list of HOI indexes.

    Returns:
        list[int] or int: The converted action index or list of action indexes.
    """
    return HOI_IDX_TO_ACT_IDX[hoi_idx]


@nncore.recursive()
def hoi_idx_to_obj_idx(hoi_idx):
    """
    Convert HOI index (0 ~ 599) to object index (0 ~ 79).

    Args:
        hoi_idx (list[int] or int): The HOI index or list of HOI indexes.

    Returns:
        list[int] or int: The converted object index or list of object indexes.
    """
    return HOI_IDX_TO_OBJ_IDX[hoi_idx]


@nncore.recursive()
def act_idx_to_hoi_idx(act_idx):
    """
    Convert action index (0 ~ 116) to HOI index (0 ~ 599).

    Args:
        act_idx (list[int] or int): The action index or list of action indexes.

    Returns:
        list[list] or list[int]: The converted HOI indexes.
    """
    hoi_idx = []
    for idx in range(600):
        if HOI_IDX_TO_ACT_IDX[idx] == act_idx:
            hoi_idx.append(idx)
    return hoi_idx


@nncore.recursive()
def obj_idx_to_hoi_idx(obj_idx):
    """
    Convert object index (0 ~ 79) to HOI index (0 ~ 599).

    Args:
        obj_idx (list[int] or int): The object index or list of object indexes.

    Returns:
        list[list] or list[int]: The converted HOI indexes.
    """
    hoi_idx = []
    for idx in range(600):
        if HOI_IDX_TO_OBJ_IDX[idx] == obj_idx:
            hoi_idx.append(idx)
    return hoi_idx


def get_act_name(act_idx):
    """
    Get action name according to action index (0 ~ 116).

    Args:
        act_idx (int): The action index.

    Returns:
        str: The action name.
    """
    return ACT_IDX_TO_ACT_NAME[act_idx]


def get_obj_name(obj_idx):
    """
    Get object name according to object index (0 ~ 79).

    Args:
        act_idx (int): The object index.

    Returns:
        str: The object name.
    """
    return OBJ_IDX_TO_OBJ_NAME[obj_idx]


def get_act_and_obj_name(hoi_idx):
    """
    Get action and object name according to HOI index (0 ~ 599).

    Args:
        hoi_idx (int): The HOI index.

    Returns:
        tuple[str]: The action and object name.
    """
    act_idx = hoi_idx_to_act_idx(hoi_idx)
    obj_idx = hoi_idx_to_obj_idx(hoi_idx)

    act_name = get_act_name(act_idx)
    obj_name = get_obj_name(obj_idx)

    return act_name, obj_name


def get_hoi_name(hoi_idx):
    """
    Get HOI name according to HOI index (0 ~ 599).

    Args:
        hoi_idx (int): The HOI index.

    Returns:
        str: The HOI name.
    """
    act_name, obj_name = get_act_and_obj_name(hoi_idx)
    return '{}_{}'.format(act_name, obj_name)


def get_rare_hoi_idx():
    """
    Get rare HOI indexes (0 ~ 599).

    Returns:
        list[int]: The list of rare HOI indexes.
    """
    return RARE_HOI_IDX


def get_non_rare_hoi_idx():
    """
    Get non-rare HOI indexes (0 ~ 599).

    Returns:
        list[int]: The list of non-rare HOI indexes.
    """
    return [idx for idx in range(600) if idx not in RARE_HOI_IDX]


def get_seen_hoi_idx(type, id=None):
    """
    Get seen HOI indexes (0 ~ 599).

    Args:
        type (str): Type of the zero-shot scenario. Expected values include
            ``'uc'``, ``'uo'`` and ``'ua'``, representing unseen action-object
            combination, unseen object and unseen action scenarios.
        id (int, optional): Only valid when ``type='uc'``. Expected values are
            in the range of ``0 ~ 4``, indicating the 5 groups of unseen
            action-object combination settings in [1, 2].

    Returns:
        list[int]: The list of seen HOI indexes.

    References:
        1. Liu et al. (https://arxiv.org/abs/2008.06254)
        2. Bansal et al. (https://arxiv.org/abs/1904.03181)
    """
    unseen_idx = get_unseen_hoi_idx(type, id=id)
    return [idx for idx in range(600) if idx not in unseen_idx]


def get_unseen_hoi_idx(type, id=None):
    """
    Get unseen HOI indexes (0 ~ 599).

    Args:
        type (str): Type of the zero-shot scenario. Expected values include
            ``'uc'``, ``'uo'`` and ``'ua'``, representing unseen action-object
            combination, unseen object and unseen action scenarios.
        id (int, optional): Only valid when ``type='uc'``. Expected values are
            in the range of ``0 ~ 4``, indicating the 5 groups of unseen
            action-object combination settings in [1, 2].

    Returns:
        list[int]: The list of unseen HOI indexes.

    References:
        1. Liu et al. (https://arxiv.org/abs/2008.06254)
        2. Bansal et al. (https://arxiv.org/abs/1904.03181)
    """
    assert type in ('uc', 'uo', 'ua')
    if type == 'uc':
        return UC_HOI_IDX[id]
    elif type == 'uo':
        return UO_HOI_IDX
    else:
        return UA_HOI_IDX


def scatter_hum_to_hoi(tensor):
    """
    Scatter human to HOI classes.

    Args:
        tensor (:obj:`torch.Tensor[N, 1]`): The input tensor.

    Returns:
        :obj:`torch.Tensor[N, 600]`: The scattered tensor.
    """
    return tensor.repeat(1, 600)


def scatter_act_to_hoi(tensor):
    """
    Scatter action classes to HOI classes.

    Args:
        tensor (:obj:`torch.Tensor[N, 117]`): The input tensor.

    Returns:
        :obj:`torch.Tensor[N, 600]`: The scattered tensor.
    """
    return tensor[:, HOI_IDX_TO_ACT_IDX]


def scatter_obj_to_hoi(tensor):
    """
    Scatter object classes to HOI classes.

    Args:
        tensor (:obj:`torch.Tensor[N, 80]`): The input tensor.

    Returns:
        :obj:`torch.Tensor[N, 600]`: The scattered tensor.
    """
    return tensor[:, HOI_IDX_TO_OBJ_IDX]
