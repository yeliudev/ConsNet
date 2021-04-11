# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import nncore
import torch

from .bbox import pair_iou
from .data import (get_hoi_name, get_non_rare_hoi_idx, get_rare_hoi_idx,
                   get_seen_hoi_idx, get_unseen_hoi_idx, hoi_idx_to_obj_idx,
                   load_anno, obj_idx_to_hoi_idx)


def _compute_ap(cls_anno, cls_blob):
    if (num_blob := cls_blob.size(0)) == 0:
        return 0, 0

    anno_map = dict()
    imgs = cls_anno[:, 0].unique().int().tolist()
    for img_id in imgs:
        anno_map[img_id] = [False] * sum(cls_anno[:, 0] == img_id).item()

    tp = torch.zeros(num_blob)
    fp = torch.zeros(num_blob)

    for i in range(num_blob):
        if (img_id := cls_blob[i, 0].int().item()) not in cls_anno[:, 0]:
            fp[i] = 1
            continue

        keep = cls_anno[:, 0] == img_id
        anno = cls_anno[keep][:, 2:]

        iou = pair_iou(anno, cls_blob[None, i, 1:9])
        max_iou, idx = iou.max(dim=0)

        if max_iou >= 0.5 and not anno_map[img_id][idx]:
            anno_map[img_id][idx] = True
            tp[i] = 1
        else:
            fp[i] = 1

    tp = tp.cumsum(0)
    fp = fp.cumsum(0)
    prc = tp / (tp + fp)
    rec = tp / cls_anno.size(0)

    ap = 0
    for i in range(11):
        p = prc[rec >= 0.1 * i]
        ap += p.max().item() / 11 if p.size(0) > 0 else 0

    rec = rec[-1].item()
    return ap, rec


@nncore.recursive(key='mode', type='dict')
def hico_det_eval(blob,
                  anno,
                  split='test',
                  mode=['def', 'ko'],
                  zero_shot=None,
                  logger=None):
    """
    Perform standard evaluation on HICO-DET dataset using mean average
    precision (mAP) as introduced in [1].

    Args:
        blob (list[:obj:`Tensor[N, 10]`]): Human-object pairs and their
            detection scores to be evaluated. The length of the list should be
            600 and each item should be an ``N * 10`` tensor in
            ``(batch_id, x1, y1, x2, y2, ..., score)`` format.
        anno (:obj:`torch.Tensor` or str): The annotations object or path to
            the ``anno_bbox.mat`` file.
        split (str, optional): The dataset split to be evaluated. Expected
            values are ``'train'`` and ``'test'``.
        mode (list[str] or str, optional): Mode of evaluation. Expected values
            are ``'def'``, ``'ko'`` or a list containing these terms,
            denoting the default mode and known-object mode introduced in [1].
        zero_shot (dict or None, optional): Configurations for zero-shot
            settings. It should contain the following fields:

            - `type` (str): Expected values include ``'uc'``, ``'ub'`` and \
                ``'ua'``, representing unseen action-object combination, \
                unseen object and unseen action scenarios introduced in [2].
            - `id` (int, optional): Only valid when ``type='uc'``. Expected \
                values are in the range of ``0 ~ 4``, indicating the 5 groups \
                of unseen action-object combination settings in [2, 3].

        logger (:obj:`logging.Logger` or str or None, optional): The potential
            logger or name of the logger to be used.

    Returns:
        dict: Evaluation results including mean average precision (mAP) and \
            mean recall (mRec) values of multiple dataset splits under \
            different evaluation modes.

    Example:
        >>> results = hico_det_eval(blob, '<path-to-anno_bbox.mat>')
        >>> print(results)
        ... {'def_mAP': xxx, 'def_mRec': xxx, ... }

    References:
        1. Chao et al. (https://arxiv.org/abs/1702.05448)
        2. Liu et al. (https://arxiv.org/abs/2008.06254)
        3. Bansal et al. (https://arxiv.org/abs/1904.03181)
    """
    assert mode in ('def', 'ko')
    nncore.log_or_print(f'Evaluating mAP in *{mode}* mode...', logger)

    if isinstance(anno, str):
        anno = load_anno(anno, split=split)

    rare_idx = get_rare_hoi_idx()
    non_rare_idx = get_non_rare_hoi_idx()

    if (zero_shot_mode := zero_shot is not None):
        seen_idx = get_seen_hoi_idx(**zero_shot)
        unseen_idx = get_unseen_hoi_idx(**zero_shot)

    ap, rec = torch.zeros(600), torch.zeros(600)
    for hoi_idx in range(600):
        cls_anno = anno[anno[:, 1] == hoi_idx]
        cls_blob = blob[hoi_idx]

        inds = cls_blob[:, -1].argsort(descending=True)
        cls_blob = cls_blob[inds]

        if mode == 'ko':
            obj_idx = hoi_idx_to_obj_idx(hoi_idx)
            keep_hoi_idx = obj_idx_to_hoi_idx(obj_idx)

            keep_imgs = torch.cat([
                anno[anno[:, 1] == idx][:, 0] for idx in keep_hoi_idx
            ]).unique()

            keep = torch.full_like(cls_blob[:, 0], False, dtype=torch.bool)
            for img_id in keep_imgs:
                keep += cls_blob[:, 0] == img_id
            cls_blob = cls_blob[keep]

        cls_ap, cls_rec = _compute_ap(cls_anno, cls_blob)
        ap[hoi_idx], rec[hoi_idx] = cls_ap, cls_rec

        if zero_shot_mode:
            cls_type = 'SEEN' if hoi_idx in seen_idx else 'UNSEEN'
        else:
            cls_type = 'RARE' if hoi_idx in rare_idx else 'NON_RARE'

        nncore.log_or_print(
            '{:03d} - {:<30} AP: {:.3f} | REC: {:.3f} | GT: {:<4} | '
            'DET: {:<6} | {}'.format(hoi_idx, get_hoi_name(hoi_idx),
                                     cls_ap, cls_rec, cls_anno.size(0),
                                     cls_blob.size(0), cls_type), logger)

    results = {
        f'{mode}_mAP': ap,
        f'{mode}_mRec': rec,
        f'{mode}_mAP_rare': ap[rare_idx],
        f'{mode}_mRec_rare': rec[rare_idx],
        f'{mode}_mAP_non_rare': ap[non_rare_idx],
        f'{mode}_mRec_non_rare': rec[non_rare_idx]
    }

    if zero_shot_mode:
        results.update({
            f'{mode}_mAP_seen': ap[seen_idx],
            f'{mode}_mRec_seen': rec[seen_idx],
            f'{mode}_mAP_unseen': ap[unseen_idx],
            f'{mode}_mRec_unseen': rec[unseen_idx]
        })

    results = {k: round(v.mean().item(), 4) for k, v in results.items()}
    return results
