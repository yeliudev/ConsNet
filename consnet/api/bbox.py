# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import nncore
import torch
from nncore.ops import bbox_iou


def pair_iou(bboxes1, bboxes2):
    """
    Compute the intersection-over-unions (IoUs) among human-object pairs.

    Args:
        bboxes1 (:obj:`Tensor[N, 8]`): Human-object pairs to be computed. They
            are expected to be in ``(x1, y1, x2, y2, ...)`` format.
        bboxes2 (:obj:`Tensor[M, 8]`): Human-object pairs to be computed. They
            are expected to be in ``(x1, y1, x2, y2, ...)`` format.

    Returns:
        :obj:`Tensor[N, M]`: The computed pairwise IoU values
    """
    assert bboxes1.size(1) == bboxes2.size(1) == 8
    h_iou = bbox_iou(bboxes1[:, :4], bboxes2[:, :4])
    o_iou = bbox_iou(bboxes1[:, 4:], bboxes2[:, 4:])
    return torch.min(h_iou, o_iou)


def pair_nms(bboxes,
             scores,
             method='fast',
             hard_thr=0.5,
             soft_thr=0.3,
             sigma=0.5,
             score_thr=1e-6):
    """
    Perform non-maximum suppression (NMS) on human-object pairs. This method
    supports multiple NMS types including Fast NMS [1], Cluster NMS [2], Normal
    NMS [3] and Soft NMS [4] with linear or gaussian suppression terms.

    Args:
        bboxes (:obj:`Tensor[N, 9]`): Batches of human-object pairs to be
            suppressed. The values are expected to be in
            ``(batch_id, x1, y1, x2, y2, ...)`` format.
        scores (:obj:`Tensor[N]`): Human-object interaction detection scores
            to be considered.
        method (str, optional): Type of NMS. Expected values include
            ``'fast'``, ``'cluster'``, ``'normal'``, ``'linear'`` and
            ``'gaussian'``, indicating Fast NMS, Cluster NMS, Normal NMS and
            Soft NMS with linear or gaussian suppression terms.
        hard_thr (float, optional): Hard threshold of NMS. This attribute
            is applied to all NMS methods. Human-object pairs with IoUs higher
            than this value will be discarded.
        soft_thr (float, optional): Soft threshold of NMS. This attribute
            is only applied to ``linear`` and ``gaussian`` methods.
            Human-object pairs with IoUs lower than ``hard_thr`` but higher
            than this value will be suppressed in a soft manner.
        sigma (float, optional): Hyperparameter for ``gaussian`` method.
        score_thr (float, optional): Score threshold. This attribute is
            applied to ``normal``, ``linear`` and ``gaussian`` methods.
            Human-object pairs with suppressed scores lower than this value
            will be discarded.

    Returns:
        :obj:`Tensor[N, 10]`: Human-object pairs and their updated scores \
            after NMS. The values are expected to be in \
            ``(batch_id, x1, y1, x2, y2, ..., score)`` format.

    References:
        1. Bolya et al. (https://arxiv.org/abs/1904.02689)
        2. Zheng et al. (https://arxiv.org/abs/2005.03572)
        3. Neubeck er al. (https://doi.org/10.1109/icpr.2006.479)
        4. Bodla et al. (https://arxiv.org/abs/1704.04503)
    """
    assert bboxes.size(1) == 9
    assert bboxes.size(0) == scores.size(0)
    assert method in ('fast', 'cluster', 'normal', 'linear', 'gaussian')

    if (num_bboxes := bboxes.size(0)) == 0:
        return torch.cat((bboxes, scores[:, None]), dim=1)

    if method in ('fast', 'cluster'):
        batch_ids = bboxes[:, None, 0]
        coors = (c := bboxes[:, 1:]) + batch_ids * (c.max() + 1)

        scores, inds = scores.sort(descending=True)
        bboxes, coors = bboxes[inds], coors[inds]
        iou = pair_iou(coors, coors).triu(diagonal=1)

        if method == 'fast':
            keep = iou.amax(dim=0) <= hard_thr
        else:
            c = iou
            for _ in range(num_bboxes):
                max_iou = (a := c).amax(dim=0)
                c = iou * (max_iou < hard_thr)[:, None].float().expand_as(a)
                if torch.equal(a, c):
                    break
            keep = max_iou < hard_thr

        blob = torch.cat((bboxes[keep], scores[keep, None]), dim=1)
    else:
        batch_ids, collected = bboxes[:, 0].unique(), []
        for batch_id in batch_ids:
            keep = bboxes[:, 0] == batch_id
            blob = torch.cat((bboxes[keep], scores[keep, None]), dim=1)

            num_bboxes = blob.size(0)
            for i in range(num_bboxes - 1):
                max_score, max_idx = blob[i:, -1].max(dim=0)
                if max_score < score_thr:
                    blob = blob[:i]
                    break

                blob = nncore.swap_element(blob, i, max_idx + i)
                iou = pair_iou(blob[i, None, 1:9], blob[i + 1:, 1:9])[0]
                blob[i + 1:, -1][iou >= hard_thr] = 0

                if method == 'normal':
                    continue

                keep = iou >= soft_thr
                if method == 'linear':
                    blob[i + 1:, -1][keep] *= (1 - iou[keep])
                else:
                    blob[i + 1:, -1][keep] *= (-iou[keep].pow(2) / sigma).exp()

            collected.append(blob)

        blob = torch.cat(collected)

    return blob
