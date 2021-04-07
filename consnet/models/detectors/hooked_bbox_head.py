# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import torch
from mmcv.ops import batched_nms
from mmdet.models import Shared2FCBBoxHead

from ..builder import HEADS


def _hooked_nms(bboxes, scores, feat, score_thr, nms, max_per_img=-1):
    scores = scores[:, :-1]
    num_samples, num_classes = scores.size()

    if bboxes.size(1) > 4:
        bboxes = bboxes.view(num_samples, -1, 4)
    else:
        bboxes = bboxes[:, None].expand(num_samples, num_classes, 4)

    gt_blob = feat.clone()
    feat = feat.repeat_interleave(80, dim=0)
    conf = scores.repeat_interleave(80, dim=0)

    labels = bboxes.new_tensor([0] + [1] * (num_classes - 1), dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    valid_mask = scores > score_thr

    inds = valid_mask.nonzero()[:, 0]
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if bboxes.numel() == 0:
        return gt_blob, bboxes.new_empty(0, 1110)

    dets, keep = batched_nms(bboxes, scores, labels, nms)

    if max_per_img > 0:
        dets = dets[:max_per_img]
        keep = keep[:max_per_img]

    bbox = dets[:, :4]
    conf = conf[inds][keep]
    feat = feat[inds][keep]
    score = dets[:, None, -1]
    label = labels[keep, None]

    dt_blob = torch.cat((bbox, conf, feat, score, label), dim=1)
    return gt_blob, dt_blob


@HEADS.register()
class HookedBBoxHead(Shared2FCBBoxHead):

    def get_bboxes(self, *args, **kwargs):
        cfg = kwargs.pop('cfg')
        blobs = super(HookedBBoxHead, self).get_bboxes(*args, **kwargs)

        num_rois, gt_blobs, dt_blobs = blobs[0][0].size(0), [], []
        for i, blob in enumerate(zip(*blobs)):
            feat = self._feat[i * num_rois:(i + 1) * num_rois]
            gt_blob, dt_blob = _hooked_nms(*blob, feat, **cfg)
            gt_blobs.append(gt_blob)
            dt_blobs.append(dt_blob)

        return gt_blobs, dt_blobs

    def forward(self, x):
        x = self.shared_fcs[0](x.flatten(1))
        self._feat = x.clone()
        x = self.relu(self.shared_fcs[1](self.relu(x)))

        cls_score = self.fc_cls(x)
        bbox_pred = self.fc_reg(x)

        return cls_score, bbox_pred
