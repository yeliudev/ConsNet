# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import torch
from mmdet.models import TwoStageDetector

from ..builder import DETECTORS, build_head


@DETECTORS.register()
class HookedRCNN(TwoStageDetector):

    def __init__(self, roi_head, test_cfg, **kwargs):
        super(HookedRCNN, self).__init__(
            roi_head=None, test_cfg=test_cfg, **kwargs)
        self.roi_head = build_head(roi_head, test_cfg=test_cfg.rcnn)

    def simple_test(self, img, img_metas, proposals=None):
        factors = [img.new_tensor(meta['scale_factor']) for meta in img_metas]
        num_imgs = len(img_metas)

        x = self.extract_feat(img)

        if proposals is not None:
            props = [props * fac for props, fac in zip(proposals, factors)]
            gt_blobs, _ = self.roi_head.simple_test(
                x, props, img_metas=img_metas)
        else:
            gt_blobs = [torch.empty(0, 1024) for _ in range(num_imgs)]

        proposals = self.rpn_head.simple_test_rpn(x, img_metas)
        _, dt_blobs = self.roi_head.simple_test(
            x, proposals, img_metas=img_metas)

        proposals = [dt_blobs[i][:, :4] * fac for i, fac in enumerate(factors)]
        blobs, _ = self.roi_head.simple_test(x, proposals, img_metas=img_metas)

        for i in range(num_imgs):
            dt_blobs[i][:, 84:1108] = blobs[i][-dt_blobs[i].size(0):]

        gt_blobs = [blob.cpu() for blob in gt_blobs]
        dt_blobs = [blob.cpu() for blob in dt_blobs]

        return gt_blobs, dt_blobs
