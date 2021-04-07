# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

from mmdet.models import StandardRoIHead, build_roi_extractor

from ..builder import HEADS, build_head


@HEADS.register()
class HookedRoIHead(StandardRoIHead):

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def simple_test(self, x, proposal_list, img_metas):
        return self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=True)
