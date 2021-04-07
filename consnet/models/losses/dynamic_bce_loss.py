# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F

from consnet.api import get_unseen_hoi_idx
from ..builder import LOSSES


@LOSSES.register()
class DynamicBCELoss(nn.Module):

    def __init__(self, zero_shot=None, reduction='mean', loss_weight=1):
        super(DynamicBCELoss, self).__init__()
        self._reduction = reduction
        self._loss_weight = loss_weight

        if zero_shot is not None:
            self._unseen_idx = get_unseen_hoi_idx(**zero_shot)
        else:
            self._unseen_idx = None

    def forward(self, pred, target, weight=None):
        if weight is not None and self._unseen_idx is not None:
            weight[:, self._unseen_idx] = 0

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weight=weight, reduction=self._reduction)

        return loss * self._loss_weight
