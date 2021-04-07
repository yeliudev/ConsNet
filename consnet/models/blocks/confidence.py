# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import torch.nn as nn

from consnet.api import scatter_hum_to_hoi, scatter_obj_to_hoi
from ..builder import BLOCKS


@BLOCKS.register()
class ConfBlock(nn.Module):

    def __init__(self, stream):
        super(ConfBlock, self).__init__()
        self._stream = stream

    def forward(self, data):
        if self._stream == 'h_conf':
            conf = scatter_hum_to_hoi(data['h_conf'][:, None, 0])
        else:
            conf = scatter_obj_to_hoi(data['o_conf'])
        return conf
