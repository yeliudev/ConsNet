# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import torch
import torch.nn as nn
from nncore.nn import build_mlp

from ..builder import BLOCKS


@BLOCKS.register()
class FusionBlock(nn.Module):

    def __init__(self, stream, h_dims, o_dims, l_dims, map_dims, ind_dims,
                 **kwargs):
        super(FusionBlock, self).__init__()
        self._stream = stream

        self.h_appr = build_mlp(h_dims, with_last_act=True, **kwargs)
        self.o_appr = build_mlp(o_dims, with_last_act=True, **kwargs)
        self.layout = build_mlp(l_dims, with_last_act=True, **kwargs)
        self.mapper = build_mlp(map_dims, **kwargs)
        self.indicator = build_mlp(ind_dims, **kwargs)

    def forward(self, data):
        h_feat = self.h_appr(data['h_appr'])
        o_feat = self.o_appr(data['o_appr'])
        layout = self.layout(data['layout'])

        hidden = torch.cat((h_feat, o_feat, layout), dim=1)

        emb = self.mapper(hidden)
        scores = self.indicator(hidden)

        return emb, scores
