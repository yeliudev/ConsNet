# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import torch.nn as nn
from nncore.nn import build_mlp

from ..builder import BLOCKS


@BLOCKS.register()
class MapperBlock(nn.Module):

    def __init__(self, stream, shared_dims, map_dims, ind_dims, **kwargs):
        super(MapperBlock, self).__init__()
        self._stream = stream

        self.shared = build_mlp(shared_dims, with_last_act=True, **kwargs)
        self.mapper = build_mlp(map_dims, **kwargs)
        self.indicator = build_mlp(ind_dims, **kwargs)

    def forward(self, data):
        feat = data[f'{self._stream[0]}_appr']
        hidden = self.shared(feat)

        emb = self.mapper(hidden)
        scores = self.indicator(hidden)

        return emb, scores
