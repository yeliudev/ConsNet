# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import nncore
import torch
import torch.nn as nn
from nncore.nn import build_msg_pass_modules

from ..builder import BLOCKS


@BLOCKS.register()
class SemanticBlock(nn.Module):

    def __init__(self, stream, dims, graph=None, **kwargs):
        super(SemanticBlock, self).__init__()
        self._stream = stream
        self._use_cache = False
        self._cache = None

        if graph is not None and nncore.is_file(graph):
            graph = nncore.load(graph)
            nodes, graph = graph['nodes'], graph['graph']
        else:
            nodes, graph = torch.empty(798, 1024), torch.empty(798, 798)

        self.register_buffer('nodes', nodes)
        self.register_buffer('graph', graph)

        self.msg_pass = build_msg_pass_modules(dims, **kwargs)

    def train(self, mode=True):
        self._use_cache = not mode
        self._cache = None
        return super(SemanticBlock, self).train(mode=mode)

    def propagate(self):
        x = self.nodes
        for layer in self.msg_pass:
            x = layer(x, self.graph)
        return x

    def forward(self):
        if self._use_cache:
            if self._cache is None:
                self._cache = self.propagate()
            emb = self._cache
        else:
            emb = self.propagate()

        emb = dict(
            h_emb=emb[None, 600],
            o_emb=emb[718:],
            a_emb=emb[601:718],
            t_emb=emb[:600])

        return emb
