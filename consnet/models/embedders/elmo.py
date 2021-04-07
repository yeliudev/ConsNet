# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import torch
from allennlp.modules.elmo import _ElmoBiLm, batch_to_ids
from allennlp.nn.util import remove_sentence_boundaries

from ..builder import EMBEDDERS


@EMBEDDERS.register()
class ELMo(object):

    def __init__(self, options, weights, level):
        self._elmo_bilm = _ElmoBiLm(options, weights)
        self._level = level

    def embed(self, tokens):
        character_ids = batch_to_ids([tokens])
        bilm_out = self._elmo_bilm(character_ids)

        wo_bos_eos = [
            remove_sentence_boundaries(layer, bilm_out['mask'])
            for layer in bilm_out['activations']
        ]

        emb = torch.cat([ele[0][:, None] for ele in wo_bos_eos], dim=1)
        sep = int(wo_bos_eos[0][1][0, :].sum())

        emb = emb[0, :, :sep, :].detach()[self._level]
        return emb
