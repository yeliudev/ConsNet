# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
from nncore.nn import kaiming_init_
from nncore.ops import cosine_similarity

from consnet.api import (scatter_act_to_hoi, scatter_hum_to_hoi,
                         scatter_obj_to_hoi)
from .builder import MODELS, build_block, build_loss


@MODELS.register()
class ConsNet(nn.Module):

    def __init__(self, modules, score_factor, train_cfg=None, test_cfg=None):
        super(ConsNet, self).__init__()
        self._score_factor = score_factor
        self._train_cfg = train_cfg
        self._test_cfg = test_cfg

        self.blocks = nn.ModuleDict(
            OrderedDict(
                {k: build_block(v, stream=k)
                 for k, v in modules.items()}))

        if train_cfg is not None:
            self.ind_loss = build_loss(train_cfg['ind_loss'])
            self.cls_loss = build_loss(train_cfg['cls_loss'])

        self.apply(lambda m: kaiming_init_(m)
                   if isinstance(m, nn.Linear) else None)

    def _scatter_to_hoi(self, stream, scores):
        if stream == 'h_emb':
            return scatter_hum_to_hoi(scores)
        elif stream == 'a_emb':
            return scatter_act_to_hoi(scores)
        elif stream == 'o_emb':
            return scatter_obj_to_hoi(scores)
        else:
            return scores

    def _con_blocks(self, data, blob):
        blob['conf'] = 1
        for stream in ['h_conf', 'o_conf']:
            blob['conf'] *= self.blocks[stream](data)
        return blob

    def _app_blocks(self, data, blob):
        blob['ind_scores'] = 0
        for stream in ['h_emb', 'o_emb', 'a_emb', 't_emb']:
            emb, ind = self.blocks[stream](data)
            blob['vis_emb'][stream] = emb
            blob['ind_scores'] += ind
        blob['ind_props'] = blob['ind_scores'].sigmoid()
        return blob

    def _sem_blocks(self, blob):
        blob['cls_scores'] = 0
        blob['sem_emb'] = self.blocks.sem_emb()
        for stream in ['h_emb', 'o_emb', 'a_emb', 't_emb']:
            similarity = cosine_similarity(
                blob['vis_emb'][stream],
                blob['sem_emb'][stream]) * self._score_factor
            blob['cls_scores'] += self._scatter_to_hoi(stream, similarity)
        blob['cls_props'] = blob['cls_scores'].sigmoid() * blob['conf']
        return blob

    def _generate_log(self, data, blob, output):
        pos = (data['indicator'] == 1).float()
        for thr in self._train_cfg.log_vars.ind:
            hits = (blob['ind_props'] > thr) == data['indicator']
            output[f'ind_acc_{thr}'] = hits.float().mean()
            output[f'ind_rec_{thr}'] = (
                hits * pos).sum() / pos.sum() if pos.sum() > 0 else 0
        for thr in self._train_cfg.log_vars.cls:
            hits = (blob['cls_props'] > thr) == data['hoi_idx']
            output[f'cls_acc_{thr}'] = hits.all(dim=1).float().mean()
        return output

    def _compute_loss(self, data, blob, output):
        output['ind_loss'] = self.ind_loss(blob['ind_scores'],
                                           data['indicator'])
        output['cls_loss'] = self.cls_loss(
            blob['cls_scores'], data['hoi_idx'], weight=blob['conf'])
        return output

    def _generate_det(self, data, blob, output):
        keep = blob['ind_props'][:, 0] > self._test_cfg['ind_thr']
        cls_props = blob['cls_props'] * blob['ind_props'].repeat(1, 600)
        output['_out'] = torch.cat((data['meta'], cls_props), dim=1)[keep]
        return output

    def forward(self, data, mode):
        blob = defaultdict(dict)

        blob = self._con_blocks(data, blob)
        blob = self._app_blocks(data, blob)
        blob = self._sem_blocks(blob)

        if mode == 'train':
            keep = (data['indicator'] == 1)[:, 0]
            data['hoi_idx'] = data['hoi_idx'][keep]
            for key in ['conf', 'cls_scores', 'cls_props']:
                blob[key] = blob[key][keep]

        output = OrderedDict(_num_samples=data['hoi_idx'].size(0))

        if mode == 'train':
            output = self._generate_log(data, blob, output)
            output = self._compute_loss(data, blob, output)
        else:
            output = self._generate_det(data, blob, output)

        return output
