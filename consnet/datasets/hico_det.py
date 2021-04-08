# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (ye-liu at whu.edu.cn)
# -----------------------------------------------------

import nncore
import torch
from torch.utils.data import Dataset

from consnet.api import get_unseen_hoi_idx, hico_det_eval, pair_nms
from .builder import DATASETS


@DATASETS.register()
class HICO_DET(Dataset):

    def __init__(self, blob, neg_pos_ub=0, zero_shot=None, eval=None):
        print('loading annotations into memory...')
        self._blob = blob
        self._neg_pos_ub = neg_pos_ub
        self._eval = eval
        self._key = 'gt' if neg_pos_ub == -1 else 'pos'
        self._neg_inds = []

        with nncore.open(blob) as f:
            if zero_shot is not None:
                hoi_idx = get_unseen_hoi_idx(**zero_shot)
                pos_label = torch.from_numpy(f[self._key][:, 2217:])
                inds = (pos_label[:, hoi_idx] == 0).all(dim=1).nonzero()[:, 0]
                self._pos_inds = inds.int().tolist()
            else:
                self._pos_inds = list(range(f[self._key].shape[0]))

    def __exit__(self):
        if hasattr(self._blob, 'close'):
            self._blob.close()

    def __len__(self):
        return len(self._pos_inds)

    def __getitem__(self, idx):
        if isinstance(self._blob, str):
            self._blob = nncore.open(self._blob)

        idx = self._pos_inds[idx]
        mix_blob = torch.from_numpy(self._blob[self._key][[idx]])

        if (fac := self._neg_pos_ub) > 0:
            if len(self._neg_inds) < fac:
                num_neg = self._blob['neg'].shape[0]
                self._neg_inds = torch.randperm(num_neg).tolist()
            inds = [self._neg_inds.pop() for _ in range(fac)]
            neg_blob = [torch.from_numpy(self._blob['neg'][[i]]) for i in inds]
            mix_blob = torch.cat((mix_blob, *neg_blob))

        mix_blob = list(mix_blob.split((9, 80, 80, 1024, 1024, 600), dim=1))
        mix_blob.append(mix_blob[5].any(dim=1).float()[:, None])
        mix_blob.append(self._compute_layout(mix_blob[0][:, 1:]))

        keys, data = [
            'meta', 'h_conf', 'o_conf', 'h_appr', 'o_appr', 'hoi_idx',
            'indicator', 'layout'
        ], []

        for i in range(mix_blob[0].size(0)):
            data.append({k: b[i] for k, b in zip(keys, mix_blob)})

        return data

    def _compute_layout(self, bboxes):
        h_bbox, o_bbox = bboxes.split(4, dim=1)

        lt = torch.min(h_bbox[:, :2], o_bbox[:, :2])
        rb = torch.max(h_bbox[:, 2:], o_bbox[:, 2:])

        wh = (rb - lt + 1).clamp(min=0)
        union = wh[:, None, 0] * wh[:, None, 1]

        hx = (h_bbox[:, None, 0] - lt[:, None, 0]) / union
        hy = (h_bbox[:, None, 2] - lt[:, None, 0]) / union
        hz = (h_bbox[:, None, 1] - lt[:, None, 1]) / union
        hw = (h_bbox[:, None, 3] - lt[:, None, 1]) / union

        ox = (o_bbox[:, None, 0] - lt[:, None, 0]) / union
        oy = (o_bbox[:, None, 2] - lt[:, None, 0]) / union
        oz = (o_bbox[:, None, 1] - lt[:, None, 1]) / union
        ow = (o_bbox[:, None, 3] - lt[:, None, 1]) / union

        return torch.cat((hx, hy, hz, hw, ox, oy, oz, ow), dim=1)

    def evaluate(self, blob, logger=None):
        cfg = self._eval.copy()
        score_thr, nms = cfg.pop('score_thr'), cfg.pop('nms')
        blob, collected = torch.cat(blob).cpu(), []

        nncore.log_or_print('Performing Pair NMS...', logger)
        prog_bar = nncore.ProgressBar(num_tasks=600)
        for hoi_idx in range(600):
            bboxes = blob[:, :9]
            scores = blob[:, hoi_idx + 9]

            keep = scores >= score_thr
            bboxes = bboxes[keep]
            scores = scores[keep]

            cls_blob = pair_nms(bboxes, scores, **nms)
            collected.append(cls_blob)

            prog_bar.update()

        results = hico_det_eval(collected, logger=logger, **cfg)
        return results
