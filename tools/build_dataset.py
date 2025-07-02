# -----------------------------------------------------
# ConsNet
# Licensed under the GNU General Public License v3.0
# Written by Ye Liu (coco.ye.liu@connect.polyu.hk)
# -----------------------------------------------------

import argparse

import nncore
import torch
import torch.nn.functional as F
from nncore.ops import cosine_similarity

from consnet.api import (get_act_and_obj_name, get_act_name, get_obj_name,
                         hoi_idx_to_act_idx, hoi_idx_to_obj_idx, load_anno,
                         pair_iou)
from consnet.models import build_dataloader, build_detector, build_embedder


def detect_objects(model, data_loader, annos, split):
    anno, blob = annos[split], dict()

    print(f'detecting objects in *{split}* split')
    prog_bar = nncore.ProgressBar(num_tasks=len(data_loader))
    for data in data_loader:
        img, img_metas = data['img'][0].data[0], data['img_metas'][0].data[0]
        data['proposals'], bboxes, img_ids, img_annos = [[]], [], [], []

        for i in range(num_imgs := img.size(0)):
            img_ids.append(img_id := int(img_metas[i]['filename'][-12:-4]))
            img_annos.append(img_anno := anno[anno[:, 0] == img_id])
            bboxes.append(torch.cat(img_anno[:, 2:].split(4, dim=1)))
            data['proposals'][0].append(bboxes[i].to(img.device))

        with torch.no_grad():
            gt_blobs, dt_blobs = model(return_loss=False, **data)

        for i in range(num_imgs):
            num_anno = img_annos[i].size(0)
            inds = hoi_idx_to_obj_idx(img_annos[i][:, 1].int().tolist())

            (h_conf := torch.zeros((num_anno, 80)))[:, 0] = 1
            (o_conf := torch.zeros((num_anno, 80)))[range(num_anno), inds] = 1

            conf = torch.cat((h_conf, o_conf))
            gt_blobs[i] = torch.cat((bboxes[i], conf, gt_blobs[i]), dim=1)

            blob[img_ids[i]] = dict(gt=gt_blobs[i], dt=dt_blobs[i])

        prog_bar.update()

    return blob


@nncore.open(mode='a', format='h5')
def build_dataset(annos, blobs, cfg, split, f):
    anno, blob, cfg = annos[split], blobs[split], cfg[split]
    img_list = anno[:, 0].unique().int().tolist()

    print(f'Building *{split}* split of the dataset')
    prog_bar = nncore.ProgressBar(num_tasks=len(img_list))
    for img_id in img_list:
        img_anno = anno[anno[:, 0] == img_id][:, 1:]
        img_blob = blob[img_id]

        gt_blob = img_blob['gt']
        dt_blob = img_blob['dt']

        gt_blob = gt_blob.split(int(gt_blob.size(0) / 2))
        gt_blob = [b.split((4, 80, 1024), dim=1) for b in gt_blob]
        gt_blob = torch.cat(nncore.concat_list(zip(*gt_blob)), dim=1)

        h_blob = dt_blob[dt_blob[:, -1] == 0][:, :-1]
        o_blob = dt_blob[dt_blob[:, -1] == 1][:, :-1]

        if (sep := cfg.max_h_as_o) > 0:
            h_inds = h_blob[:, -1].argsort(descending=True)
            o_blob = torch.cat((o_blob, h_blob[h_inds[:sep]]))
        else:
            o_blob = torch.cat((o_blob, h_blob))

        h_blob = h_blob[h_blob[:, -1] > cfg.score_thr.hum]
        o_blob = o_blob[o_blob[:, -1] > cfg.score_thr.obj]

        h_inds = h_blob[:, -1].argsort(descending=True)
        o_inds = o_blob[:, -1].argsort(descending=True)

        if (max_num := cfg.max_per_img.hum) > 0:
            h_inds = h_inds[:max_num]

        if (max_num := cfg.max_per_img.obj) > 0:
            o_inds = o_inds[:max_num]

        num_h = h_inds.size(0)
        num_o = o_inds.size(0)

        h_blob = h_blob[h_inds][:, :-1].repeat_interleave(num_o, dim=0)
        o_blob = o_blob[o_inds][:, :-1].repeat(num_h, 1)

        h_blob = h_blob.split((4, 80, 1024), dim=1)
        o_blob = o_blob.split((4, 80, 1024), dim=1)
        dt_blob = torch.cat(nncore.concat_list(zip(h_blob, o_blob)), dim=1)

        if dt_blob.numel() > 0:
            m_iou = pair_iou(dt_blob[:, :8], img_anno[:, 1:]).amax(dim=1)
            po_inds = (m_iou >= cfg.iou_thr.pos).nonzero()[:, 0]
            ne_inds = (m_iou < cfg.iou_thr.neg).nonzero()[:, 0]
        else:
            po_inds = ne_inds = []

        po_blob = dt_blob[po_inds]
        ne_blob = dt_blob[ne_inds]

        gt_label = torch.zeros((gt_blob.size(0), 600))
        po_label = torch.zeros((po_blob.size(0), 600))
        ne_label = torch.zeros((ne_blob.size(0), 600))

        gt_iou = pair_iou(gt_blob[:, :8], img_anno[:, 1:])
        po_iou = pair_iou(po_blob[:, :8], img_anno[:, 1:])

        for i, iou in enumerate(gt_iou):
            gt_label[i][img_anno[:, 0][iou >= 0.5].long()] = 1

        for i, iou in enumerate(po_iou):
            po_label[i][img_anno[:, 0][iou >= 0.5].long()] = 1

        gt_img_id = torch.full((gt_blob.size(0), 1), img_id)
        po_img_id = torch.full((po_blob.size(0), 1), img_id)
        ne_img_id = torch.full((ne_blob.size(0), 1), img_id)

        gt_blob = torch.cat((gt_img_id, gt_blob, gt_label), dim=1)
        po_blob = torch.cat((po_img_id, po_blob, po_label), dim=1)
        ne_blob = torch.cat((ne_img_id, ne_blob, ne_label), dim=1)

        if split == 'train':
            po_blob = torch.cat((po_blob, gt_blob))
            if (fac := cfg.neg_pos_ub) > 0:
                ne_blob = ne_blob[:po_blob.size(0) * fac]
            nncore.dump(po_blob.numpy(), f, format='h5', dataset='pos')
            nncore.dump(ne_blob.numpy(), f, format='h5', dataset='neg')
        else:
            po_blob = torch.cat((po_blob, ne_blob))
            nncore.dump(gt_blob.numpy(), f, format='h5', dataset='gt')
            nncore.dump(po_blob.numpy(), f, format='h5', dataset='pos')

        prog_bar.update()


def build_graph(annos, blobs, cfg):

    def _link(graph, a, b):
        graph[a, b] = graph[b, a] = 1

    print('building consistency graph...')
    size = dict(act=117, obj=80, tri=600)
    feat = {k: [[] for _ in range(v)] for k, v in size.items()}

    for split in ['train', 'test']:
        anno = annos[split]
        img_list = anno[:, 0].unique().int().tolist()

        for img_id in img_list:
            hoi_list = anno[anno[:, 0] == img_id][:, 1].int().tolist()
            img_feat = blobs[split][img_id]['gt'][:, 84:]
            base_idx = int(len(hoi_list) / 2)

            for i, hoi_idx in enumerate(hoi_list):
                act_idx = hoi_idx_to_act_idx(hoi_idx)
                obj_idx = hoi_idx_to_obj_idx(hoi_idx)

                a_feat = img_feat[i]
                o_feat = img_feat[base_idx + i]

                feat['act'][act_idx].append(a_feat)
                feat['obj'][obj_idx].append(o_feat)
                feat['tri'][hoi_idx].append((a_feat + o_feat) / 2)

    for key in feat:
        cls_feat = [torch.stack(f).mean(dim=0) for f in feat[key]]
        feat[key] = torch.stack(cls_feat)

    embedder = build_embedder(cfg.embedder)

    emb = dict(hum=[embedder.embed(['human'])[0]], act=[], obj=[], tri=[])
    u_w, t_w, e_w = cfg.uni_weight, cfg.tri_weight, cfg.emb_weight

    for idx in range(117):
        tokens = get_act_name(idx).split('_')
        out = embedder.embed(tokens)
        out = out[0] * u_w.act + out[-1] * (1 - u_w.act)
        emb['act'].append(out)

    for idx in range(80):
        tokens = get_obj_name(idx).split('_')
        out = embedder.embed(tokens)
        out = out[0] * u_w.obj + out[-1] * (1 - u_w.obj)
        emb['obj'].append(out)

    for idx in range(600):
        a_name, o_name = get_act_and_obj_name(idx)
        a_t = a_name.split('_')
        o_t = o_name.split('_')
        tokens = ['human'] + a_t + o_t
        out = embedder.embed(tokens)

        a_emb, o_emb = out[1:].split((len(a_t), len(o_t)))
        a_emb = a_emb[0] * u_w.act + a_emb[-1] * (1 - u_w.act)
        o_emb = o_emb[0] * u_w.obj + o_emb[-1] * (1 - u_w.obj)

        t_emb = out[0] * t_w.hum + a_emb * t_w.act + o_emb * t_w.obj
        emb['tri'].append(t_emb)

    emb = {k: torch.stack(v) for k, v in emb.items()}

    mix_emb = dict()
    for key in feat:
        vis_emb = F.normalize(feat[key]) * e_w.vis
        sem_emb = F.normalize(emb[key]) * e_w.sem
        mix_emb[key] = torch.cat((vis_emb, sem_emb), dim=1)

    edges = dict()
    for key, m_emb in mix_emb.items():
        similarity = cosine_similarity(m_emb, m_emb)
        inds = similarity.argsort(descending=True)
        edges[key] = inds[:, :cfg.num_edges[key] + 1].tolist()

    base_idx = dict(act=601, obj=718, tri=0)
    graph = torch.zeros(798, 798)

    for idx in range(600):
        _link(graph, idx, 600)
        _link(graph, idx, hoi_idx_to_act_idx(idx) + base_idx['act'])
        _link(graph, idx, hoi_idx_to_obj_idx(idx) + base_idx['obj'])

    for key, idx in base_idx.items():
        for i, edge in enumerate(edges[key]):
            for j in edge:
                _link(graph, i + idx, j + idx)

    nodes = torch.cat((emb['tri'], emb['hum'], emb['act'], emb['obj']))
    return dict(nodes=nodes, graph=graph)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--anno',
        help='annotation file',
        default='data/hico_20160224_det/anno_bbox.mat')
    parser.add_argument(
        '--config', help='config file', default='configs/build.py')
    parser.add_argument(
        '--checkpoint',
        help='checkpoint file',
        default='checkpoints/faster_rcnn_r50_fpn_3x_coco-26df6f6b.pth')
    parser.add_argument(
        '--out', help='output directory', default='data/hico_det')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = nncore.Config.from_file(args.config)

    model = build_detector(cfg.model, args.checkpoint)

    annos, blobs = dict(), dict()
    for split in ['train', 'test']:
        annos[split] = load_anno(args.anno, split)

        data_loader = build_dataloader(
            cfg.data[split],
            num_samples=cfg.data.samples_per_gpu,
            num_workers=cfg.data.workers_per_gpu)

        blobs[split] = detect_objects(model, data_loader, annos, split)

        file = nncore.join(args.out, f'hico_det_{split}.hdf5')
        build_dataset(annos, blobs, cfg.build, split, file=file)

    graph = build_graph(annos, blobs, cfg.graph)
    nncore.dump(graph, nncore.join(args.out, 'consistency_graph.pkl'))


if __name__ == '__main__':
    main()
