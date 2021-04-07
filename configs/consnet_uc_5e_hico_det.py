# model settings
anno_root = 'data/hico_20160224_det/'
data_root = 'data/hico_det/'
zero_shot = dict(type='uc', id=0)
model = dict(
    type='ConsNet',
    modules=dict(
        h_conf=dict(type='ConfBlock'),
        o_conf=dict(type='ConfBlock'),
        h_emb=dict(
            type='MapperBlock',
            shared_dims=[1024, 1024],
            map_dims=[1024, 1024],
            ind_dims=[1024, 512, 1]),
        o_emb=dict(
            type='MapperBlock',
            shared_dims=[1024, 1024],
            map_dims=[1024, 1024],
            ind_dims=[1024, 512, 1]),
        a_emb=dict(
            type='FusionBlock',
            h_dims=[1024, 512],
            o_dims=[1024, 512],
            l_dims=[8, 128, 256],
            map_dims=[1280, 1024, 1024],
            ind_dims=[1280, 512, 1]),
        t_emb=dict(
            type='FusionBlock',
            h_dims=[1024, 512],
            o_dims=[1024, 512],
            l_dims=[8, 128, 256],
            map_dims=[1280, 1024, 1024],
            ind_dims=[1280, 512, 1]),
        sem_emb=dict(
            type='SemanticBlock',
            graph=data_root + 'consistency_graph.pkl',
            msg_pass_cfg=dict(type='GAT'),
            dims=[1024, 4096, 4096, 1024],
            heads=[8, 8, 4])),
    score_factor=10,
    train_cfg=dict(
        ind_loss=dict(type='DynamicBCELoss'),
        cls_loss=dict(
            type='DynamicBCELoss', zero_shot=zero_shot, loss_weight=80),
        log_vars=dict(ind=[0.5], cls=[0.5])),
    test_cfg=dict(ind_thr=0.01))
# dataset settings
dataset_type = 'HICO_DET'
data = dict(
    train=dict(
        type=dataset_type,
        blob=data_root + 'hico_det_train.hdf5',
        neg_pos_ub=3,
        zero_shot=zero_shot,
        loader=dict(batch_size=16, num_workers=4, shuffle=True)),
    test=dict(
        type=dataset_type,
        blob=data_root + 'hico_det_test.hdf5',
        eval=dict(
            anno=anno_root + 'anno_bbox.mat',
            zero_shot=zero_shot,
            score_thr=0.001,
            nms=dict(method='linear', hard_thr=1, soft_thr=0)),
        loader=dict(batch_size=256, num_workers=4, shuffle=False)))
