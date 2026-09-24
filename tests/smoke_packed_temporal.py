import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def run(path, expect_temporal):
    cfg_from_yaml_file(path, cfg)
    logger = common_utils.create_logger('/tmp/packed_temporal_smoke.log')
    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=0, logger=logger,
        training=True, total_epochs=1, seed=666,
    )
    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset).cuda().train()
    batch = next(iter(loader))
    load_data_to_gpu(batch)
    assert int(batch['batch_size']) == 4, batch['batch_size']
    assert int(batch['voxel_coords'][:, 0].max()) >= 3
    ret, tb, _ = model(batch)
    loss = ret['loss']
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for n, p in model.named_parameters()
             if 'temporal_blocks' in n and p.requires_grad]
    if expect_temporal:
        assert grads and any(g is not None and torch.isfinite(g).all()
                             and g.abs().sum() > 0 for g in grads)
    print('PACKED_OK', path, 'temporal', expect_temporal,
          'loss', float(loss), 'voxels', int(batch['voxel_features'].shape[0]))


run('cfgs/nuscenes_models/centerpoint_packed_frame_baseline_4sweep_10e.yaml', False)
run('cfgs/nuscenes_models/centerpoint_packed_temporal_4sweep_10e.yaml', True)
