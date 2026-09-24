from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

cfg_from_yaml_file(
    'cfgs/nuscenes_models/centerpoint_packed_temporal_4sweep_10e.yaml', cfg
)
logger = common_utils.create_logger('/tmp/packed_data_smoke.log')
dataset, loader, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
    batch_size=2, dist=False, workers=0, logger=logger,
    training=False, total_epochs=1, seed=666,
)
batch = next(iter(loader))
print('PACKED_DATA_OK', 'batch_size', batch['batch_size'],
      'voxels', batch['voxels'].shape,
      'coords', batch['voxel_coords'].shape,
      'coord_batch_minmax', batch['voxel_coords'][:, 0].min(),
      batch['voxel_coords'][:, 0].max())
assert int(batch['batch_size']) == 8
assert int(batch['voxel_coords'][:, 0].max()) >= 7
