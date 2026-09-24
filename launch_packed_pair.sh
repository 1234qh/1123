#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sdb/tzm/OpenPCDet
source /mnt/sdb/25_tzm/dl_env/bin/activate
export PYTHONPATH=/home/tzm/.local/lib/python3.10/site-packages:/mnt/sdb/tzm/OpenPCDet:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEED_CKPT=/mnt/sdb/tzm/OpenPCDet/output/nuscenes_models/centerpoint_complete_baseline_4sweep_10e/complete_baseline_4gpu_b16_10e/ckpt/checkpoint_epoch_10.pth
COMMON=(--launcher pytorch --batch_size 16 --workers 4 --epochs 10
        --pretrained_model "$SEED_CKPT" --fix_random_seed
        --num_epochs_to_eval 1 --ckpt_save_interval 1)

python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
  tools/train.py --cfg_file tools/cfgs/nuscenes_models/centerpoint_packed_frame_baseline_4sweep_10e.yaml \
  "${COMMON[@]}" --extra_tag packed_frame_baseline_same_init

python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
  tools/train.py --cfg_file tools/cfgs/nuscenes_models/centerpoint_packed_temporal_4sweep_10e.yaml \
  "${COMMON[@]}" --extra_tag packed_temporal_same_init
