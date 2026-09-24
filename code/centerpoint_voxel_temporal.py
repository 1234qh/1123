"""CenterPoint with temporal attention on the single voxel encoder output.

This is the strict architecture requested for the detection comparison:

    multi-sweep points -> one shared VoxelResBackBone8x
    -> shifted-window temporal attention on encoded sparse voxels
    -> current-frame sparse tensor -> HeightCompression -> CenterHead

There is no PTv3 branch, semantic-segmentation checkpoint, occupancy mask,
motion predictor, or second point-cloud encoder.  The VoxelNet weights are
shared by all sweeps; the temporal attention is the only added module.
"""
import os
import sys

import torch
import torch.nn as nn

from .centerpoint import CenterPoint
from ..backbones_3d.spconv_backbone import VoxelResBackBone8x
from ...utils.spconv_utils import spconv

_PTV3_ROOT = os.environ.get("PTV3_ROOT", "/mnt/sdb/tzm/PointTransformerV3Main")
if _PTV3_ROOT not in sys.path:
    sys.path.insert(0, _PTV3_ROOT)
from temporal_segmentation_model import ShiftedBEVTemporalBlock


class _ZeroMotion(nn.Module):
    def forward(self, value):
        return value.new_zeros((value.shape[0], 3))


def _fixed_attention_block(dim, num_sweeps, window_size, shift):
    """Instantiate the original shifted-window attention without extra gates."""
    block = ShiftedBEVTemporalBlock(
        dim=dim,
        num_sweeps=num_sweeps,
        window_size=window_size,
        shift=shift,
        heads=8,
        capacity_per_frame=None,
        neighbors_per_frame=None,
        query_chunk_size=64,
    )
    # Keep the q/k/v, positional encoding, local windows and the original
    # residual update.  Disable only motion compensation.  The original
    # learnable residual_scale is intentionally retained and therefore
    # included in the optimizer; fixing it at 1.0 makes a randomly initialized
    # temporal branch perturb the baseline before it has learned anything.
    block.motion = _ZeroMotion()
    return block


class VoxelResBackBone8xTemporal(VoxelResBackBone8x):
    """The stock voxel backbone applied once to all sweep batches."""

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size,
                 point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg, input_channels=input_channels,
                         grid_size=grid_size, **kwargs)
        self.num_sweeps = int(model_cfg.get("NUM_SWEEPS", 4))
        # When enabled, the data processor has already voxelized each sweep
        # independently and the collate function has packed B*S frames into
        # the sparse batch dimension.  This flag is deliberately explicit:
        # silently guessing from batch_size is unsafe when B happens to be
        # divisible by S.
        self.separated_sweeps = bool(model_cfg.get("SEPARATED_SWEEPS", False))
        self.enable_temporal = bool(model_cfg.get("ENABLE_TEMPORAL", True))
        self.temporal_window = float(model_cfg.get("TEMPORAL_WINDOW", 2.5))
        self.voxel_size = torch.as_tensor(voxel_size, dtype=torch.float32)
        self.point_cloud_range = torch.as_tensor(point_cloud_range, dtype=torch.float32)
        self.temporal_blocks = nn.ModuleList()
        if self.enable_temporal:
            self.temporal_blocks.extend([
                _fixed_attention_block(self.num_point_features, self.num_sweeps,
                                       self.temporal_window, 0.0),
                _fixed_attention_block(self.num_point_features, self.num_sweeps,
                                       self.temporal_window, self.temporal_window / 2.0),
            ])

    def _metric_coords(self, indices):
        # encoded_spconv_tensor is stride 8 in x/y; conv_out downsamples z once
        # more than x/y, matching the original HeightCompression lattice.
        vs = self.voxel_size.to(indices.device, torch.float32)
        pc = self.point_cloud_range.to(indices.device, torch.float32)
        idx = indices.to(torch.float32)
        return torch.stack([
            (idx[:, 3] + 0.5) * vs[0] * 8.0 + pc[0],
            (idx[:, 2] + 0.5) * vs[1] * 8.0 + pc[1],
            (idx[:, 1] + 0.5) * vs[2] * 16.0 + pc[2],
        ], dim=1)

    def _temporal_fuse(self, out, original_batch_size):
        features = out.features
        indices = out.indices
        sample = indices[:, 0].long() // self.num_sweeps
        frame = indices[:, 0].long() % self.num_sweeps
        fused_features, fused_indices = [], []

        for sample_idx in range(original_batch_size):
            qmask = (sample == sample_idx) & (frame == 0)
            hmask = (sample == sample_idx) & (frame > 0)
            if not qmask.any():
                continue
            query_xyz = self._metric_coords(indices[qmask])
            query = features[qmask]
            if hmask.any():
                history_xyz = self._metric_coords(indices[hmask])
                history = features[hmask]
                history_frames = frame[hmask]
                history_time = history_frames.to(history.dtype) / max(self.num_sweeps - 1, 1)
                for block in self.temporal_blocks:
                    query = block(
                        query_xyz, query, history_xyz, history,
                        history_time, history_frames,
                    )
            cur_indices = indices[qmask].clone()
            cur_indices[:, 0] = sample_idx
            fused_features.append(query)
            fused_indices.append(cur_indices)

        if not fused_features:
            raise RuntimeError("Temporal voxel encoder produced no current-frame features")
        fused = torch.cat(fused_features, dim=0)
        current_indices = torch.cat(fused_indices, dim=0).int()
        return spconv.SparseConvTensor(
            features=fused,
            indices=current_indices,
            # ``self.sparse_shape`` is the input voxel grid.  The output
            # tensor has already been downsampled by the backbone; using the
            # input shape here would make HeightCompression allocate a huge
            # dense tensor.
            spatial_shape=out.spatial_shape,
            batch_size=original_batch_size,
        )

    def forward(self, batch_dict):
        voxel_features = batch_dict["voxel_features"]
        voxel_coords = batch_dict["voxel_coords"]
        packed_batch_size = int(batch_dict["batch_size"])
        if self.separated_sweeps:
            if packed_batch_size % self.num_sweeps != 0:
                raise RuntimeError(
                    f"Packed batch {packed_batch_size} is not divisible by "
                    f"NUM_SWEEPS={self.num_sweeps}"
                )
            original_batch_size = packed_batch_size // self.num_sweeps
            effective_batch_size = packed_batch_size
        else:
            original_batch_size = packed_batch_size
            effective_batch_size = original_batch_size * self.num_sweeps
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=effective_batch_size,
        )

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        out = self._temporal_fuse(out, original_batch_size)

        batch_dict.update({
            "encoded_spconv_tensor": out,
            "encoded_spconv_tensor_stride": 8,
            "multi_scale_3d_features": {
                "x_conv1": x_conv1, "x_conv2": x_conv2,
                "x_conv3": x_conv3, "x_conv4": x_conv4,
            },
            "multi_scale_3d_strides": {
                "x_conv1": 1, "x_conv2": 2,
                "x_conv3": 4, "x_conv4": 8,
            },
        })
        return batch_dict


class CenterPointVoxelTemporal(CenterPoint):
    """CenterPoint plus only post-voxel-encoder temporal attention."""

    def build_networks(self):
        self.module_topology = [
            "vfe", "backbone_3d", "map_to_bev_module", "pfe",
            "backbone_2d", "dense_head", "point_head", "roi_head",
        ]
        return super().build_networks()

    def build_backbone_3d(self, model_info_dict):
        cfg = self.model_cfg.BACKBONE_3D
        module = VoxelResBackBone8xTemporal(
            model_cfg=cfg,
            input_channels=model_info_dict["num_point_features"],
            grid_size=model_info_dict["grid_size"],
            voxel_size=model_info_dict["voxel_size"],
            point_cloud_range=model_info_dict["point_cloud_range"],
        )
        model_info_dict["module_list"].append(module)
        model_info_dict["num_point_features"] = module.num_point_features
        model_info_dict["backbone_channels"] = module.backbone_channels
        return module, model_info_dict
