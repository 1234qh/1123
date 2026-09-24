"""VFE helpers for metadata-rich nuScenes point inputs."""
import torch

from .mean_vfe import MeanVFE


class CenterPointFiveFeatureVFE(MeanVFE):
    """Keep the stock CenterPoint VFE contract for seven-channel points.

    The dataset supplies ``[x, y, z, intensity, ring, sweep_id, timestamp]``.
    Ring and sweep id are consumed only by the raw-point temporal branch; the
    ordinary CenterPoint branch remains exactly five-dimensional.
    """

    def __init__(self, model_cfg, num_point_features, **kwargs):
        if int(num_point_features) != 7:
            raise ValueError(
                "CenterPointFiveFeatureVFE expects seven input features "
                "[x,y,z,intensity,ring,sweep_id,timestamp], got "
                f"{num_point_features}."
            )
        super().__init__(model_cfg, num_point_features=5, **kwargs)

    def forward(self, batch_dict, **kwargs):
        voxels = batch_dict['voxels']
        if voxels.shape[-1] != 7:
            raise RuntimeError(
                f"Expected seven-channel voxels, got {tuple(voxels.shape)}."
            )
        centerpoint_voxels = torch.cat(
            [voxels[:, :, :4], voxels[:, :, 6:7]], dim=-1
        )
        num_points = batch_dict['voxel_num_points']
        normalizer = torch.clamp_min(num_points.view(-1, 1), 1).type_as(
            centerpoint_voxels
        )
        batch_dict['voxel_features'] = (
            centerpoint_voxels.sum(dim=1) / normalizer
        ).contiguous()
        return batch_dict


class TemporalMeanVFE(MeanVFE):
    """Build CenterPoint features plus per-sweep sparse history features.

    Input points are [x, y, z, intensity, sweep_id, timestamp].  The normal
    CenterPoint branch receives its original five features, while history
    points are grouped by sweep so a shared sparse backbone can encode them
    independently in the temporal branch.
    """

    def __init__(self, model_cfg, num_point_features, **kwargs):
        super().__init__(model_cfg, num_point_features=5, **kwargs)
        self.num_sweeps = int(model_cfg.get('NUM_SWEEPS', 10))

    def forward(self, batch_dict, **kwargs):
        voxels = batch_dict['voxels']
        num_points = batch_dict['voxel_num_points']
        coords = batch_dict['voxel_coords']

        # Restore the exact five-channel VFE input used by original CenterPoint.
        base = torch.cat([voxels[:, :, :4], voxels[:, :, -1:]], dim=-1)
        temporal_base = torch.cat([voxels[:, :, :5], voxels[:, :, -1:]], dim=-1)
        normalizer = torch.clamp_min(num_points.view(-1, 1), 1).type_as(base)
        batch_dict['voxel_features'] = (base.sum(dim=1) / normalizer).contiguous()

        sweep_ids = voxels[:, :, -2].long()
        point_mask = torch.arange(voxels.shape[1], device=voxels.device)[None, :] < num_points[:, None]
        hist_features, hist_coords = [], []
        for sweep in range(1, self.num_sweeps):
            mask = point_mask & (sweep_ids == sweep)
            count = mask.sum(dim=1)
            valid = count > 0
            if not valid.any():
                continue
            feature_sum = (temporal_base * mask[..., None].type_as(temporal_base)).sum(dim=1)
            hist_features.append(feature_sum[valid] / count[valid, None].type_as(temporal_base))
            sweep_coords = coords[valid].clone()
            # Expanded batch index keeps same-space voxels from different
            # sweeps distinct while using the original sparse backbone.
            sweep_coords[:, 0] = sweep_coords[:, 0] * self.num_sweeps + sweep
            hist_coords.append(sweep_coords)

        if hist_features:
            batch_dict['temporal_history_voxel_features'] = torch.cat(hist_features, dim=0).contiguous()
            batch_dict['temporal_history_voxel_coords'] = torch.cat(hist_coords, dim=0).contiguous()
        else:
            batch_dict['temporal_history_voxel_features'] = temporal_base.new_zeros((0, 6))
            batch_dict['temporal_history_voxel_coords'] = coords.new_zeros((0, 4))

        current_mask = point_mask & (sweep_ids == 0)
        current_count = current_mask.sum(dim=1)
        current_valid = current_count > 0
        current_sum = (temporal_base * current_mask[..., None].type_as(temporal_base)).sum(dim=1)
        batch_dict['temporal_current_voxel_features'] = (current_sum[current_valid] / current_count[current_valid, None].type_as(temporal_base)).contiguous()
        batch_dict['temporal_current_voxel_coords'] = coords[current_valid].contiguous()
        batch_dict['temporal_current_voxel_indices'] = torch.nonzero(current_valid, as_tuple=False).flatten().contiguous()
        return batch_dict


class CenterPointSweepVFE(MeanVFE):
    """Split the already-voxelized sweeps for one shared voxel encoder.

    Each output entry is still the standard five-channel CenterPoint VFE
    feature ``[x, y, z, intensity, timestamp]``.  The only difference is the
    batch index: ``sample * NUM_SWEEPS + sweep`` keeps sweeps separate while
    the same VoxelResBackBone8x weights process all of them in one sparse
    forward.  No PTv3 or second point encoder is involved.
    """

    def __init__(self, model_cfg, num_point_features, **kwargs):
        if int(num_point_features) != 7:
            raise ValueError(
                "CenterPointSweepVFE expects [x,y,z,intensity,ring,sweep_id,time]"
            )
        super().__init__(model_cfg, num_point_features=5, **kwargs)
        self.num_sweeps = int(model_cfg.get("NUM_SWEEPS", 4))
        self.separated_sweeps = bool(model_cfg.get("SEPARATED_SWEEPS", False))

    def forward(self, batch_dict, **kwargs):
        voxels = batch_dict["voxels"]
        num_points = batch_dict["voxel_num_points"].long()
        coords = batch_dict["voxel_coords"]
        if voxels.shape[-1] != 7:
            raise RuntimeError(f"Expected seven-channel voxels, got {voxels.shape}")

        # Preserve the exact stock CenterPoint five-channel physical feature.
        base = torch.cat([voxels[:, :, :4], voxels[:, :, 6:7]], dim=-1)
        if self.separated_sweeps:
            # The data processor already made one independent voxel set per
            # sweep.  Do not split/merge again; collate has encoded the frame
            # in the sparse batch index.
            normalizer = torch.clamp_min(
                num_points.view(-1, 1), 1
            ).type_as(base)
            batch_dict["voxel_features"] = (
                base.sum(dim=1) / normalizer
            ).contiguous()
            batch_dict["temporal_num_sweeps"] = self.num_sweeps
            return batch_dict

        point_mask = (
            torch.arange(voxels.shape[1], device=voxels.device)[None, :]
            < num_points[:, None]
        )
        sweep_ids = voxels[:, :, 5].round().long()
        out_features, out_coords = [], []
        for sweep in range(self.num_sweeps):
            mask = point_mask & (sweep_ids == sweep)
            count = mask.sum(dim=1)
            valid = count > 0
            if not valid.any():
                continue
            feature_sum = (base * mask[..., None].type_as(base)).sum(dim=1)
            features = feature_sum[valid] / count[valid, None].type_as(base)
            sweep_coords = coords[valid].clone()
            sweep_coords[:, 0] = sweep_coords[:, 0] * self.num_sweeps + sweep
            out_features.append(features.contiguous())
            out_coords.append(sweep_coords.contiguous())

        if not out_features:
            raise RuntimeError("No non-empty sweep voxels were produced")
        batch_dict["voxel_features"] = torch.cat(out_features, dim=0)
        batch_dict["voxel_coords"] = torch.cat(out_coords, dim=0)
        batch_dict["temporal_num_sweeps"] = self.num_sweeps
        return batch_dict
