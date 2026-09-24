"""PTv3 backbone with point-level local temporal attention for nuScenes lidarseg."""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

from model import PointTransformerV3


class ShiftedBEVTemporalBlock(nn.Module):
    """Full current-to-history cross-attention inside a BEV window.

    Every real history point in the matching window participates as K/V. The
    only geometric input to attention is relative xyz plus the time lag.
    """

    def __init__(self, dim, num_sweeps, window_size, shift, heads=4,
                 capacity_per_frame=None, neighbors_per_frame=None,
                 query_chunk_size=32, key_chunk_size=64):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift
        self.heads = heads
        self.head_dim = dim // heads
        self.num_sweeps = num_sweeps
        self.query_chunk_size = query_chunk_size
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.position_encoding = nn.Sequential(
            nn.Linear(4, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim)
        )
        # Predict a small residual motion in the aligned ego frame.  Zero
        # initialization preserves the fixed-pose baseline at step zero.
        self.motion = nn.Sequential(nn.Linear(4, dim // 4), nn.GELU(), nn.Linear(dim // 4, 3))
        self.output = nn.Linear(dim, dim, bias=False)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.motion[-1].weight)
        nn.init.zeros_(self.motion[-1].bias)

    def _window_keys(self, coord):
        window = torch.floor((coord[:, :2] + self.shift) / self.window_size).to(torch.int64)
        shifted = window + 8192
        return shifted[:, 0] * 16384 + shifted[:, 1]

    def _attend_packed(self, q_coord, q_feature, h_coord, h_feature, h_time,
                       q_windows, h_windows):
        """Exact all-point cross attention using one FlashAttention varlen kernel."""
        q_order = torch.argsort(q_windows)
        h_order = torch.argsort(h_windows)
        q_windows, h_windows = q_windows[q_order], h_windows[h_order]
        q_coord, q_feature = q_coord[q_order], q_feature[q_order]
        h_coord, h_feature, h_time = h_coord[h_order], h_feature[h_order], h_time[h_order]
        q_ids, q_counts = torch.unique_consecutive(q_windows, return_counts=True)
        h_ids, h_counts = torch.unique_consecutive(h_windows, return_counts=True)
        qi = torch.searchsorted(q_ids, h_ids)
        valid = (qi < len(q_ids)) & (q_ids[qi.clamp_max(len(q_ids)-1)] == h_ids)
        qi, hi = qi[valid], torch.nonzero(valid, as_tuple=False).flatten()
        common = h_ids[valid]
        if len(common) == 0:
            return torch.zeros_like(q_feature), q_order
        q_starts = torch.cumsum(q_counts[qi], 0) - q_counts[qi]
        h_starts = torch.cumsum(h_counts[hi], 0) - h_counts[hi]
        q_idx = torch.cat([torch.arange(s, s+n, device=q_coord.device) for s, n in zip(q_starts.tolist(), q_counts[qi].tolist())])
        h_idx = torch.cat([torch.arange(s, s+n, device=h_coord.device) for s, n in zip(h_starts.tolist(), h_counts[hi].tolist())])
        qc, qf = q_coord[q_idx], q_feature[q_idx]
        hc, hf, ht = h_coord[h_idx], h_feature[h_idx], h_time[h_idx]
        q_pe = self.position_encoding(torch.cat([qc / self.window_size, torch.zeros_like(qc[:, :1])], 1))
        h_pe = self.position_encoding(torch.cat([hc / self.window_size, ht[:, None]], 1))
        q = (self.query(qf) + q_pe).reshape(-1, self.heads, self.head_dim)
        k = (self.key(hf) + h_pe).reshape(-1, self.heads, self.head_dim)
        v = self.value(hf).reshape(-1, self.heads, self.head_dim)
        # FlashAttention kernels require fp16/bf16; restore the model dtype
        # immediately after the kernel so the surrounding PTv3 path is unchanged.
        attention_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        q, k, v = q.to(attention_dtype), k.to(attention_dtype), v.to(attention_dtype)
        cu_q = torch.cat([torch.zeros(1, device=qc.device, dtype=torch.int32), torch.cumsum(q_counts[qi], 0).to(torch.int32)])
        cu_k = torch.cat([torch.zeros(1, device=hc.device, dtype=torch.int32), torch.cumsum(h_counts[hi], 0).to(torch.int32)])
        if flash_attn_varlen_func is not None:
            out = flash_attn_varlen_func(
                q, k, v, cu_q, cu_k, int(q_counts[qi].max()),
                int(h_counts[hi].max()), 0.0, softmax_scale=None, causal=False
            )
            out = out.reshape(-1, self.dim)
        else:
            # Mathematically identical fallback for environments without the
            # optional flash-attn wheel.  It keeps every point in each local
            # window and chunks only the queries, so no history is truncated.
            # Process one variable-size window at a time and chunk its
            # queries.  This invokes PyTorch's fused SDPA kernel on GPUs
            # without flash-attn while retaining the exact all-key softmax;
            # the chunking is only a memory bound and never drops history.
            # The fused flash kernel internally accumulates in fp32.  Mirror
            # that behavior here because bf16 SDPA can overflow on the first
            # randomly initialized detection pass.
            q_work, k_work, v_work = q.float(), k.float(), v.float()
            q_counts_list = q_counts[qi].tolist()
            h_counts_list = h_counts[hi].tolist()
            outputs = []
            q_offset = h_offset = 0
            for nq, nk in zip(q_counts_list, h_counts_list):
                k_win = k_work[h_offset:h_offset + nk].transpose(0, 1).unsqueeze(0)
                v_win = v_work[h_offset:h_offset + nk].transpose(0, 1).unsqueeze(0)
                win_parts = []
                for qs in range(0, nq, self.query_chunk_size):
                    q_chunk = q_work[q_offset + qs:q_offset + min(qs + self.query_chunk_size, nq)]
                    q_win = q_chunk.transpose(0, 1).unsqueeze(0)
                    ah = F.scaled_dot_product_attention(q_win, k_win, v_win)
                    win_parts.append(ah.squeeze(0).transpose(0, 1).reshape(-1, self.dim))
                outputs.append(torch.cat(win_parts, dim=0))
                q_offset += nq
                h_offset += nk
            out = torch.cat(outputs, dim=0) if outputs else q.new_zeros((0, self.dim))
        out = self.output(out.to(qf.dtype))
        result = torch.zeros((len(q_feature), self.dim), device=qf.device, dtype=out.dtype)
        result[q_idx] = out
        return result, q_order

    def forward(self, current_coord, current_feature, history_coord, history_feature,
                history_time, history_frame):
        if history_coord.numel() == 0:
            return current_feature
        motion_input = torch.cat(
            [history_coord / self.window_size, history_time[:, None]], dim=1
        )
        history_coord = history_coord + self.motion(motion_input)
        q_windows = self._window_keys(current_coord)
        h_windows = self._window_keys(history_coord)
        temporal_sorted, q_order = self._attend_packed(
            current_coord, current_feature, history_coord, history_feature,
            history_time, q_windows, h_windows
        )
        temporal = torch.zeros_like(current_feature)
        temporal[q_order] = temporal_sorted
        return current_feature + self.residual_scale * temporal


class TemporalPointTransformerSegmentation(nn.Module):
    """Predict current-frame labels using ego-motion-aligned local history.

    PTv3 encodes every ego-aligned sweep with shared weights. Current points
    attend to history points in a regular BEV window, followed by a second
    block shifted by half a window to connect points across window boundaries.
    """

    def __init__(
        self,
        num_classes=17,
        num_sweeps=4,
        feature_dim=64,
        use_temporal_attention=True,
        temporal_window_size=2.5,
        temporal_window_capacity=None,
        temporal_neighbors_per_frame=None,
        temporal_query_chunk_size=32,
    ):
        super().__init__()
        self.num_sweeps = num_sweeps
        self.use_temporal_attention = use_temporal_attention
        self.backbone = PointTransformerV3(
            in_channels=6,
            enable_flash=True,
            upcast_attention=False,
            upcast_softmax=False,
        )
        if use_temporal_attention:
            self.temporal_blocks = nn.ModuleList([
                ShiftedBEVTemporalBlock(
                    feature_dim, num_sweeps, temporal_window_size, shift=0.0,
                    capacity_per_frame=temporal_window_capacity,
                    neighbors_per_frame=temporal_neighbors_per_frame,
                    query_chunk_size=temporal_query_chunk_size,
                ),
                ShiftedBEVTemporalBlock(
                    feature_dim, num_sweeps, temporal_window_size,
                    shift=temporal_window_size / 2,
                    capacity_per_frame=temporal_window_capacity,
                    neighbors_per_frame=temporal_neighbors_per_frame,
                    query_chunk_size=temporal_query_chunk_size,
                ),
            ])
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, num_classes),
        )
        if use_temporal_attention:
            self.temporal_classifier = nn.Linear(feature_dim, num_classes, bias=False)
            nn.init.zeros_(self.temporal_classifier.weight)

    @staticmethod
    def make_ptv3_features(coord, raw_feat):
        """Build six PTv3 input channels from nuScenes point attributes.

        raw_feat is [intensity, ring, time_lag].  Coordinates are normalized
        per axis and appended, so the input channel count matches PTv3.
        """
        coord_scale = coord.new_tensor([50.0, 50.0, 5.0])
        normalized_coord = (coord / coord_scale).clamp(-2.0, 2.0)
        intensity = raw_feat[:, :1]
        ring = raw_feat[:, 1:2] / 32.0
        time_lag = raw_feat[:, 2:3].clamp(0.0, 1.0)
        return torch.cat([intensity, ring, time_lag, normalized_coord], dim=1)

    def forward(self, coord, raw_feat, frame_id, frame_valid=None, grid_size=0.2):
        if frame_id.min().item() != 0 or frame_id.max().item() != self.num_sweeps - 1:
            raise ValueError(f"Expected frame ids 0..{self.num_sweeps - 1}.")

        backbone_input = {
            "coord": coord,
            "feat": self.make_ptv3_features(coord, raw_feat),
            "batch": frame_id,
            "grid_size": grid_size,
        }
        if getattr(self, "baseline_frozen", False):
            with torch.inference_mode():
                point = self.backbone(backbone_input)
        else:
            point = self.backbone(backbone_input)
        point_feat = point.feat
        point_frame = point.batch

        current = point_frame == 0
        baseline_features = point_feat[current]
        if getattr(self, "baseline_frozen", False):
            baseline_features = baseline_features.clone()
        if getattr(self, "baseline_frozen", False):
            with torch.inference_mode():
                baseline_logits = self.classifier(baseline_features)
        else:
            baseline_logits = self.classifier(baseline_features)
        current_features = baseline_features
        if self.use_temporal_attention:
            if frame_valid is None:
                frame_valid = torch.ones(self.num_sweeps, dtype=torch.bool, device=coord.device)
            history = (point_frame != 0) & frame_valid[point_frame]
            for block in self.temporal_blocks:
                current_features = block(
                    coord[current], current_features, coord[history],
                    point_feat[history], raw_feat[history, 2], point_frame[history]
                )
            # Fuse temporal features before the pretrained classifier.  The
            # zero-initialized residual scale keeps initialization identical
            # to the baseline while allowing the adapter to improve features.
            fused_logits = self.classifier(current_features)
            temporal_delta = self.temporal_classifier(current_features - baseline_features)
            return fused_logits + temporal_delta
        return baseline_logits
