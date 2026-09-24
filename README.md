# CenterPoint packed-sweep temporal comparison

这是当前 4-sweep 严格对照实验的代码快照。仓库不包含 nuScenes 数据、数据库采样文件、checkpoint、日志或任何认证信息。

## 实验结构

两组实验使用相同的 CenterPoint 初始化权重、数据顺序、随机种子、优化器和训练轮数：

```text
每个 sweep 独立 voxelization
        -> B*S packed sparse batch
        -> 共享的 VoxelResBackBone8x
        -> HeightCompression -> BaseBEVBackbone -> CenterHead
```

Temporal 组唯一新增的是编码后当前帧 Query 对历史帧 Key/Value 的两层 shifted-window local temporal attention。没有 PTv3 点云编码器、语义分割预训练权重、occupancy mask 或额外 motion compensation。

## 文件说明

- `code/centerpoint_voxel_temporal.py`：CenterPoint 与编码后 temporal attention。
- `code/temporal_mean_vfe.py`：按 sweep 保持帧身份并生成 5D CenterPoint VFE 特征。
- `code/data_processor.py`：`SEPARATE_SWEEPS=True` 的逐帧 voxelization 实现。
- `code/nuscenes_dataset.py`：nuScenes sweep 加载、时间戳和 sweep id 保留。
- `code/temporal_segmentation_model.py`：语义分割版本中复用的 shifted-window attention block。
- `configs/`：matched baseline 与 temporal 配置。
- `tests/`：数据打包和前向/反向 smoke test。
- `integration/registry_changes.txt`：接入 OpenPCDet 注册表所需的两处改动。
- `launch_packed_pair.sh`：先训练 matched baseline，再训练 temporal 并自动评估。

## 运行环境

代码快照是在 OpenPCDet 工作树上运行的增量模块，不是独立可安装的 Python 包。需要先将对应文件放回 OpenPCDet 的原目录，并按 `integration/registry_changes.txt` 注册模块；同时需要安装 OpenPCDet、spconv、nuScenes devkit、PyTorch 和 flash-attn（temporal attention 使用 `flash_attn_varlen_func`）。

启动脚本中的 checkpoint 和 OpenPCDet 路径是服务器实验路径，迁移到其他机器时应修改这些路径。

## 解释结果时的对照关系

原始 aggregate CenterPoint 只能作为外部参考；论文因果对照应使用本仓库生成的 `packed_frame_baseline` 与 `packed_temporal` 两个结果，因为二者的 voxelization 和 packed encoder 路径完全一致。
