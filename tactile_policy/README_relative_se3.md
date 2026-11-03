# 采集数据
```bash
python real_ur5rg2/collect_lerobot_xenstactile_contactline.py   --root real_ur5rg2/data/nut_relative_ou_no_fts   --num-episodes 100   --fps 20   --ou-frames 800   --no-collect-force   --no-force-safety --no-preview

```
# 成对触觉 SE(3) 模型

`train_relative_se3.py` 使用 `--backbone` 选择特征提取网络，默认 `spatial`。
所有模型接收 `[B,2,6,H,W]`，两时刻依次为参考帧、当前帧；每帧通道顺序为
`[Lx,Ly,Lz,Rx,Ry,Rz]`。预处理、数据划分、六轴归一化 MSE 和 SE(3) 标签约定相同。

| backbone | 特征提取方式 | 输入适配 |
| --- | --- | --- |
| `spatial` | 原 `PairwiseTactileSE3Regressor`，四张单手图共享空间编码器，拼接四个 64 维向量 | 保留原生网格 |
| `fcn` | 全连接网络，保留展平后的网格位置 | 插值到 `grid-size × grid-size`，默认 20×20 |
| `simple_cnn` | 四层卷积与池化 | 插值到 `image-size × image-size`，默认 128×128 |
| `posenet_cnn` | 五层卷积与池化 | 同上 |
| `nature_cnn` | 三层卷积 | 同上 |
| `resnet` | 四阶段残差卷积与自适应平均池化 | 同上 |
| `vit` | patch 嵌入与 Transformer | 同上；`image-size` 必须是 32 的正整数倍 |
| `tactile_tokenizer` | 双指空间编码、位置编码与 Transformer | 35×20 或 36×20 保持原尺寸，其余尺寸按原工厂逻辑插值到 36×20 |

新增网络对应 `PairwiseBackboneSE3Regressor`。每个时刻的左右六通道一起输入
`create_model` 创建的共享编码器，其输出层投影为 `feature-dim` 维特征，默认 128。
参考、当前特征按顺序拼接后，经 `512 → 512 → 256 → 6` 的回归头与 Tanh 输出
归一化相对位姿。两个时刻沿 batch 维一起编码，FCN 在训练 batch 为 1 时也有
两个编码样本。原 `spatial` 模型的结构、参数名称和默认行为保持原样。

```bash
# 原模型
python tactile_policy/train_relative_se3.py --backbone spatial

# ResNet 特征提取
python tactile_policy/train_relative_se3.py --backbone resnet --patience 20

# 换用其他骨干，例如 ViT
python tactile_policy/train_relative_se3.py --backbone vit --patience 20
```

未指定 `--output-dir` 时，`spatial` 沿用
`tactile_policy/outputs/relative_se3_200_217`；新增骨干分别写入该路径加
`_fcn`、`_resnet`、`_vit` 等后缀的目录。同一骨干的不同超参数实验应通过
`--output-dir` 指定独立目录。`--feature-dim` 只作用于新增模型，`--grid-size`
只作用于 FCN，`--image-size` 作用于 CNN、ResNet 和 ViT。

`best_model.pt` 保存 `model_type`、`model_config`、最佳轮次、归一化和训练参数。
接触线推理与 `compare_relative_se3.py` 会自动重建相应骨干。旧 checkpoint
无 `model_config` 时仍按原 `PairwiseTactileSE3Regressor` 严格加载。

```bash
python tactile_policy/infer_tactile_servo_nutbolt_contact_line.py \
  --mode contact-line-probe \
  --checkpoint tactile_policy/outputs/relative_se3_200_217_resnet/best_model.pt \
  --no-contact-line-auto-prepare \
  --contact-line-contact-established

python tactile_policy/compare_relative_se3.py \
  --dataset-root "/media/zhou/Elements SE/YBZHOU/nut_relative_ou_217" \
  --checkpoint tactile_policy/outputs/relative_se3_200_217/best_model.pt \
               tactile_policy/outputs/relative_se3_200_217_resnet/best_model.pt \
  --names spatial resnet \
  --output-dir tactile_policy/outputs/compare_spatial_resnet
```

比较时使用同一数据集、seed、验证比例与帧对清单。骨干实验同时改变了单手/双指
特征融合方式和输入尺寸，应结合真实单位的逐轴误差判断效果。

比较脚本默认选择 `relative_se3_200_217` 与 `relative_se3_200_217_resnet`。
若比较其他模型，用 `--checkpoint` 显式传入实际权重。未传 `--dataset-root` 时，
脚本使用各 checkpoint 内相同的训练数据集路径；若训练路径不同或数据迁移，
需要显式传入该参数。启动时会打印当前数据集 episode 数量及各权重训练时的数量。

新训练权重的 `dataset_info` 会保存实际 `train_episode_ids` 和 `val_episode_ids`。
比较时优先取各权重已保存验证编号的交集，在原 episode 编号保持不变的数据追加后
也能确定原验证 episode。旧权重只保存数量、seed 和比例，必须使用训练时原始数据集
重建划分；不能在不同数量的数据上按相同 seed 重划分，否则可能把训练 episode
当作验证 episode。

如果提示 `Dataset episode count differs from training`，先核对 `--dataset-root`
与 `--checkpoint` 是否指向同一次实验。例如，不要将 `relative_se3_100/200`
旧权重与 `nut_relative_ou_217` 扩充数据集混用。数据集变化后应选择新的
`--output-dir`，原 `validation_pairs.json` 的数据指纹仍会严格校验。
