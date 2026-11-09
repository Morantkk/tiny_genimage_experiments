# Tiny-GenImage: AIGC 图像检测与生成器溯源

基于 ResNet18、频域分支、Patch-Transformer 与传统手工特征融合的 AI 生成图像检测实验代码。

任务分两个层次同时评估：

- **多分类**：在 `Real` + 8 类生成器共 9 个类别上做生成器溯源；
- **二分类**：把 `Real` 记为真、其余全部生成器记为假，做真假鉴别。

---

## 模型

| 实验名 | 结构要点 |
|---|---|
| `resnet18_baseline` | ResNet18，替换 fc 层作为基线 |
| `resnet18_frequency` | ResNet18 + FFT 幅度谱分支（抑制中心低频后接小型 CNN） |
| `patch_transformer` | 图像切 g×g 块，共享 ResNet18 编码为 token，Transformer 编码后 mean-pool |
| `fusion_model` | 全局 ResNet18 + 切块 Transformer + 频域分支三路拼接 |
| `patch_transformer_optimized` | 切块版本改进：加入 CLS token 与全局 token，可调 patch grid / 层数 |
| `resnet18_traditional_*` | ResNet18 + 手工特征，支持 `concat` / `gated` / `film` / `hand_only` 四种融合 |

### 手工特征组

`train_resnet18_traditional_fusion.py` 中定义了 5 组特征，用于控制变量筛选：

| 组 | 内容 |
|---|---|
| `first20` | 灰度均值/方差、拉普拉斯方差、FFT 平均能量 + 16 bin 灰度直方图 |
| `color` | RGB 各通道均值/标准差/偏度 + 12 bin HSV 直方图 |
| `texture` | 16 bin LBP 直方图、Sobel 统计量、拉普拉斯绝对均值 |
| `frequency` | 8 bin FFT 径向能量、低/中/高频能量、高频低频比 |
| `all` | 以上四组全部拼接 |

---

## 环境

```bash
pip install -r requirements.txt
```

代码在 PyTorch 2.x + CUDA 上验证。多卡训练依赖 `torch.distributed`，需要 NCCL。

## 数据准备

需要 [Tiny-GenImage](https://github.com/Yunan-98/Tiny-GenImage) 的 parquet 分片，目录结构：

```
data/Tiny-GenImage/
└── data/
    ├── train-*.parquet
    └── validation-*.parquet
```

每个 parquet 需要 `image`（含 `bytes` 字段）和 `generator`（整数标签）两列。标签顺序固定为：

```
Real, ADM, BigGAN, GLIDE, Midjourney, SD14, SD15, VQDM, Wukong
```

首次运行会把 parquet 解包成 jpg 并写入 `cache/`，之后的运行直接复用。

## 运行

### 单机单卡

```bash
python train_resnet18_baseline.py
```

### 多卡（DDP）

```bash
python -m torch.distributed.run --nproc_per_node=8 train_resnet18_baseline.py
```

### 路径参数

所有入口都支持覆盖默认路径：

| 参数 | 默认值 |
|---|---|
| `--data-root` | `./data/Tiny-GenImage` |
| `--cache-root` | `./cache/Tiny-GenImage` |
| `--output-root` | `./outputs` |

### 批量脚本

```bash
bash run_all_experiments.sh                    # 基础三模型对比
bash run_fusion_experiment.sh                  # 三路融合
bash run_traditional_feature_experiments.sh    # 手工特征筛选 + 融合方式对比
bash run_patch_optimization_experiments.sh     # 切块模块消融
```

脚本默认调用 `python`，可通过环境变量指定解释器：

```bash
PYTHON=/path/to/python bash run_all_experiments.sh
```

> **注意**：`run_traditional_feature_experiments.sh` 会先用 `--prepare-only` 单进程预计算手工特征缓存。手工特征提取是 CPU 密集型操作，若在 DDP worker 内进行容易触发 NCCL 超时，请不要跳过这一步。

## 输出

每次实验在 `outputs/<experiment_name>/` 下产生：

```
outputs/<experiment_name>/
├── config.json                              # 本次运行的完整参数
├── history.json                             # 逐 epoch 指标
├── best_model.pth                           # 按 macro-F1 选出的最佳权重
└── best_eval/
    ├── metrics.json
    ├── classification_report.json
    ├── confusion_matrix_multiclass.csv
    ├── confusion_matrix_binary.csv
    └── predictions.csv
```

### 汇总与可视化

```bash
python compare_all_models.py             # 四模型指标汇总
python visualize_results.py              # 柱图、训练曲线、混淆矩阵、每类 F1
python compare_traditional_fusion.py     # 手工特征实验对比
python compare_patch_optimization.py     # 切块/频域融合对比
```

图片输出到 `outputs/figures/` 及各自的 `*_analysis/` 目录。

## 结果

> 以下图表由 `visualize_results.py` 与各 `compare_*.py` 生成，将在补充实验记录后填入。

<!-- TODO: 填入 outputs/figures/model_metric_comparison.png -->
<!-- TODO: 填入混淆矩阵与每类 F1 对比图 -->

---

## 已知限制

- 所有骨干网络均为 `weights=None`，从零训练。在有 ImageNet 预训练的情况下精度通常明显更高，这部分未做对照实验。
- `patch_transformer` 中每个 patch 会被 `F.interpolate` 放大回原分辨率，g 较大时插值会模糊掉原始高频细节。
- DDP 模式下 `common.evaluate` 的 `val_loss` 只反映当前 rank 的局部均值，仅作日志参考，模型选择依据是 macro-F1。
- 仅支持 ImageNet 归一化参数，未适配其它预训练权重。

## 引用

如果这个项目对你有帮助：

```bibtex
@misc{chen2026tinygenimage,
  title  = {Tiny-GenImage: AIGC Image Detection and Generator Attribution},
  author = {Chen, Fakun},
  year   = {2026},
  url    = {https://github.com/Morantkk/tiny_genimage_experiments}
}
```

## 许可

代码以 [MIT License](LICENSE) 发布。

**数据集许可是独立的。** Tiny-GenImage 由第三方发布，其许可条款不受本项目覆盖；数据集中的生成图像还可能受各生成模型自身使用条款的约束。使用前请自行确认。本项目不附带任何数据。

## 致谢

数据集来自 [Tiny-GenImage](https://github.com/Yunan-98/Tiny-GenImage)。骨干网络来自 `torchvision`。
