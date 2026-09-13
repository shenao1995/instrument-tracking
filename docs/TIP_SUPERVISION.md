# 端点监督更新

本轮保持 ResNet-34、6 通道 RGB 配对、FK、nvdiffrast 和 mask/RGB loss 实现不变；不调整网络初始开合角。移除 `angle_loss`，保留网络中的角度硬约束。

## 损失与指标

```text
L = 0.1 * Lrgb + 1 * Lmask + 1 * Ltips
Ltips = 1 * Lposition + 1 * Lgap
Lgap = SmoothL1((distance(predicted tips) - distance(GT tips)) / GT_scale)
GT_scale = max(10 pixels, GT gripper bbox diagonal)
```

`--rgb-weight / --mask-weight / --tips-weight / --tips-position-weight / --tips-gap-weight` 可调整上述系数。默认值是实验起点，不保证最优。位置与间距都用 Smooth L1（beta=0.05），位置支持交换左右标签。归一化尺度来自 GT，以像素为单位；GT 间距为零不会导致除零。

保留位置项，因为单独的间距不约束平移与方向。二维间距也受深度、视角影响，不能独立证明三维开合角准确。预测端点直接由 FK 投影得到，不从阈值化预测 mask 中提取。精确重合的预测端点处，欧氏范数的零点梯度不能单独打开夹爪，因此保留位置项和原来的非零开合初始化；测试覆盖近闭合状态的实际 FK 开合梯度。

打印与 TensorBoard：`loss/rgb/mask/tips/tips_position/tips_gap`、`tip_pair_ratio`、`tip_gap_error_px`。损失分项按与总损失相同的阶段权重平均；端点比例和像素 MAE 在最终预测阶段计算。整个 epoch 的 MAE 累加有效端点对的误差再除以有效对数，不平均各 batch 的条件均值。没有有效对时 MAE 为 N/A（JSON null），不写一个误导性的零标量。

## 标签与旧数据兼容

合成数据仍用原始几何端点作为 GT 坐标，避免把 mesh 末端中心和 mask 外轮廓末端混为一谈。GT mask 提取用于核查与置信度修正；真实数据没有几何 GT 时使用 mask 伪标签。

mask 提取流程：裁剪腕部/夹爪区域，过滤小连通域，从腕部连接带执行多源测地距离搜索，在多个近端截断比例中寻找两个足够长的远端分支，对末端小连通区域求中心。孤立噪声、短支刺、边缘截断和无法分开的单夹爪不会强行产生可靠端点对。

几何标签检查：端点必须有限、在画面内，且距可见夹爪 mask 不超过 2 像素。原来无效的端点，仅在两个 mask 远端能与两个几何端点完成一一匹配、各误差均不超过 3 像素时恢复；恢复置信度为 0.5，原来有效的标签保留其置信度。不把真实遮挡端点统一置为有效。

训练自动生成 `data/synthetic_pose_v1/tip_labels_v2.npz` 与同名 JSON 检查报告。缓存绑定 metadata、manifest、样本大小/修改时间和标签算法源码，数据或算法变化会重算。**不改写源 NPZ、RGB、mask，也不重新渲染 OBJ。** 第一次扫描约需数分钟，后续复用缓存。检查本身不依赖 GPU。可单独生成检查图：

```powershell
Set-Location E:\Work\pythonWorkplace\instrument-tracking
& .\.venv\Scripts\python.exe -m utils.pose_tip_labels --data data/synthetic_pose_v1 --preview data/synthetic_pose_v1/tip_label_audit.png
```

## 输入尺寸与启动

本机 `data/synthetic_pose_v1/metadata.json` 的 `image_size=[512,640]`，即高 512、宽 640。目标和 moving 分别为 `[B,3,512,640]`，在通道维拼接成 `[B,6,512,640]`；batch size 8 时为 `[8,6,512,640]`。没有统一 resize 到 224。新生成数据的 CLI 默认尺寸仍为高 256、宽 320，若生成 512×640 数据，需显式传 `--height 512 --width 640`。

沿用停止训练前的权重，开启新 loss 实验：

```powershell
& .\.venv\Scripts\python.exe training_pose.py --data data/synthetic_pose_v1 --output runs/pose_tips_v2 --init-from runs/pose_exp1/best.pt --epochs 200 --batch-size 8 --steps 2 --val-every 2 --rgb-weight 0.1 --mask-weight 1 --tips-weight 1 --tips-position-weight 1 --tips-gap-weight 1
& .\.venv\Scripts\python.exe -m tensorboard.main --logdir runs/pose_tips_v2/tensorboard --port 6006
```

`--init-from` 只加载模型参数，重新初始化优化器、scheduler、epoch 和最佳评分；也可省略它从头训练。新旧 loss 定义不同，旧实验不能直接 `--resume`。当前版本实验断点续训则使用 `--resume`，保持总 epochs、权重和验证策略一致。

每次验证仍保存首个 batch 的 B 行、两列 overlap：GT 左，预测右。GT 端点/连线为黄色，预测为粉色，无效 GT 为灰色叉；右列同时保留 GT 作为比较，并显示间距误差。原始 RGB 对照图不画端点。检查图与短训练只能验证代码、标签与梯度；收敛改善需完整训练后比较夹爪 Dice、有效对覆盖率及端点间距 MAE。

## 本机验证（2026-09-12）

- `python -m pytest -q`：27 passed，包含 nvdiffrast/CUDA 梯度测试。新增用例覆盖不等长夹爪、噪声、边界截断、标签恢复条件、缺失/非有限标签、尺度和左右交换不变性、真实 FK 的开合梯度、缓存失效及图像布局。
- 现有 5,000 样本完成缓存：训练集有效端点对比例从 18.675% 到 76.325%，验证集从 19.8% 到 73.1%；分别恢复 3,295、754 个端点，恢复置信度 0.5。这个比例表示算法接受的监督覆盖率，不是人工标注准确率。
- `runs/tips_v2_smoke_20260912`：从 `runs/pose_exp1/best.pt` 初始化，2 epochs、每轮 2 train batches，第 2 轮 2 val batches，batch size 8、512×640、2 stages、未启用 AMP。4 次优化器更新全部完成；正确生成 `last.pt`、`best.pt`、history、TensorBoard 和两类验证图。此目录明确标记为 smoke test，不作为性能结论。
- 重新读取 TensorBoard 事件，确认 3 类 scalar 分组都有位置/间距/有效比例/像素误差，没有 angle；验证 step 仅为 2，两个图像标签均为高 4096、宽 1280，即 8 行、2 列。检查结果保存在该运行目录的 `tensorboard_verification.json`。
