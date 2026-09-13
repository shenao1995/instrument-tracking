# RGB 配对训练验证记录

日期：2026-09-12。路径：`E:\Work\pythonWorkplace\instrument-tracking`。

## 已完成

- `python -m pytest -q`：**18 passed**。包含 RGB loss 的前景归一化/空图处理、仅四项损失、Dice 缺失部件处理、两列 batch 图布局、材质渲染与旧 mask 几何一致性，以及仅靠 RGB loss 向两阶段位姿头和 ResNet 第一层反传的检查。
- 所有根目录 Python 文件编译检查通过。
- 使用原始 OBJ、MTL 和 nvdiffrast RGB 配置 v4 生成 8 个 128×160 样本（6 训练、2 验证），数据为 `data/synthetic_rgb_smoke/`。颜色由渲染器产生，没有用随机 mask 着色冒充纹理。
- 运行 4 轮 smoke 训练，每轮限制 1 个 batch、batch size 2、两阶段、BF16 AMP，实际优化器更新 4 次、溢出跳过 0 次。
- 仅第 2、4 轮出现验证记录；每个 train/val batch 都打印总损失和 RGB、mask、tips、angle 四个分项。
- 两轮验证均打印三部件 Dice，并按较低的验证总损失保存 best；第 4 轮验证损失为 4.580853。该值仅验证流程，不表示正式精度。
- 实际读取 TensorBoard 事件文件：全部验证 loss/Dice 的 step 为 `[2,4]`；训练 batch loss 的 step 为 `[1,2,3,4]`。
- overlap 和 RGB 对照图均在 step 2、4 存在，尺寸为 320×256，对应每行两幅 160×128 图、两行样本；不是只保存第一个样本。
- checkpoint 第一层输入通道数为 6，`input_mode=rgb_pair`。恢复已完成的 `last.pt` 成功，模型/优化器/scheduler/AMP/随机状态可读取。
- 新 RGB checkpoint 在真实样例前 5 帧完成两阶段推理，保存位姿、mask 和叠加图。真实样例没有参与合成训练或验证。

## 产物

| 路径 | 内容 |
| --- | --- |
| `runs/pytest_rgb_training.log` | 18 项测试结果 |
| `runs/rgb_training_smoke.log` | 每 batch 五项损失、第 2/4 轮验证、Dice 和 best 保存记录 |
| `runs/rgb_training_smoke/history.jsonl` | 完整 epoch 记录；未验证轮次 `val=null` |
| `runs/rgb_training_smoke/tensorboard/` | TensorBoard events |
| `runs/rgb_training_smoke/tensorboard_verification.json` | 从事件文件读出的标量/图像 step 和图像尺寸 |
| `runs/rgb_training_smoke/val_images/` | 与 TensorBoard 同内容的 PNG |
| `runs/rgb_training_smoke/best.pt`、`last.pt` | 明确标记为 smoke test 的流程验证权重 |
| `runs/rgb_training_smoke/inference/` | 5 帧真实样例的推理输出 |

```powershell
& .\.venv\Scripts\python.exe -m tensorboard.main --logdir runs/rgb_training_smoke/tensorboard --port 6006
```

## 使用限制

当前提供的 MTL 没有图像纹理贴图，RGB 来自 CAD 材质色与平滑光照。真实内镜的高光、组织反射、曝光和遮挡可能与渲染外观不同；没有以这次小数据实验声称真实 IoU 或三维位姿精度得到提升。

新训练需要渲染一致的 RGB 数据，旧随机颜色 RGB 数据被明确拒绝。旧 9 通道 checkpoint 可以继续在推理入口走兼容路径，但不能恢复为新 6 通道模型训练。原有 `runs/exp1`、`runs/pose_exp1` 和 `data/synthetic_pose_v1` 未被本次验证覆盖。
