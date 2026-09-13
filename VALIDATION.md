# 本机验证记录

下文为历史 mask 阶段记录。当前 nvdiffrast RGB 配对训练的测试、TensorBoard 和产物见 [RGB_TRAINING_VALIDATION.md](RGB_TRAINING_VALIDATION.md)。

验证日期：2026-09-12。工作目录：`E:\Work\pythonWorkplace\instrument-tracking`。

环境：Windows、RTX 4060 Ti 16 GB、Python 3.10、torch 2.12.1+cu126、torchvision 0.27.1+cpu（本项目仅使用其 ResNet）、numpy 2.2.6、trimesh 5.0.0。运行解释器为 `.venv\Scripts\python.exe`。

## 最新：v2 渲染器平滑修复

- 使用不透明 z-buffer 和可见轮廓可微抗锯齿，移除同部件内部三角面的软透明混合；默认宽高各 2 倍超采样，另验证了 4 倍采样。
- 同一原始 mesh、初始化位姿、相机及 320×256 输出下，对比图在 `runs/renderer_inspection_v2/before_after.png` 和 `runs/renderer_inspection_v2_ss4/before_after.png`。展示的是实际用于损失的浮点 soft mask，无额外模糊或补洞。
- 腕部内部（以新版 >0.5 mask 向内腐蚀 3 像素定义，共 1873 像素）的覆盖率均值由 0.9442 变为 1，标准差由 0.0638 变为 0；杆身内部也恢复到均匀覆盖。该指标衡量内部伪影，不衡量与真实位姿的对齐精度。具体数据在 `runs/renderer_inspection_v2/quality_metrics.json`。
- 全部测试 **10 passed、1 skipped**。新增检查：不同三角剖分的平面输出一致；共享三角边没有暗纹；轮廓梯度与有限差分一致；1/4 倍采样保持同一相机投影；边界保留软像素。
- 真实器械 mesh 的 12 维位姿编码均收到有限梯度。新生成 8 个样本，完成两阶段 BF16 smoke 训练（2 个 batch、每批 2 个样本），实际参数更新 2 次，AMP 跳过 0 次。数据位于 `data/synthetic_renderer_v2_smoke`，训练日志与权重位于 `runs/renderer_v2_smoke`。这些权重仍仅用于流程验证。
- 新数据含 `renderer_config.version=2` 和采样参数；训练拒绝旧版或不匹配的渲染数据。正式实验需要重新生成数据并训练，不能把旧版标签继续与新版投影混用。
- nvdiffrast 本机检查仍因依赖缺失跳过。下文 5.18 FPS 为旧版后端结果，不代表 v2 或高速后端性能。

## v1 历史验证

- `python -m pytest -q`：**7 passed，1 skipped**。覆盖 SO(3)/关节限位、参考齐次变换运动学、近裁剪面、三角面遮挡可见性、mask 梯度方向、端点退化情况/匹配及真实样例读取。跳过项为 nvdiffrast CUDA 检查。
- 三个命令行入口和全部项目 Python 模块编译通过。
- 使用提供的真实 mesh 生成 **24 个 64×80 合成样本**，其中 22 个训练、2 个验证；位姿、相机及可见端点标签已写出。
- 两阶段预测、可微三角 mesh 渲染、全部损失和反向传播完成 **2 个 smoke-test epoch，每个 3 个 batch，每 batch 2 个样本**；BF16 AMP 实际更新参数 6 次，溢出跳过 0 次。合成验证损失从 8.9867 降到 8.2777，这仅表明小规模优化链路工作，不能据此判断模型已收敛。
- `last.pt` 模型、优化器、scheduler、随机状态及 AMP 状态可恢复；加载完成后无剩余 epoch，正常结束。
- 真实样例前 **5 帧**按时间顺序推理成功，保存合法位姿、投影 mask、叠加图和统计。

## v1 历史验证产物

| 路径 | 内容 |
| --- | --- |
| `data/synthetic_smoke_final/` | 最终小规模合成数据和元数据 |
| `runs/smoke_final/history.jsonl` | 最终训练/验证日志 |
| `runs/smoke_final/best.pt` | 流程验证权重，明确标记 `smoke_test=true` |
| `runs/smoke_final/inference/poses.jsonl` | 五帧位姿预测 |
| `runs/smoke_final/inference/metrics.json` | 最终推理统计 |
| `runs/smoke_final/inference/overlays/` | 五帧叠加图 |
| `runs/smoke_final/mask_only_check/` | 单次预测、mask-only、跳过最终渲染路径检查 |

早期调试产生的 `data/synthetic_smoke`、`data/synthetic_verified`、`runs/smoke`、`runs/verified_smoke`、`runs/verified_inference` 及根目录 `render_*.png` 为中间结果。清理操作被自动审批策略拦截，已原样保留；后续训练请使用自己新生成的数据或上述 `*_final` 产物。

## 已测速度及未验证项

最终五帧 smoke 测试配置：torch 渲染后端、64×80、BF16 网络、两次预测、包含最终渲染、2 帧 warmup。

| 指标 | 结果 |
| --- | --- |
| 预测管线平均延迟 | 192.88 ms |
| p95 延迟 | 194.76 ms |
| 预测管线速度 | 5.18 FPS |
| 含文件处理/结果保存的吞吐 | 4.84 FPS |

这是少量帧上的流程测速，且使用低分辨率及未充分训练的权重，**没有达到实时要求**。不能推广到 256×320 或正式模型。当前机器未安装 nvdiffrast 所需工具链，高速后端的运行、梯度和速度尚未验证，不能承诺 30 FPS。正式部署需安装该后端、重新生成同后端数据、训练完整模型，再实测完整视频管线。

没有执行大规模训练，没有真实逐帧 3D 位姿真值，因此尚无真实位姿精度结论。样例 `transforms.json` 仅作为相机/初始姿态来源，未充当监督真值。
