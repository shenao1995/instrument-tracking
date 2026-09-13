# 器械位姿预测的 OBJ 渲染器选型

本文记录 v3 mask 阶段的渲染器选型。当前 RGB 配对训练已增加材质渲染及 RGB loss，并按用户要求取消 pose 真值监督；最新损失、训练命令和验证以 [README.md](../README.md) 与 [RGB_TRAINING_VALIDATION.md](RGB_TRAINING_VALIDATION.md) 为准。

调研与实现日期：2026-09-12。目标是保留 `mask loss → 投影 → FK → 位姿网络` 的梯度，并提高 NVIDIA GPU 上多标签 mask 的渲染速度。

## 结论

本项目选择 **NVIDIA nvdiffrast 的 CUDA 后端，搭配可微抗锯齿和默认 2× 超采样**。默认保留提供的原始 OBJ 网格，先做重复顶点/重复面/退化面清理，不再默认简化成约一万个三角面。自写 torch 后端保留为显式选择的参考实现，训练、推理、数据生成的默认路径均已切换。

这是结合本项目硬件、mask 输出和梯度要求作出的工程选择。没有在同一环境安装并实测所有候选库，因此不声称 nvdiffrast 对所有场景都“全局最快”。实测范围及可复现实验见 [VALIDATION.md](VALIDATION.md) 和 `scripts/benchmark_renderer.py`。

## 论文及官方实现实际采用了什么

| 工作 | 渲染方式和依据 | 对本项目的意义 |
| --- | --- | --- |
| FoundationPose，CVPR 2024 | [论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Wen_FoundationPose_Unified_6D_Pose_Estimation_and_Tracking_of_Novel_Objects_CVPR_2024_paper.pdf)；[官方 Utils.py](https://github.com/NVlabs/FoundationPose/blob/main/Utils.py) 的 `nvdiffrast_render` 使用 nvdiffrast 栅格化、纹理/颜色插值 | 说明该库已用于实际的姿态估计/追踪系统。不过该封装明确说明不支持梯度，也没有调用 `dr.antialias`，不能直接照搬它来监督 silhouette 位姿梯度 |
| Self6D，ECCV 2020 | [论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460103.pdf) 第 3 节采用 DIB-R；[官方渲染代码](https://github.com/THU-DA-6D-Pose-Group/Self6D-Diff-Renderer) 扩展 Kaolin 中的 DIB-R，支持深度等输出 | 通过渲染监督位姿与本项目需求接近。但该发布版依赖旧 Kaolin/PyTorch 生态，在当前 Windows 环境不是首选集成路径 |
| MegaPose，CoRL 2022 | [官方项目](https://github.com/megapose6d/megapose6d)；[场景渲染实现](https://github.com/megapose6d/megapose6d/blob/master/src/megapose/panda3d_renderer/panda3d_scene_renderer.py) 使用 Panda3D | 是常规 render-and-compare 网络路线。该渲染输出没有本任务所需的 PyTorch mask-to-pose 自动微分链路 |
| nvdiffrast，SIGGRAPH Asia 2020 / TOG | [论文](https://arxiv.org/abs/2011.03277)；[官方文档](https://nvlabs.github.io/nvdiffrast/) | CUDA 加速栅格化、插值和抗锯齿；可以只计算本项目需要的语义通道，不必计算纹理、光照或复杂材质 |
| PyTorch3D，相机优化示例 | [官方教程](https://pytorch3d.org/tutorials/camera_position_optimization_with_differentiable_rendering) 加载 OBJ，以 `SoftSilhouetteShader` 优化相机 | 可行的备选，提供更完整的相机/mesh/shader 组合。教程的软轮廓保存每像素多个候选面；内存和速度需按自己的 `faces_per_pixel` 等配置测量，不能直接与此项目测速比较 |

## 为什么 MeshLab 看起来光滑

OBJ 是顶点、面、法线、纹理等信息的文件格式，本身不决定采用哪种栅格化算法。保留 OBJ 并更换渲染实现是可行的；nvdiffrast 读取的是从 OBJ 解析出的顶点和三角面张量。

MeshLab 的受光表面可以通过法线插值呈现光滑外观，但语义 mask 是部件的像素覆盖率，不应该包含这种光照变化。此前腕部内部三角面暗纹来自自写渲染器的面覆盖混合；新版使用不透明深度缓冲和恒定部件属性，内部同标签共享边不会因光照或三角面编号变暗。轮廓还受原始几何、简化程度和输出分辨率影响，故默认保留原始网格并进行抗锯齿。

不采用图像高斯模糊、形态学闭运算或人为补洞来美化 mask；这些操作可能改变细小夹爪、真实缝隙和测量所需的轮廓。

## 当前可微链路

```text
ResNet-34 → 受约束位姿 → Instrument-Splatting 正向运动学
          → 相机/裁剪坐标 → nvdiffrast CUDA 栅格化（不透明深度遮挡）
          → 3 通道语义插值 → dr.antialias → 面积下采样 → soft mask loss
```

杆身、腕部、左右夹爪的语义通道为 `[shaft, wrist, grippers]`；夹爪仍分别执行运动学，只在 mask 标签中合并。输出前景内部覆盖率为 1，边界可为 0–1；背景在三个通道均为 0。

根据 [nvdiffrast 的栅格化与抗锯齿说明](https://nvlabs.github.io/nvdiffrast/)，点采样本身不能提供覆盖率/可见性变化的梯度，需要后续的 `dr.antialias`。实现缓存 CUDA context、int32 面索引及抗锯齿拓扑哈希；语义属性按 batch 广播，关闭不需要的重心坐标屏幕导数梯度。原有 FK、端点投影及相机像素中心约定保持一致。

抗锯齿提供局部可见边界的梯度，不代表遮挡拓扑变成全局连续函数。预测与目标完全不相交时，mask 的定位信号仍可能不足，所以保留合成 pose 真值监督、端点损失和多阶段监督。

本机验证还发现 0.4.0 在批量 AA 反传时会混合不同实例的梯度：[固定版本的 CUDA 源码](https://github.com/NVlabs/nvdiffrast/blob/253ac4fcea7de5f396371124af597e6cc957bfae/csrc/common/antialias.cu) 中，`AntialiasGradKernel` 的合并原子操作分组键只使用三角面和边，没有实例索引。代码检查结合“同一位姿单独渲染与放进 batch 的梯度不一致”实验支持这一定位。适配层在训练时逐样本调用原生 AA，栅格化和插值仍批量执行；推理及数据生成无需反传，继续使用批量 AA。没有修改或替换 NVIDIA 的 CUDA 源码。测试比较批量/单样本梯度，并在远离角点的直边上做有限差分检查；不要求离散可见性切换处有全局光滑导数。

## loss 的变化范围

本次替换没有随意改动 loss 公式或权重；改变的是输入 mask 的渲染质量及对应的梯度计算。之前已经实现的改进继续使用：

- 三尺度 soft Dice + 部件平衡 BCE，避免预测端硬阈值切断梯度。
- 合成端点来自原始 mesh 的远端几何，使用射线/三角形遮挡检查；真实 mask 端点从腕部连接处的测地距离寻找远端分支，处理粘连、缺失和截断，并携带置信度。
- 端点采用归一化坐标上的 Smooth L1；比较左右两种完整对应，取总损失较小者。
- 有标签的合成样本增加旋转、平移和关节监督；每次预测均监督。
- 网络参数化保证 SO(3)、深度和关节角范围、夹爪角和非负；显式角度项作为一致性约束，合法输出时通常为零。

训练验证额外检查“只使用 mask loss”也能反传到位姿头和 ResNet 的第一层，以排除只是 pose 真值损失在更新网络的情况。

## 复现与迁移

实现固定于 nvdiffrast 0.4.0、源码提交 `253ac4fcea7de5f396371124af597e6cc957bfae`。Windows 项目本地工具链准备脚本为 `scripts/setup_nvdiffrast_windows.py`；NVIDIA/Microsoft 下载来源及哈希保存在 `.tools/provenance.json`。源码和依赖安装限制见各自随附许可证。

默认渲染配置版本为 3。旧版或不同后端数据不能直接混合训练；生成新目录的数据并重新训练。推理允许加载旧权重作对比并提示不匹配，但旧权重结果不能作为新版训练精度结论。

```powershell
& .\.venv\Scripts\python.exe -m scripts.inspect_renderer
& .\.venv\Scripts\python.exe -m scripts.benchmark_renderer
& .\.venv\Scripts\python.exe -m pytest -q
```

`--supersample 1` 可用于速度实验，`2` 是默认质量/速度折中，`4` 用于细边界检查；更高采样不能消除 OBJ 本身真实存在的几何棱角。
