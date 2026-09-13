# 内镜器械位姿网络

![内镜器械位姿预测演示](docs/assets/prediction.gif)

15 秒预测演示：左侧为输入画面，右侧为预测叠加。GIF 来自 `runs/video_demo/prediction.mp4`，保留完整时长，以 10 FPS、960 × 384 显示；画面中的 FPS 是原推理过程记录的处理速度。

当前训练使用 **6 通道配对 masked RGB**：`concat(target RGB × target mask, moving RGB × moving mask)`。ResNet-34 每个阶段都使用这两个 RGB 图像，预测腕部位姿和三个关节角；nvdiffrast 根据正向运动学渲染预测 RGB 和三通道 mask，保留跨阶段梯度。

## Conda 环境与依赖安装

以下以 **Python 3.10 + CUDA 12.6 版 PyTorch** 为基础。默认可微渲染器 nvdiffrast 需要 NVIDIA GPU、兼容驱动及 CUDA/C++ 构建工具；仅安装 CPU 版 PyTorch 无法运行默认训练流程。Conda 管理 Python 环境，PyTorch 和 Python 包在激活环境后用 pip 安装。

```powershell
git clone https://github.com/shenao1995/instrument-tracking.git
cd instrument-tracking
conda env create -f environment.yml
conda activate instrument-tracking
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

上述 PyTorch/torchvision 配对与原项目环境一致；这是安装基线，并非已经在新 Conda 环境完成的复现验证。其他 CUDA 版本请同时调整 PyTorch 和编译工具链，参考 [PyTorch 官方版本安装说明](https://pytorch.org/get-started/previous-versions/)。

### 安装可微渲染器

**Windows：** 项目提供 CUDA 12.6/MSVC/Windows SDK 本地构建辅助脚本，下载到被 Git 忽略的 `.tools/` 与 `.reference/`，无需复制原机器上的虚拟环境。先激活上面的 Conda 环境，再运行：

```powershell
python -m pip install -r requirements-renderer-windows.txt
# 自动读取当前 GPU 的计算能力；多 GPU 架构可显式设置为分号分隔的列表。
$env:TORCH_CUDA_ARCH_LIST = python -c "import torch; print('.'.join(map(str, torch.cuda.get_device_capability())))"
python setup_nvdiffrast_windows.py
```

**Linux 或已有完整 CUDA/C++ 工具链的 Windows：** 使用与 PyTorch CUDA 版本匹配的 CUDA Toolkit（包含 nvcc）；Windows 另需 MSVC C++ Build Tools 和 Windows SDK。安装项目固定版本：

```text
python -m pip install setuptools wheel ninja
python -m pip install git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae --no-build-isolation
```

渲染器环境要求见 [nvdiffrast 官方说明](https://nvlabs.github.io/nvdiffrast/)。安装后检查：

```powershell
python -c "import torch, torchvision, nvdiffrast.torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
```

主要依赖：`torch` / `torchvision`（网络与训练）、`numpy` / `scipy`（数值计算、端点处理）、`Pillow` / `opencv-python`（图像与视频）、`trimesh` / `fast-simplification`（网格）、`matplotlib`（可视化）、`tensorboard`（训练日志）、`nvdiffrast`（可微渲染）。`pytest` 是开发测试依赖，公开代码包不包含本地 `tests/`。具体版本下限见 `requirements.txt`；Windows 构建辅助包单列于 `requirements-renderer-windows.txt`。

### 运行前准备数据

仓库不包含 `data/`、`runs/`、`tests/`、本地环境、缓存或训练权重。首页 GIF 单独保存在 `docs/assets/`；原始 MP4 不上传。运行前需要自行准备：

- `data/instrument_mesh/`：`transformed_shaft.obj`、`transformed_wrist.obj`、`transformed_gripper_left.obj`、`transformed_gripper_right.obj` 以及匹配的 MTL 材质文件；文件名以代码中的 `PARTS` 为准。
- `data/surgpose_sample/transforms.json`：相机标定与初始位姿；真实图像推理另需 `color/` 和 `l_mask/`。
- 用 `create_data.py` 生成合成训练数据，再运行 `training_pose.py` 获取权重；推理时通过 `--checkpoint` 指定本地权重。

可以通过 `--mesh-dir`、`--calibration` 和相应数据参数使用其他路径。以下历史实验目录和验证记录用于说明工作流，不代表仓库附带这些数据或已训练模型。

## 本次训练更新

- 总损失为 `λrgb Lrgb + λmask Lmask + λtips (λposition Lposition + λgap Lgap)`，默认权重分别为 `0.1、1、1、1、1`。位置与端点间距由 FK 投影监督，**不加入三维 pose 真值损失**；冗余 `angle_loss` 已移除，网络硬约束保留。
- 每个 batch 打印总损失、RGB、mask、tips、tips_position、tips_gap、有效端点对比例及最终阶段端点间距 MAE（像素）。损失使用相同的多阶段权重；`tips` 已包含两个子项，计算总损失时不重复相加。没有有效端点对时，间距 MAE 为 N/A，间距损失为可反传的零。
- 默认 `--val-every 2`：第 2、4、6…轮验证并打印验证总损失及 `shaft / wrist / grippers` 的 Dice。Dice 在最终阶段阈值 0.5 的 mask 上按样本计算后平均；两者均缺失的部件不参与平均。
- 每轮保存 `last.pt`；验证时依据 `--save-best-by val_loss`（默认越小越好）或 `--save-best-by mean_dice`（越大越好）更新 `best.pt`。没有验证的轮次不会虚构验证指标或更新 best。
- TensorBoard 默认写入 `output/tensorboard`：逐 batch、逐 epoch 的总损失与各分项，以及端点指标和验证 Dice。
- 每次验证保存第一个 validation batch 的两列图，每个样本一行：**左列 GT overlap，右列预测 overlap**。两列使用相同 target RGB 作为底图，叠加对应语义 mask；杆身红、腕部绿、夹爪蓝。GT 端点及连线为黄色，预测端点及连线为粉色；无效 GT 为灰色叉，不连接无效端点。同时记录一张 target RGB / 预测渲染 RGB 对照图。若验证集小于 batch size，显示实际样本数。
- 相同图像同步保存到 `output/val_images/epoch_XXXX_*.png`，无需启动 TensorBoard 也能检查。

## RGB 渲染与损失

提供的 OBJ 有 UV/法线、MTL 材质颜色，但没有 `map_Kd` 纹理贴图。`pose_appearance.py` 按面的 `usemtl/Kd` 加载材质，使用平滑顶点法线和相机方向光渲染 RGB；它是 CAD 材质外观，不是恢复出的真实内镜纹理。杆身和腕部 OBJ 中的 MTL 引用名称与实际文件名不同，加载器会明确提示并使用同名 `.mtl`，不修改源 OBJ 文件。

RGB、语义 mask 共享深度缓冲和可微抗锯齿。RGB 输出已经包含像素覆盖率，不会再次乘 soft mask 使边缘变暗。RGB loss 为三个尺度的 Charbonnier 鲁棒误差，按前景并集归一化，避免大面积黑背景稀释误差。预测 RGB 不 detach；并集权重仅作为归一化支持区域使用。

mask loss 为三尺度 soft Dice + 部件平衡 BCE。末端位置采用左右交换匹配的 Smooth L1；间距项比较两端点的欧氏距离。两者使用 GT 夹爪 bbox 对角线归一化（最小 10 像素），不使用预测尺度，也不除以接近零的 GT 间距。间距项仅对两个端点均有效的样本计算。角度参数化已经保证夹爪各自范围和角和非负，因此移除原本恒零的角度惩罚。详见 [TIP_SUPERVISION.md](TIP_SUPERVISION.md)。

nvdiffrast 的训练 AA 逐样本调用，以隔离固定版本中批量梯度合并的问题；栅格化、插值以及无需梯度的 AA 保持批量执行。原始网格默认完整保留，RGB 材质模式当前要求 `--faces-per-part 0`。如添加真实 `map_Kd` 贴图，当前加载器会明确要求接入 UV 贴图 shader，不会默默丢弃贴图后假称用了真实纹理。

## 数据生成、训练和 TensorBoard

先按上文安装并激活 `instrument-tracking` Conda 环境。所有命令均在项目根目录运行。

当前 `data/synthetic_pose_v1` 已使用 version 4 材质 RGB 渲染器，**本轮端点更新不需要重新生成数据**。首次训练自动从现有 mask 和几何端点构建 `tip_labels_v2.npz/.json` 缓存，不改写原始样本。只有更早的随机颜色 RGB 数据仍需重新生成。当前数据输入高 512、宽 640，网络拼接后为 `[B,6,512,640]`；尺寸读取数据 metadata，并非 ResNet 默认的 224。

```powershell
conda activate instrument-tracking

# 先生成少量样本确认外观和可见性，再扩大 num-samples
python create_data.py --output data/synthetic_rgb_pose --num-samples 10000 --height 512 --width 640 --batch-size 4 --supersample 1

python training_pose.py --data data/synthetic_rgb_pose --output runs/pose_rgb --epochs 50 --batch-size 4 --steps 2 --amp --pretrained --val-every 2 --rgb-weight 0.1 --mask-weight 1 --tips-weight 1 --tips-position-weight 1 --tips-gap-weight 1

python -m tensorboard.main --logdir runs/pose_rgb/tensorboard --port 6006
```

浏览器打开 `http://localhost:6006`：SCALARS 查看 `train_batch`、`train_epoch`、`val`；IMAGES 查看 `val/overlap_GT_left_prediction_right` 和 `val/RGB_target_left_render_right`。

想按 Dice 选择最佳权重，在开始新训练时加 `--save-best-by mean_dice`。断点续训保持原定总 epochs、阶段数、损失权重和验证/选择策略一致：

```powershell
python training_pose.py --data data/synthetic_rgb_pose --output runs/pose_rgb --resume runs/pose_rgb/last.pt --epochs 50 --batch-size 4 --steps 2 --amp
python infer_pose.py --checkpoint runs/pose_rgb/best.pt --output runs/pose_rgb/test --track --amp
```

新旧 loss 的评分不可比较：上一轮 6 通道 RGB 权重请通过 `--init-from runs/pose_exp1/best.pt` 加载到新的 `--output`，重新初始化优化器、scheduler 和 best score；`--resume` 仅用于当前版本同配置实验的精确续训。更早的 9 通道 checkpoint 不能作为 RGB 模型初始化。`infer_pose.py` 仍能识别旧 9 通道权重并使用原 mask 配对路径；新 RGB 权重要求目标 RGB，不能使用 `--mask-only`。

训练合成 RGB 直接来自同一个材质渲染器，不再用 mask 随机颜色或 RGB dropout 替换监督目标。真实数据 `data/surgpose_sample/color`、`l_mask` 继续仅用于测试。`transforms.json` 用于相机内参和初始位姿，其中重复矩阵不作为逐帧真值。仅合成训练仍存在外观域差异，加入 RGB 项并不保证真实位姿精度提高，需要完整实验验证。

## 输出验证集可视化

```powershell
python infer_pose.py --checkpoint runs/pose_exp2/best.pt --split val --output runs/pose_exp2/validation
```

`--split val` 默认从 checkpoint 的训练配置找到合成数据目录，只读取 manifest 中的 `val` 样本，使用其保存的初始位姿、相机内参和标签；也可以显式传 `--data data/synthetic_pose_v1`。不传 `--split` 而显式指定含 manifest 的数据目录时，自动选择验证集。默认输出全部验证样本，先检查少量图片可加 `--limit 20`。验证样本相互独立，不能使用视频的 `--track` 或 `--reset-every`。

端点缓存检查也只处理选中的验证样本（包含 `--limit` 限制），不会扫描全部 5,000 个训练/验证样本。进度格式为 `1/1000, sample=0000007`；文件名保留原数据编号，出现 `0004999.png` 不代表输出了 5,000 张。当前验证集共 1,000 个样本，完整 RGB 推理在 `overlays`、`rgb_pairs`、`masks` 中各生成 1,000 张 PNG，总计 3,000 张。

- `overlays/<样本名>.png`：左侧 GT overlap，右侧预测 overlap；GT 端点/连线黄色，预测粉色，无效 GT 灰色叉。与训练 TensorBoard 使用同一个可视化函数。
- `rgb_pairs/<样本名>.png`：目标 masked RGB 与预测渲染 RGB 对照（RGB checkpoint）。
- `masks/<样本名>.png`：预测语义 mask，编码仍为 0/10/20/30。
- `poses.jsonl`：逐样本位姿、Dice、IoU、预测/GT 端点与有效端点间距误差。
- `metrics.json`：全验证集部件 Dice/IoU、有效端点对比例、间距 MAE（像素）以及耗时。端点缺失不按零误差统计；未测量三维位姿准确率。

当前数据每张 overlap/RGB 对照图高 512、宽 1280。独立推理不强制安装或启动 TensorBoard。已有真实视频命令仍可使用；未指定数据或 split 时默认读取 `data/surgpose_sample`。

## 连续 15 秒视频与视频推理

```powershell
conda activate instrument-tracking

# 15 秒 × 30 FPS = 450 帧。原来的随机训练数据生成模式不变。
python create_data.py --video --output data/continuous_video_15s --duration 15 --fps 30 --height 512 --width 640 --supersample 1 --min-depth 0.10 --max-depth 0.15

# 视频模式默认使用上一帧预测初始化下一帧，不需要另加 --track。
python infer_pose.py --checkpoint runs/pose_exp2/best.pt --video data/continuous_video_15s/video.mp4 --output runs/pose_exp2/video_15s
```

输出目录需为空。`create_data.py --video` 使用解析正弦轨迹，平移、空间旋转、腕关节和夹爪开合随时间连续变化，且独立于渲染 batch 大小；不逐帧随机抽样，不通过丢帧或重采样修正运动。默认时长 15 秒、30 FPS；帧数为 `round(duration × fps)`，实际时长为帧数/FPS。`--motion-rotation` 控制视频方向振幅（默认 5°），原来的 `--rotation-range` 只控制随机训练样本。

生成目录包含：

- `video.mp4`：器械材质 RGB 视频，MP4V 编码，黑色背景；这是合成器械运动，不模拟真实内镜背景和照明。
- `masks/frame_000000.png` 等：逐帧同步的无损 0/10/20/30 语义 mask，不从有损 mask 视频中恢复标签。
- `video_metadata.json`：FPS、尺寸、帧数、K、固定初始位姿、模型哈希和渲染设置。
- `poses_gt.jsonl`：逐帧时间戳、GT 位姿和投影端点，供核查运动连续性。**推理不读取这个文件中的逐帧 GT**。

视频推理输出 `prediction.mp4`：左侧完整原视频帧，右侧为原帧叠加预测多标签 mask；右上角显示红色实测完整处理帧率，例如 `FPS: 14.3`。输出帧率沿用原视频，每个输入帧都写入输出；空 mask 帧也保留，并重置跟踪初始化。`poses.jsonl` 保存逐帧预测，`metrics.json` 保存速度和 Dice。`--independent-frames` 可关闭连续跟踪，`--reset-every N` 可周期性重置为固定初始位姿，`--limit N` 可限制处理帧数。

视频右侧默认叠加部件局部坐标轴：**X 红、Y 绿、Z 蓝**。三个语义类别对应四个独立刚体，因此分别显示 `S`（shaft）、`W`（wrist）、`L`（左夹爪）、`R`（右夹爪）。方向严格来自预测 FK，而不是 mask 的二维主方向；坐标约定是 `InstrumentMesh` 对齐后的 canonical 部件坐标系。箭头是真实三维轴的透视投影，朝向相机时会变短。为了显示在器械上，绘制原点放在部件内：杆身距腕部约 25 mm，腕部包围盒中心，夹爪靠近远端。这些是平移后的显示原点，不是物理关节枢轴。坐标轴作为诊断叠加层可显示在遮挡区域上。

右上角 FPS 采用最近 30 个已完成帧的帧数除以总耗时，包含视频解码、mask 读取、预处理、GPU 传输、网络、全部渲染、Dice、坐标轴/overlap、视频编码和 JSON 写入。为计入编码耗时，画面使用截至上一帧的统计值，首帧显示 `FPS: --`；不是源视频的播放帧率，也不是仅网络 forward 的速度。`--fps-window 10` 可改为最近 10 帧；初始化、预热、分割网络和最终编码器关闭不计入画面滚动 FPS，最终关闭耗时仍计入 `metrics.json` 的整体 `end_to_end_fps`。`poses.jsonl` 的 `display_fps` 记录每帧实际显示的数值。

默认轴长 4 mm，可用 `--axis-length-mm 6` 调整；`--no-part-axes` 关闭。只改变视频显示，不改变网络预测和训练；轴投影/绘制耗时包含在 `end_to_end_fps` 中。示例（使用新的空输出目录）：

```powershell
python infer_pose.py --checkpoint runs/pose_exp2/best.pt --video data/video_demo/video.mp4 --output runs/video_demo_axes --axis-length-mm 4
```

推理自动读取视频旁的 `<视频名>_metadata.json` 和对应 mask。外部真实视频需提供 `--mask-dir`（按自然文件名排序，与解码帧逐一对应）以及 `--calibration`；也可提供 `--video-metadata`。当前网络不包含自动分割模型，不能仅凭未分割 RGB 替代已有 mask 输入。

实时性看 `metrics.json`：`pipeline_fps` 仅统计网络与初始/中间/最终渲染，GPU 计时包含同步；`end_to_end_fps` 包含视频解码、预计算 mask 读取、预处理、传输、推理、Dice、overlap、视频编码及 JSON 写入，排除模型加载和预热。另有完整处理的 P95 延迟、源帧间隔预算和预算内帧比例。**视频以 30 FPS 播放不代表处理达到了 30 FPS**；这些计时不包含分割模型耗时，也不等同于真实内镜位姿精度。

## 运动学和语义约定

输出 `R,t` 为腕部到 OpenCV 相机的变换；相机 +x 向右、+y 向下、+z 向前。平移单位米，角度单位弧度，关节顺序为 `alpha, theta_left, theta_right`。SO(3) 用 6D 连续编码正交化；平移范围 x/y ±0.08 m、z 0.025–0.22 m，腕关节 ±90°，夹爪各 ±80° 且角和非负。

```text
T_shaft_camera = T_wrist_camera × inverse(T_wrist_shaft(alpha))
T_left_camera  = T_wrist_camera × Translate(0.009, 0, 0) × Rz(theta_left)
T_right_camera = T_wrist_camera × Translate(0.009, 0, 0) × Rz(-theta_right)
```

依据 [Instrument-Splatting 的运动学](https://github.com/jinlab-imvr/Instrument-Splatting/blob/ac56d1b407734817b44f589cf5d0778837a65e9d/utils/instrument.py)。mask 标签为 `0/10/20/30`，网络监督通道为杆身、腕部、合并左右夹爪。夹爪执行独立运动学，但现有标签不能分别统计左右夹爪 Dice。

## 渲染器安装与验证

渲染器调研见 [RENDERER_RESEARCH.md](RENDERER_RESEARCH.md)，当前端点更新与验证见 [TIP_SUPERVISION.md](TIP_SUPERVISION.md)。上一版四项损失训练的历史验证见 [RGB_TRAINING_VALIDATION.md](RGB_TRAINING_VALIDATION.md)，历史 mask 模式检查记录在 `VALIDATION.md`。

```powershell
python -m pip install -r requirements.txt
python -m pip install setuptools wheel ninja
python -m pip install git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae --no-build-isolation
```

本机 CUDA 12.6 的项目内工具链可由 `setup_nvdiffrast_windows.py` 重建，构建辅助依赖在 `requirements-renderer-windows.txt`。下载来源及哈希保存于 `.tools/provenance.json`，不修改全局驱动/PATH。

本地开发副本如包含 `tests/`，可运行 `python -m pytest -q`；本仓库按发布范围不提供测试目录。

`inspect_renderer.py` 和 `benchmark_renderer.py` 默认检查 mask 渲染，RGB 外观可检查新合成数据的 `preview/*_rgb.png` 以及训练验证图。速度统计中 `pipeline_fps` 不包含分割、文件读取和结果保存；实际运行还需检查 `end_to_end_fps`。真实 mask 端点伪标签仅用于损失评估，推理输入路径不执行该提取。
