# 连续视频验证（2026-09-13）

生成与推理入口分别为 `create_data.py --video` 和 `infer_pose.py --video <文件>`。使用现有 `runs/pose_exp2/best.pt`，512×640、2 stages、未开启 AMP，默认上一帧预测作为下一帧初始值。相机/固定初始位姿来自视频元数据，推理未读取逐帧 GT 位姿。

本机已生成并完整处理以下文件：

- `data/continuous_video_15s/video.mp4`：原始 15 秒视频，30 FPS，共 450 帧。
- `data/continuous_video_15s/masks/`：450 张同步语义标签 PNG。
- `data/continuous_video_15s/poses_gt.jsonl`：连续轨迹与时间戳。
- `runs/pose_exp2/video_15s/prediction.mp4`：高 512、宽 1280；左侧原视频，右侧预测 mask overlap，右上角红色帧数。
- 同输出目录的 `metrics.json`、`poses.jsonl` 和 `video_verification.json`：速度、逐帧预测和文件核验结果。
- `opening_frame.png`：第 79 帧夹爪张开示例；`preview.png`：多个时间点的对照。

核验重新解码了输入与输出，二者均为 450 帧、30 FPS、15 秒；逐帧 JSON 与 mask 数量一致，帧序号 1–450。抽查帧 1、151、301、450 的右上角红色数字。左侧为原始视频，仅发生二次有损编码；平均绝对像素差约 0.174/255。

GT 相邻帧最大平移变化 0.388 mm、最大关节角变化 1.268°。轨迹是解析光滑函数，测试还验证了分批生成与一次求值结果一致、关节范围合法、夹爪有充分开合变化。

| 实测项目 | 结果 |
|---|---:|
| 网络 + 初始/中间/最终渲染平均耗时 | 25.29 ms |
| 上述 pipeline FPS | 39.54 |
| 上述 pipeline P95 | 29.87 ms |
| 完整流程耗时 | 31.37 s / 450 帧 |
| 完整流程 FPS | 14.34 |
| 完整逐帧处理 P95 | 74.21 ms |
| shaft / wrist / grippers Dice | 0.8849 / 0.8657 / 0.8289 |

完整流程包括视频解码、预计算 mask 读取、预处理、传输、预测、Dice、overlap、视频编码和 JSON 写入，计入编码器关闭耗时，排除初始化和预热。不包含语义分割网络。输出文件的 30 FPS 是播放帧率，不能据此宣称完整系统达到 30 FPS；当前完整处理尚不满足这一要求。Dice 衡量合成视频上的二维 mask 重叠，不是三维位姿准确率。

`python -m pytest -q`：35 passed。除现有渲染和损失测试外，新增连续轨迹、开合范围、批次独立性、相机缩放、红色计数与左右布局、编解码帧序及预热后重启对齐测试。

## 部件坐标轴更新

新增 `part_transforms` 与显示辅助函数，部件变换已逐顶点对照 `InstrumentMesh.forward` 验证。三个语义类别中的左右夹爪分别旋转，因此视频显示 S/W/L/R 四组坐标轴，X 红、Y 绿、Z 蓝，默认长度 4 mm。显示原点在部件内部，方向保持 canonical FK 局部坐标方向；坐标轴不是二维 mask 主方向，也不是新增预测输出。

使用现有 `data/video_demo/video.mp4` 生成 `runs/video_demo_axes/prediction.mp4`，重新解码确认 450 帧。检查图为同目录 `opening_axes.png`、`closed_axes.png`、`last_axes.png`。测试包含 FK 与网格顶点一致、左右夹爪独立旋转、轴长、近裁剪面/出画处理以及左侧原图保持不变。坐标轴 CPU FK 和绘制不额外渲染 mesh；其耗时包含在新视频的完整流程指标中。该次运行的实际速度见该目录 `metrics.json`，不沿用上一版视频的速度作为新测量结果。
