# DREB-Net / VisDrone-VID 项目交接文档

更新时间：2026-08-18

本文档用于下一个窗口继续开发，记录当前目标、已完成工作、实验结果、已知问题和推荐执行顺序。

## 1. 项目目标和当前边界

DREB 原本用于单帧图像去模糊检测，当前目标是将其适配到 VisDrone-VID 视频检测数据集，并最终验证多帧输入是否能够提升去模糊检测性能。

当前实验分为两个阶段：

1. 先建立可靠的单帧 DREB-VID baseline，与后续多帧输入使用完全一致的数据和评价协议。
2. 多帧验证有效后，再完善全量训练、视频时序质量、类别不均衡和真实视频泛化。

当前不把 50 帧/sequence 的小规模子集结果当作完整数据集最终性能。

## 2. 环境和数据路径

项目目录：

```text
/home/zhuhongxiang/XM/DREB-Net
```

Conda 环境：

```text
DREBNet
```

GPU 约定：当前训练和测试通常使用物理 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=1
```

原始 VisDrone-VID：

```text
/home/zhuhongxiang/DataSet/VisDrone2019-VID
```

生成的 DREB 数据：

```text
/home/zhuhongxiang/DataSet/VisDrone2019-VID-DREB
```

DREB 数据结构主要为：

```text
VisDrone2019-VID-DREB/
├── train/
│   ├── blur_images/
│   ├── annotations_frame/
│   ├── annotations_dreb4.json
│   └── manifest.jsonl
├── val/
└── test-dev/
```

当前保持原版 DREB 的四类：

```text
people, car, truck, bus
```

暂时不新增 ignore mask、不自动补充 VisDrone 漏标、不生成伪标签，以保证与原版 DREB 的可比性。

## 3. 已完成的主要工作

### 3.1 标注准备

已实现并运行：

```text
tools/visdrone_vid/prepare_annotations.py
```

功能：

- 将 sequence 级 annotation 拆成逐帧 annotation；
- 过滤并转换为 DREB 所需四类；
- 生成逐帧文本标注和 COCO JSON；
- 生成 manifest，关联清晰图、模糊图和 annotation。

当前已验证：

- manifest 共约 33,682 行；
- 清晰图、模糊图和标注路径对应；
- COCO JSON 可以由 `pycocotools.COCO` 加载；
- 抽样解码 240 张清晰/模糊图，无零字节、损坏或尺寸不一致问题。

### 3.2 模糊图生成

已实现：

```text
tools/visdrone_vid/generate_motion_blur.py
```

当前使用 RIFE 插值，并支持 GPU 加速，输出结构保持不变。`exposure_span` 已取消必须小于等于 1 的限制。

已验证的一组配置：

```text
exposure_span = 5.0
num_subframes = 21
weight_mode   = uniform
```

当前调试数据的模糊效果可接受，但正式实验仍需确认 train/val/test-dev 的全量模糊图是否生成完整。

### 3.3 VID loader 和训练接入

主要文件：

```text
lib/datasets/dataset/visdrone2019DET.py
lib/datasets/dataset_factory.py
lib/opts.py
main.py
bash/train.sh
```

已完成：

- 新增 `dataset=visdrone_vid`；
- 接入清晰图、模糊图和 COCO annotation 路径；
- `max_objs` 提升为 256；
- 增加 `max_frames_per_sequence`；
- 调试训练固定使用每个 sequence 的前 N 个连续帧；
- 当前单帧训练使用 `shuffle=True`，验证使用 `shuffle=False`；
- 训练日志支持每 N 个 batch 打印；
- 每轮输出 train loss、val loss、最佳 loss、epoch 时间和总 ETA。

当前调试训练配置为：

```text
train sequences: 56
每个 sequence 前 50 帧
train images: 2800
val sequences: 7
val images: 350
epochs: 200
```

### 3.4 测试、评估和可视化

主要文件：

```text
test.py
lib/datasets/dataset/visdrone2019DET.py
lib/utils/debugger.py
bash/evaluation_vid.sh
```

已完成：

- COCO bbox AP/AR 评估；
- 输出参数量、平均延迟和 FPS；
- 测试时在输入模糊图上绘制检测框、类别和置信度；
- 支持保留 sequence 目录结构保存可视化图；
- 修复 `Debugger` 对 `visdrone_vid` 类别名缺失的问题；
- 修正 COCO evaluator，使其只评估当前 dataset 实际使用的 `self.images`。

评估修正位于：

```python
coco_eval.params.imgIds = list(self.images)
```

没有可视化时可通过环境变量关闭：

```bash
SAVE_VISUALIZATIONS=0
```

## 4. 已完成单帧训练和测试结果

训练输出目录：

```text
exp/detect/train/train_DREB_Net_VID_original/
```

权重：

```text
model_best.pth  # 第 11 轮，按验证 loss 选择
model_last.pth  # 第 200 轮
```

训练日志：

```text
train_single.log
```

训练现象：

- train loss 持续下降；
- val loss 在约第 11 轮达到最低，随后上升；
- `deblur_loss` 在 `deblur_train_end_epoch=100` 后关闭，因此前后阶段的总 loss 不完全可比。

测试输出：

```text
exp/detect/test/test_VID_single_epoch11/result.txt
exp/detect/test/test_VID_single_epoch200/result.txt
```

两次测试均使用同一 350 张验证子集：7 个 sequence，每个 sequence 前 50 帧。

### 4.1 检测指标对比

| 指标 | Epoch 11 | Epoch 200 |
|---|---:|---:|
| AP@[0.50:0.95] | 0.078 | 0.110 |
| AP50 | 0.198 | 0.230 |
| AP75 | 0.046 | 0.094 |
| AP small | 0.029 | 0.026 |
| AP medium | 0.075 | 0.098 |
| AP large | 0.155 | 0.205 |
| AR@1 | 0.037 | 0.013 |
| AR@10 | 0.101 | 0.079 |
| AR@100 | 0.151 | 0.144 |

按类别重新计算的 AP：

| 类别 | Epoch 11 | Epoch 200 |
|---|---:|---:|
| people | 0.071 | 0.105 |
| car | 0.232 | 0.334 |
| truck | 0.010 | 0.000 |
| bus | 0.000 | 0.000 |

### 4.2 结果解释

- 从检测 AP 看，epoch 200 没有出现明显整体过拟合，反而优于 epoch 11；
- `people` 和 `car` 的性能提升明显；
- `truck` 和 `bus` 几乎没有学会，存在类别不均衡和类别偏置；
- epoch 200 的 AP 提升伴随 AR@1/10/100 略有下降，说明模型更偏向少量高质量框，整体召回没有同步提升；
- `model_best.pth` 按联合验证 loss 选择，并不等于检测 AP 最优模型；
- 当前结论仅适用于固定的 350 帧子集，不能直接推断完整 VisDrone-VID 性能。

### 4.3 运行指标

模型参数量：

```text
30.284 M
```

平均延迟：

```text
约 75.7 ms/frame
```

FPS：

```text
约 13.2 FPS
```

注意：默认测试使用 prefetch 流程，因此这里的 `tot` 主要反映 detector 内部推理和后处理，不能完全等同于包含磁盘读取的真实端到端视频 FPS。

## 5. 常用命令

### 5.1 单帧训练

默认使用 GPU 1、每个 sequence 前 50 帧、200 epoch：

```bash
bash bash/train.sh
```

常用覆盖方式：

```bash
CUDA_TRAIN_DEVICE=1 \
MAX_FRAMES_PER_SEQUENCE=50 \
NUM_EPOCHS=200 \
PRINT_ITER=100 \
bash bash/train.sh
```

训练权重位置：

```text
exp/detect/train/train_DREB_Net_VID_original/model_best.pth
exp/detect/train/train_DREB_Net_VID_original/model_last.pth
```

### 5.2 测试 50 帧验证子集并保存可视化

```bash
MODEL_PATH=./exp/detect/train/train_DREB_Net_VID_original/model_best.pth \
EXP_ID=test_VID_single_epoch11 \
MAX_FRAMES_PER_SEQUENCE=50 \
SAVE_VISUALIZATIONS=1 \
bash bash/evaluation_vid.sh
```

输出目录：

```text
exp/detect/test/test_VID_single_epoch11/
├── result.txt
├── results.json
└── visualizations/
```

### 5.3 测试但关闭可视化

```bash
MODEL_PATH=./exp/detect/train/train_DREB_Net_VID_original/model_last.pth \
EXP_ID=test_VID_single_epoch200_fixed \
MAX_FRAMES_PER_SEQUENCE=50 \
SAVE_VISUALIZATIONS=0 \
bash bash/evaluation_vid.sh
```

建议每次使用新的 `EXP_ID`，因为 `result.txt` 使用追加模式。

### 5.4 测试完整 test-dev

```bash
TEST_SPLIT=test-dev \
MAX_FRAMES_PER_SEQUENCE=0 \
SAVE_VISUALIZATIONS=0 \
EXP_ID=test_VID_testdev \
bash bash/evaluation_vid.sh
```

不要在当前 VID 测试中增加 `--not_prefetch_test`：

- `prefetch_test()` 在 `--trainval` 下使用 `test-dev`；
- 普通 `test()` 分支在 `--trainval` 下使用 `test`，当前数据准备的是 `test-dev`。

## 6. 当前已知问题和注意事项

1. 正式全量 blur 图生成和完整性检查仍需确认。
2. 当前测试代码默认只输出整体 COCO 指标，不自动输出 per-class AP；类别分析需要额外评估脚本或后续改造。
3. 当前 `model_best` 按 loss 选择，后续应增加按检测 AP 选择 checkpoint 的流程。
4. 当前验证子集只有 350 帧，`truck` 和 `bus` 各约 100 个目标，类别结论不稳定。
5. 当前尚未保存测试阶段 `deblur_out`，因此没有 PSNR、SSIM、LPIPS 等去模糊质量指标。
6. 当前尚未实现多帧 loader；多帧输入必须由 dataset 根据中心帧构造，不能依赖 batch 顺序。
7. 当前尚未评估时序一致性、闪烁、光流 warp error 或连续帧检测稳定性。
8. 当前 VisDrone 漏标和 ignored regions 暂按原版 DREB 策略处理，不要在 baseline 阶段私自加入伪标签或 ignore mask。
9. `test.py` 的 `result.txt` 使用追加写入；重新测试应使用新的 `EXP_ID`，避免混淆旧结果。

## 7. 推荐下一个窗口的执行顺序

1. 检查当前代码和数据路径，确认 `doc/todo.md` 与本文档一致。
2. 用修正后的 evaluator 重新确认 epoch 11/200 的 350 帧结果，并保留新的 `EXP_ID`。
3. 如需要可靠 checkpoint 曲线，测试 epoch 11、50、100、150、200，并记录 AP、AR 和 per-class AP。
4. 实现三帧 loader：中心帧 `t` 配套 `[t-1,t,t+1]`，处理 sequence 边界，标签对应中心帧。
5. 在完全相同的 50 帧/sequence 子集上训练多帧 baseline。
6. 用相同 checkpoint 规则比较单帧和多帧：AP、AP50、AP75、四类 AP、AR、FPS、显存。
7. 只有多帧检测性能确认有效后，再进行全量训练、去模糊质量和时序一致性评价。

## 8. 关键文件索引

```text
doc/todo.md
doc/HANDOFF.md
doc/rife配置.txt

tools/visdrone_vid/prepare_annotations.py
tools/visdrone_vid/generate_motion_blur.py

lib/datasets/dataset/visdrone2019DET.py
lib/datasets/dataset_factory.py
lib/detectors/ctdet_detector.py
lib/utils/debugger.py
lib/opts.py
lib/trains/ctdet_trainer.py
main.py
test.py

bash/train.sh
bash/evaluation_vid.sh
```
