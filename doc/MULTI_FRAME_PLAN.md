# DREB-VID 多帧输入方案与 3F-Align 实验计划

更新时间：2026-08-18

## 1. 文档目的和当前阶段

目标是将 DREB 扩展为视频多帧输入模型：利用中心帧前后相邻的模糊帧恢复中心帧特征和图像，再对中心帧执行检测，验证视频时序信息是否能够提升去模糊-检测联合任务。

当前单帧 DREB-VID baseline 已完成，不在本阶段重新实现 1F，也不先做全部消融。当前只准备并实现第一组多帧实验：

```text
3F-Align: [blur[t-1], blur[t], blur[t+1]]
         邻帧特征对齐到中心帧
         时序门控融合
         中心帧去模糊和检测
```

只有 3F-Align 的数据、训练和评估流程稳定，并且结果相对已完成的 1F baseline 有明确分析价值后，才继续实现 3F-Repeat、3F-NoAlign、3F-Shuffle 等消融。

本文件是实现前的设计方案。审核通过前不修改模型、数据集、训练和测试代码。

## 2. 设计原则

1. 保持现有 1F baseline 可复现。默认 `DREB_Net` 不改变，新增多帧模型使用独立架构名，例如 `DREB_Net_MF`。
2. 输入只使用模糊视频帧，不能把相邻清晰帧作为输入，避免数据泄漏。
3. 所有检测标注只对应中心帧 `t`。
4. 第一版使用独立三帧 clip，不引入跨 batch 或跨 sequence 的隐状态，因此训练仍可以 `shuffle=True`。
5. 尽量复用 DREB 的共享特征、MAGFF、LFAMM、检测头和去模糊解码器，减少结构变化带来的混淆。
6. 第一轮不同时改变检测损失、去模糊损失和训练阶段设置，保证与 1F 结果可比较。
7. 3F-Align 先验证时序利用是否有效，参数量、速度和显存作为同步记录项；其他消融暂不提前实现。

## 3. 推荐网络：DREB-MF-Align

### 3.1 总体结构

```text
blur[t-1] ─┐
blur[t]   ─┼─ 共享浅层编码器 ─→ S0[i], S1[i], S2[i]
blur[t+1] ─┘                         │
                                    │
                 S2[t-1], S2[t+1] 对齐到 S2[t]
                                    │
                         时序门控融合
                                    │
                              S2_temporal
                                    │
center frame S0[t] ─→ 原 DREB 检测编码器 ─→ D2[t]
                                    │             │
                                    │             └─ LFAMM + MAGFF(D2[t], S2_temporal)
                                    │                              │
                                    │                        stage3/4 + 检测头
                                    │
S2_temporal + S1[t] + S0[t] ─→ 原 DREB 去模糊解码器 ─→ sharp_hat[t]
```

这里的“先去模糊再检测”采用 DREB 原有的特征级联合方式：不是先生成 RGB 清晰图，再额外运行一套检测 backbone，而是先形成包含时序恢复信息的中心帧特征，再同时服务去模糊解码器和检测分支。

### 3.2 输入和特征提取

输入张量为：

```text
[B, 3, 3, H, W]
```

其中第二维是时间帧，第三维是 RGB 通道，时间顺序固定为 `[t-1, t, t+1]`。

三帧共享现有 DREB 的浅层路径：

```python
stage0
deblur_down1
deblur_down2
```

当前训练输入为 `1024×1024` 时，特征大致为：

```text
S0: 64  × 512 × 512
S1: 64  × 256 × 256
S2: 128 × 128 × 128
```

只有中心帧的 `S0[t]` 继续经过原检测主干 `stage1` 和 `stage2` 得到 `D2[t]`。邻帧不运行完整检测 backbone，以控制多帧计算量和显存。

### 3.3 邻帧特征对齐

第一版只在 `S2` 的 1/8 分辨率进行对齐，避免在 512×512 的 `S0` 上运行昂贵的可变形卷积。

对每个邻帧使用共享的对齐模块：

```text
concat(S2[i], S2[t])
    → offset/mask predictor
    → 3×3 DeformConv2d(S2[i])
    → A[i]
```

其中 `i` 为 `t-1` 或 `t+1`。中心帧直接使用：

```text
A[t] = S2[t]
```

对齐模块使用当前环境已有的 `torchvision.ops.DeformConv2d`，不新增 MMCV 或自定义 C++ 扩展依赖。对齐模块的过去帧和未来帧共享参数，减少参数量并保持时间方向对称。

### 3.4 时序门控融合

不能直接平均或直接拼接邻帧特征，因为邻帧可能存在遮挡、运动和不同程度的模糊。融合模块使用中心帧引导的时序权重：

```text
q = Embed(S2[t])
k[i] = Embed(A[i])
logit[i] = Gate(q, k[i])
w = softmax(logit[t-1], logit[t], logit[t+1])

S2_temporal = S2[t] + Fuse(w[t-1]A[t-1], w[t]A[t], w[t+1]A[t+1])
```

最终融合层采用中心帧残差形式，并将最后一层初始化为接近零，使加载 1F 权重时模型初始行为接近中心帧 DREB。这样可以降低新增时序模块随机初始化对检测分支的干扰。

### 3.5 与 DREB 检测和去模糊分支的连接

保留现有中心帧检测特征 `D2[t]`，只将原来 MAGFF 的去模糊特征输入由 `S2[t]` 替换为 `S2_temporal`：

```python
out_LFAMM = LFAMM(D2_t)
out_joint = MAGFF(D2_t, S2_temporal)
out = out_LFAMM + out_joint
```

然后继续使用原有 `stage3`、`stage4` 和检测头。

去模糊解码器使用：

```text
S2_temporal 作为低分辨率主特征
S1[t]         作为跳跃连接
S0[t]         作为跳跃连接
```

第一版不使用邻帧的高分辨率跳跃连接，避免运动区域直接复制产生重影。

## 4. 数据协议

### 4.1 Dataset 输出

在现有 Dataset 返回值基础上增加：

```text
blur_clip   = [blur[t-1], blur[t], blur[t+1]]
blur_input  = blur[t]              # 保留给现有去模糊 loss 使用
sharp_input = sharp[t]
hm/reg/wh   = frame t 的检测标注
```

模型使用 `blur_clip`，去模糊监督使用中心帧 `sharp_input`，检测监督使用中心帧标注。

### 4.2 邻帧索引

不能通过全局 image id 加减取得邻帧。应当按：

```text
sequence_id + 数字帧文件名
```

建立每个 sequence 的有序帧索引，并且禁止跨 sequence 取帧。

当前每个 sequence 只取前 50 帧时，邻帧也限制在这 50 帧内部：

```text
第一帧： [t, t, t+1]
中间帧：[t-1, t, t+1]
最后一帧：[t-1, t, t]
```

这样 3F-Align 不会使用当前实验子集之外的第 51 帧。

### 4.3 数据增强

三张模糊图和中心清晰图必须共享：

```text
随机 crop 参数
resize/affine 参数
左右翻转决定
颜色增强参数
```

不能分别对三帧随机 crop 或随机 flip，否则模型会把增强误认为运动。检测框只对中心帧 `t` 做坐标变换。

## 5. 需要修改的代码范围

审核通过后预计修改以下部分：

### 5.1 模型

建议新增：

```text
lib/models/networks/DREB_Net_multiframe_model.py
```

并在：

```text
lib/models/model.py
```

增加 `DREB_Net_MF` 的 factory 映射。原有 `DREB_Net` 不改，避免破坏已经完成的 1F baseline。

新增模块预计包括：

```text
TemporalDeformAlign
TemporalGateFusion
DREB_Net_MF
```

### 5.2 Dataset 和 batch

修改：

```text
lib/datasets/dataset/visdrone2019DET.py
lib/datasets/sample/ctdet.py
```

主要内容：sequence 帧索引、三帧读取、边界处理、共享增强、`blur_clip` 返回。

### 5.3 Trainer 和 loss

修改：

```text
lib/trains/ctdet_trainer.py
```

主要内容：

```text
SB_deblur 模型输入改为 batch['blur_clip']
去模糊监督仍使用 batch['sharp_input']
Stripformer loss 若启用则使用中心帧 batch['blur_input']
```

模型的 train/val 返回格式尽量保持不变，避免扩大训练框架改动范围。

### 5.4 选项和脚本

修改：

```text
lib/opts.py
bash/train.sh
bash/evaluation_vid.sh
```

新增或固定：

```text
--arch DREB_Net_MF
--num_input_frames 3
```

实验使用新的 `EXP_ID`，不覆盖 1F 输出。

### 5.5 测试和预取

修改：

```text
test.py
lib/detectors/ctdet_detector.py
```

测试时必须按中心帧构造三帧 clip，并使用中心帧的 meta 进行后处理。可视化仍绘制中心模糊帧上的检测框。

## 6. 训练与初始化策略

### 6.1 第一阶段：工程 smoke test

先不进行完整训练，完成以下检查：

1. 一个 batch 的 `blur_clip` 形状为 `[B,3,3,H,W]`。
2. 三帧来自同一 sequence，顺序正确，边界不跨 sequence。
3. 三帧增强后的空间尺寸完全一致。
4. 模型输出检测头尺寸与 1F 相同。
5. train mode 能输出检测结果和 `deblur_out`。
6. loss 能正常反向传播。
7. 只输入重复中心帧时，新增网络不会产生 NaN 或尺寸错误。

可以加载已有 1F 的 `model_last.pth` 做兼容性检查，但这次只作为初始化和工程验证，不把 warm-start 结果直接当作最终公平对比结果。

### 6.2 第二阶段：正式 3F-Align

正式实验保持 1F baseline 的主要协议：

```text
train sequences: 56
val sequences: 7
每个 sequence 前 50 帧
输入分辨率: 1024
训练/验证 sequence 划分不变
epoch: 200
deblur_loss: mse_ssim
deblur_weight: 0.05
deblur_train_end_epoch: 100
```

训练从新的 `DREB_Net_MF` 初始化开始，随机种子与 1F 一致。若显存不足导致 batch size 必须下降，应使用梯度累积或明确记录有效 batch size，不能只比较不同训练预算下的结果。

建议实验 ID：

```text
train_DREB_Net_VID_3f_align
test_VID_3f_align_epochXXX
```

### 6.3 模型选择

现有 `model_best.pth` 按联合验证 loss 选择，但 1F 结果表明 loss 最优不等于检测 AP 最优。因此 3F-Align 第一轮至少保留：

```text
model_best_loss.pth
model_last.pth
epoch 11/50/100/150/200 的 checkpoint 或检测结果
```

首轮不强行修改现有 checkpoint 机制，但测试时以中心帧检测 AP 为主要比较依据，并同步报告 loss 选择的 checkpoint。

## 7. 3F-Align 首轮评估

第一轮只与已经完成的 1F baseline 比较，不提前实现其他消融。

检测指标：

```text
AP@[0.50:0.95]
AP50
AP75
AP small/medium/large
people/car/truck/bus AP
AR@1/10/100
```

去模糊指标在测试流程能够稳定保存 `deblur_out` 后增加：

```text
PSNR
SSIM
LPIPS（如环境和实现允许）
```

工程指标：

```text
参数量
显存峰值
网络延迟
FPS
```

初步判断 3F-Align 是否值得继续的依据：

1. AP 或 AP50/AP75 相对 1F 有稳定提升，而不是只有单个 checkpoint 偶然提升。
2. AR 不明显恶化，尤其关注小目标和 people/car。
3. 去模糊质量没有明显下降。
4. 时延和显存开销在后续完整实验可接受范围内。
5. 结果不是由测试使用额外 sequence 外邻帧或清晰邻帧造成的。

当前 350 帧验证子集只用于快速筛选，不能作为最终结论。若 3F-Align 有效，再在完整验证集和不同 exposure span 上确认。

## 8. 后续消融顺序（3F-Align 有效后再做）

后续顺序暂定为：

1. `3F-Repeat`：输入 `[blur[t], blur[t], blur[t]]`，控制新增参数和训练结构。
2. `3F-NoAlign`：保留时序融合，移除可变形对齐，验证对齐模块贡献。
3. `3F-Shuffle`：邻帧顺序或内容打乱，验证收益是否来自真实时间关系。
4. `5F-Align`：在 `exposure_span=5.0` 下扩大输入窗口，分析帧数影响。

如果 `3F-Align` 相对 1F 没有提升，优先检查数据对齐、边界处理、时序门控和 checkpoint 选择，不立即增加更复杂的循环网络。

## 9. 主要风险

1. 当前 `exposure_span=5.0` 的合成模糊时间窗口大于三帧输入窗口，三帧可能只覆盖部分模糊形成过程。
2. 连续帧模糊图由相邻清晰帧生成，曝光窗口可能存在较强重叠，时序收益可能对当前合成数据偏乐观。
3. UAV 视频存在目标和相机快速运动，错误对齐可能产生重影，因此必须保留中心帧残差和时序门控。
4. `truck`、`bus` 的类别不均衡不能通过多帧结构自动解决，仍需独立报告类别结果。
5. 多帧模型的去模糊 loss 在第 100 轮后关闭，后期时序模块可能更偏向检测目标；第一轮保持原协议，后续再单独研究。

## 10. 参考思路

- EDVR：金字塔、级联、可变形特征对齐，以及时空注意力融合。<https://openaccess.thecvf.com/content_CVPRW_2019/html/NTIRE/Wang_EDVR_Video_Restoration_With_Enhanced_Deformable_Convolutional_Networks_CVPRW_2019_paper.html>
- STFAN：视频去模糊中的隐式时空对齐和动态滤波，强调模糊帧光流估计的不稳定性。<https://openaccess.thecvf.com/content_ICCV_2019/html/Zhou_Spatio-Temporal_Filter_Adaptive_Network_for_Video_Deblurring_ICCV_2019_paper.html>
- DSTNet：对邻帧特征进行判别性时序融合，避免直接堆叠有害邻帧信息。<https://openaccess.thecvf.com/content/CVPR2023/html/Pan_Deep_Discriminative_Spatial_and_Temporal_Network_for_Efficient_Video_Deblurring_CVPR_2023_paper.html>
- ESTRNN：具有全局时空注意力的递归视频去模糊方案，作为后续长序列建模参考，不作为当前三帧首版。<https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/5116_ECCV_2020_paper.php>
