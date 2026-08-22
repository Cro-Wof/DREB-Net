# DREB_Net_MF：仿 EDVR 两级 PCD 多帧设计方案

更新时间：2026-08-20

## 1. 目标与约束

当前 3F-Align 只带来有限的检测提升，因此下一步优先完善多帧信息利用机制，而不是立即进行 3F-Repeat 等来源消融。

本方案在现有 DREB 结构基础上，引入 EDVR 风格的多尺度特征对齐和时序融合，重点解决邻帧“对不齐”和“融合不合理”两个问题。

设计约束如下：

1. 尽量不修改 DREB 检测主干、LFAMM、MAGFF、检测头和检测损失。
2. 多帧输入仍为 `[t-1, t, t+1]`，只对中心帧 `t` 进行检测和去模糊监督。
3. 第一版只实现两级 PCD：在 `s2` 和 `s1` 上进行粗到细对齐，不在高显存的 `s0` 上进行 DCN。
4. 保持现有去模糊监督策略不变，不修改 `deblur_train_end_epoch` 和当前 loss 权重策略。
5. 直接覆盖现有 `DREB_Net_MF` 实现，不新增模型名称。原始 1F `DREB_Net` 保持不变。
6. 当前版本仍采用滑动窗口式 3F 输入，不引入跨样本或跨 sequence 的隐状态。

EDVR 的核心设计是 PCD 对齐和 TSA 融合：先利用金字塔特征进行粗到细的可变形对齐，再进行时空注意力融合。[EDVR](https://arxiv.org/abs/1905.02716)

## 2. 当前实现与目标结构的差异

当前 `DREB_Net_MF` 的时序分支主要存在以下限制：

| 当前实现 | 目标修改 |
|---|---|
| 只在 `s2` 做一次邻帧对齐 | 在 `s2 -> s1` 上进行两级粗到细 PCD |
| 每个邻帧只有一次 DCN | 低分辨率估计粗偏移，高分辨率继续细化 |
| 使用简单时序门控融合 | 使用中心帧引导的 temporal-spatial fusion |
| 时序特征主要通过一次 MAGFF 进入检测 | 保持 MAGFF 接口，但提供更可靠的 `s2_temporal` |
| 去模糊 decoder 只有 `s2` 使用时序特征 | `s2` 和 `s1` 使用时序融合特征，`s0`暂保留中心帧 |
| 融合卷积全零初始化 | 使用中心帧残差初始化，邻帧分支能够获得梯度 |

不修改 DREB 主干的原因是：当前实验重点是判断多帧利用方式，而不是同时重构 DREB 的检测和频域模块。

## 3. 总体结构

```text
blur[t-1] ─┐
           ├─ 共享 DREB 浅层编码器 ─→ s0[i], s1[i], s2[i]
blur[t]   ─┤                                  │
           │                  ┌─────────────┘
blur[t+1] ─┘                  两级 PCD 对齐
                                      │
                         s2_temporal, s1_temporal
                                      │
              ┌───────────────────────┴───────────────────────┐
              │                                               │
       原 DREB 检测主干                              原 DREB 去模糊 decoder
              │                                               │
   MAGFF(out_center, s2_temporal)              s2_temporal → up2
              │                                s1_temporal → up3
       原 stage3/stage4/heads                  s0_center   → up4
```

中心帧仍然经过原有 DREB 检测主干，邻帧只经过共享的浅层特征路径和新增的时序对齐分支，不运行完整检测 backbone。

## 4. 共享特征提取

输入保持：

```text
x: [B, 3, 3, H, W]
```

其中时间维顺序固定为 `[t-1, t, t+1]`。

三帧共享 DREB 已有的浅层路径：

```python
stage0       -> s0
deblur_down1 -> s1
deblur_down2 -> s2
```

在 `input_res=1024` 时，特征尺寸约为：

```text
s0: [B, 3,  64, 512, 512]
s1: [B, 3,  64, 256, 256]
s2: [B, 3, 128, 128, 128]
```

特征按时间维拆分：

```python
s0_prev, s0_center, s0_next
s1_prev, s1_center, s1_next
s2_prev, s2_center, s2_next
```

中心帧的 `s0_center` 继续进入原有 `stage1` 和 `stage2`。邻帧不进入完整检测主干，以控制显存和计算量。

## 5. 两级 PCD 对齐

### 5.1 设计原则

第一版只在 `s2` 和 `s1` 上对齐：

```text
s2：1/8 分辨率，负责大范围和粗粒度运动
s1：1/4 分辨率，负责局部运动和边缘细化
s0：不做邻帧 DCN，保留中心帧特征
```

这样保留 EDVR PCD 的粗到细思想，同时避免在 `s0` 的 512×512 特征上进行高显存 DCN。

### 5.2 单个邻帧的对齐过程

前帧和后帧分别对齐到中心帧，两个方向共享对齐模块参数。

以邻帧 `i` 为例：

```text
                 s2_i, s2_center
                         │
                   L3 offset/mask
                         │
                 Modulated DCN(s2_i)
                         │
                  aligned_s2_i
                         │
             上采样并参与 L2 offset 预测
                         │
                 s1_i, s1_center
                         │
                   L2 offset/mask
                         │
                 Modulated DCN(s1_i)
                         │
                  aligned_s1_i
                         │
                  cascade refinement
```

### 5.3 L3 粗尺度对齐

在 `s2` 上预测邻帧到中心帧的offset和mask：

```python
offset_l3, mask_l3 = OffsetMaskNetL3(
    torch.cat([s2_i, s2_center], dim=1)
)

aligned_s2_i = DeformConvL3(
    s2_i,
    offset_l3,
    mask_l3
)
```

L3使用较大的有效感受野处理无人机视频中的相机运动和目标位移。

### 5.4 L2 细尺度对齐

L2同时使用当前尺度特征和L3的对齐结果：

```python
coarse_s2 = upsample(aligned_s2_i, size=s1_i.shape[-2:])

offset_l2, mask_l2 = OffsetMaskNetL2(
    torch.cat([s1_i, s1_center, coarse_s2], dim=1)
)

aligned_s1_i = DeformConvL2(
    s1_i,
    offset_l2,
    mask_l2
)
```

L2重点修正目标边缘、局部位移和L3对齐产生的残差误差。

### 5.5 Cascade refinement

在完成L2对齐后，再使用一个轻量级cascade DCN进行最终细化：

```python
aligned_s1_i = CascadeRefine(
    aligned_s1_i,
    s1_center
)
```

Cascade模块不再重新进行大范围搜索，主要用于消除局部错位和边缘重影。

## 6. TSA风格时序融合

对齐后的特征为：

```text
aligned_prev
s_center
aligned_next
```

分别在`s2`和`s1`上进行融合。

### 6.1 Temporal attention

使用中心帧作为参考，生成每个空间位置的时间权重：

```python
q = Query(center)
k_prev = Key(aligned_prev)
k_ctr  = Key(center)
k_next = Key(aligned_next)

score_i = q * k_i
weight = softmax([score_prev, score_ctr, score_next], dim=time)
```

时间权重形状为：

```text
[B, 3, H_l, W_l]
```

这样不同目标和不同空间区域可以使用不同的邻帧信息。

### 6.2 Spatial attention

将时间融合后的特征输入轻量空间注意力：

```python
spatial_weight = SpatialAttention(fused_feature)
```

空间注意力用于抑制以下区域：

- 邻帧运动边界；
- 遮挡区域；
- 对齐失败区域；
- 邻帧中仍然严重模糊的区域。

### 6.3 残差式输出

时序融合不直接替换中心帧特征，而采用：

```python
temporal_feature = center + alpha * temporal_residual
```

其中 `alpha` 可以初始化为 `0.1` 并设置为可学习参数。

目标输出为：

```text
s2_temporal = TSA(s2_prev_aligned, s2_center, s2_next_aligned)
s1_temporal = TSA(s1_prev_aligned, s1_center, s1_next_aligned)
```

## 7. 与原DREB检测分支连接

中心帧检测主干保持不变：

```python
out = stage1(s0_center)
out = stage2(out)
out_LFAMM = LFAMM(out)
```

仍然使用原有 MAGFF，只替换其第二个输入：

```python
out_temporal = MAGFF_attention(out, s2_temporal)
out = out_LFAMM + out_temporal
```

之后保持：

```python
out = stage3(out)
out = stage4(out)
out = deconv_layers1(out)
out = deconv_layers2(out)
out = deconv_layers3(out)
```

检测头和输出格式完全不变，仍然只对中心帧生成检测结果。

第一版不额外修改检测主干内部的stage1/stage2输入，避免多帧分支和DREB主干同时发生结构变化。

## 8. 与原DREB去模糊分支连接

### 8.1 多帧特征连接方式

去模糊解码器使用两级时序特征：

```python
down3 = deblur_down3(s2_temporal)
down4 = deblur_down4(down3)
up1 = deblur_up1(down4, down3)
up2 = deblur_up2(up1, s2_temporal)
up3 = deblur_up3(up2, s1_temporal)
up4 = deblur_up4(up3, s0_center)
deblur_out = deblur_up5(up4, None)
```

这样邻帧信息可以进入`s2`主路径和`s1`跳跃连接，同时保留`s0_center`作为高分辨率中心帧细节，控制显存和重影风险。

### 8.2 去模糊监督保持原样

本次设计不修改现有去模糊监督策略：

- `deblur_train_end_epoch`保持原值；
- 当前`deblur_weight`保持原值；
- 现有`mse_ssim`或`Stripformer`损失保持原样；
- 去模糊监督仍然只使用中心帧`sharp_input`；
- 不新增额外的光流损失、对齐损失或清晰度监督。

这样可以避免将性能提升同时归因于新的loss设计。

## 9. 融合层初始化

当前时序融合层的全部卷积权重为零，会使新增分支在训练初期严格退化为中心帧路径，并使邻帧对齐分支早期梯度较弱。

改为残差式初始化：

```python
temporal_feature = center + alpha * residual
alpha = 0.1
```

中心帧路径保持恒等映射，邻帧路径使用小幅非零响应，确保：

1. 模型初始行为接近1F DREB；
2. 邻帧对齐模块从训练初期即可获得梯度；
3. 对齐错误不会直接覆盖中心帧特征。

如从已有1F权重初始化，加载已有DREB主干参数；新增PCD和TSA模块随机初始化，使用`strict=False`加载。

## 10. 训练与测试协议

### 10.1 训练输入

```text
blur_clip = [blur[t-1], blur[t], blur[t+1]]
sharp_input = sharp[t]
det_target = target[t]
```

三帧必须共享几何增强参数：

- crop；
- resize/affine；
- flip；
- 与几何变换相关的坐标变换。

检测标注只对应中心帧。

### 10.2 序列边界

仍然采用边界重复策略：

```text
第一帧：[t, t, t+1]
中间帧：[t-1, t, t+1]
最后帧：[t-1, t, t]
```

邻帧不能跨 sequence 取样，也不能超过当前实验限制的sequence帧范围。

### 10.3 评价指标

除现有检测指标外，继续记录：

- 参数量；
- 平均单帧延迟；
- FPS；
- 显存峰值；
- 中心帧去模糊输出的PSNR/SSIM，如现有测试流程能够导出图像。

## 11. 修改范围

本次直接修改现有多帧模型文件：

```text
lib/models/networks/DREB_Net_multiframe_model.py
```

直接覆盖当前 `DREB_Net_MF` 中的：

```text
TemporalDeformAlign
TemporalGateFusion
DREB_Net_MF.forward
```

建议保留以下模块名称，减少其他代码改动：

```text
TemporalDeformAlign       → 改为两级 PCD 对齐实现
TemporalGateFusion        → 改为 TSA 风格融合实现
DREB_Net_MF               → 保持原 factory 名称
```

不新增 `DREB_Net_MF_EDVR`，训练和测试脚本继续通过：

```text
--arch DREB_Net_MF
--num_input_frames 3
```

调用。

预计不需要修改：

```text
lib/models/networks/DREB_Net_model.py
lib/trains/ctdet_trainer.py
lib/models/model.py
```

除非实现过程中发现现有输入或返回接口不兼容。

## 12. 实施顺序

1. 在现有 `DREB_Net_MF` 中加入两级 `s2+s1` PCD 对齐。
2. 检查前帧、中心帧、后帧的特征尺寸和边界样本。
3. 将当前门控融合替换为残差式 TSA 融合。
4. 将 `s2_temporal` 接入原有 MAGFF。
5. 将 `s1_temporal` 接入去模糊解码器的跳跃连接。
6. 保持原有去模糊监督和训练阶段设置不变。
7. 使用已有1F权重进行兼容性加载测试，确认非时序参数正确加载。
8. 进行显存、前向输出尺寸和反向传播检查。
9. 重新训练当前 `DREB_Net_MF`，与已有1F和旧3F-Align结果比较。
10. 只有新版本性能稳定后，再设计3F-Repeat等消融。

## 13. 成功判据

本设计的第一目标不是立即降低参数量或提高速度，而是验证更充分的多帧利用是否能够提升联合任务性能。

优先关注：

1. AP是否稳定高于已有1F和旧3F-Align；
2. AP50和AP75是否同时改善；
3. AR是否不出现明显下降；
4. small/medium目标是否受益；
5. 去模糊结果是否具有更清晰的目标边缘；
6. 显存是否能在单张3090上完成训练。

如果AP仍然只有约0.003的边际提升，则应优先检查PCD对齐和TSA权重是否真正学习，而不是立即增加输入帧数或进行来源消融。
