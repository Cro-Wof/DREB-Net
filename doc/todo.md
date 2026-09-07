# TODO

## 项目阶段划分

当前目标是：先在 VisDrone-VID 上建立与原版单帧 DREB 可比较的单帧 baseline，再验证多帧输入是否能够提升去模糊检测性能。

- 当前阶段只关注检测性能的公平对比，不把小规模子集结果当作完整数据集最终性能。
- 后续阶段再完善全量训练、视频时序质量、类别不均衡和真实场景泛化问题。

## 1. 当前阶段：单帧 baseline 与多帧输入对比

### 1.1 数据、标注和模糊图

- [x] 将 VisDrone-VID 原始 sequence annotation 拆分为逐帧标注。
- [x] 生成 DREB 所需四类：`people`、`car`、`truck`、`bus`。
- [x] 暂时保持原版 DREB 的类别处理策略，不新增 ignore mask，不修改检测 loss。
- [x] 完成清晰图、模糊图、逐帧 annotation 和 COCO JSON 的路径接入。
- [x] 当前调试数据的清晰图、模糊图和标注路径一一对应；COCO JSON 可被正常加载。
- [x] 当前 RIFE 模糊生成配置已验证：`exposure_span=5.0`、`num_subframes=21`。
- [x] 当前 manifest 共 33,682 行，抽样解码 240 张清晰/模糊图，无损坏、零字节或尺寸不一致问题。
- [ ] 完成正式实验所需的 train/val/test-dev 全量模糊图生成和完整性检查。

### 1.2 单帧 DREB baseline

- [x] 完成 VID loader 接入、`max_objs=256` 和训练启动参数修改。
- [x] 调试训练固定使用每个 sequence 的前 N 个连续帧，不能均匀抽帧。
- [x] 当前单帧训练使用 `shuffle=True`；清晰图、模糊图和标注通过同一 image id 读取，不会错配。
- [x] 验证集使用 `shuffle=False`。
- [x] 测试时支持输出检测框可视化、FPS、平均延迟和参数量。
- [x] 修正 COCO evaluator，使其只评估当前 loader 实际选中的 `self.images`，避免 350 张子集被按完整验证集评估。
- [x] 测试脚本支持 `SAVE_VISUALIZATIONS=0` 关闭可视化输出。


### 1.3 多帧输入实现与公平比较

- [ ] 实现三帧或多帧 loader，在 dataset 内根据中心帧 `t` 构造有序的 `[t-1, t, t+1]`，不能依赖 DataLoader batch 中相邻样本拼接。
- [ ] 多帧输入的检测标注和检测 loss 始终对应中心帧 `t`。
- [ ] 显式处理 sequence 边界、第一帧和最后一帧，禁止跨 sequence 取邻帧。
- [ ] 若每个样本是独立 clip，多帧 loader 可以继续 `shuffle=True`；只有带跨 batch 隐状态的时序模型才需要关闭 shuffle。
- [ ] 保持单帧和多帧使用完全相同的数据子集、标注、训练轮数、相同 sequence、相同前 N 连续帧、相同 train/val 划分、增强、随机种子、epoch 和测试配置。
- [ ] 主要比较中心帧检测 AP，并同步比较四类 AP、AR、推理延迟和显存开销。

## 2. 后续阶段：多帧验证通过后的完善事项

### 2.1 全量数据和类别问题

- [ ] 完成全量 train/val/test-dev 训练和测试，不再只使用每个 sequence 的前 N 帧。
- [ ] 评估不同 sequence、运动幅度和模糊程度下的性能，避免前缀子集代表性不足。
- [ ] 针对 `truck`、`bus` 等少数类别处理类别不均衡，并报告类别独立指标。
- [ ] 研究 VisDrone 漏标、ignored regions 和未纳入四类目标的问题；当前不自动补标、不自动生成伪标签，以保持与原版 DREB 的可比性。
- [ ] 如需提高数据质量，再单独建立人工审计集、all-category 标注和 ignore-aware 训练/评估方案。

### 2.2 去模糊质量和视频时序评价

- [ ] 修改测试流程保存 `deblur_out`，在有清晰参考图时计算 PSNR、SSIM 和 LPIPS。
- [ ] 明确合成长曝光模糊与中心清晰帧之间的参考关系，避免将中心帧误认为严格的物理 latent sharp target。
- [ ] 增加连续帧时序一致性评价，如光流 warp error、闪烁程度和相邻帧检测框稳定性。
- [ ] 分别报告去模糊质量提升和检测性能提升，避免只用检测 AP 代表去模糊效果。

### 2.3 训练、模型选择和工程性能

- [ ] 建立按 epoch 自动测试 AP 的流程，用检测 AP 选择最佳 checkpoint，同时保留 loss 最优 checkpoint 作对照。
- [ ] 分析 deblur loss 在 `deblur_train_end_epoch=100` 后关闭对训练 loss 可比性的影响。
- [ ] 完善全量测试的真实端到端 FPS 统计，将磁盘读取、预处理、网络推理和后处理分别统计。
- [ ] 比较单帧和多帧模型的参数量、显存、延迟和 FPS，评估时序输入的实际代价。
- [ ] 如后续扩展为多目标跟踪，再增加 HOTA、IDF1、MOTA 等跟踪指标；纯检测阶段不要求这些指标。

### 2.4 泛化与最终实验

- [ ] 在完整 VisDrone-VID test-dev 上进行最终对比，并固定最终报告协议。
- [ ] 评估不同 `exposure_span`、运动幅度和模糊强度下的鲁棒性。
- [ ] 检查模型在真实视频模糊上的泛化能力，避免只对 RIFE 合成模糊有效。
- [ ] 整理单帧、多帧、清晰输入、模糊输入和去模糊输出的可视化对比图。
