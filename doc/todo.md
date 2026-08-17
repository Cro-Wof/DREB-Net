# TODO

## VisDrone-VID 适配 DREB

- [ ] 暂时保持原版 DREB 的 VisDrone 四类处理策略：`people`、`car`、`truck`、`bus`。
- [ ] 暂时不新增 ignore mask，也不修改 DREB 的检测 loss；VisDrone 中未纳入这四类的类别和 ignored regions 按原版转换逻辑过滤。
- [ ] 记录并评估 VisDrone 的潜在漏标问题：可见但未出现在 annotation 中的目标，当前不自动补标、不自动生成伪标签，避免破坏与原版 DREB 及 VisDrone 官方结果的可比性。
- [ ] 后续如需更严格的数据质量研究，再单独实现 all-category/ignore-region 可视化、人工审计集和 ignore-aware 训练/评估，不混入当前原版复现实验。

## 当前调试数据状态

三个 split 的原始帧和模糊帧路径完全对应。
每帧都有对应的 annotations_frame/.../*.txt。
manifest 共 33,682 行，所有模糊图、清晰图、标注路径均存在。
COCO JSON 可被 pycocotools.COCO 正常加载。
类别为 DREB 所需四类：people、car、truck、bus。
RIFE 配置记录正确：exposure_span=5.0、num_subframes=21、generated=frames、skipped=0。
抽样解码 240 张清晰/模糊图，无损坏、零字节或尺寸不一致问题。

## 训练采样与多帧输入约束

- [x] 当前单帧训练保持 `train_loader(shuffle=True)`；清晰图、模糊图和标注通过同一个 image id 读取，不会因 shuffle 错配。验证集保持 `shuffle=False`。
- [x] 调试训练通过 `max_frames_per_sequence` 固定使用每个 sequence 的前 N 个连续帧；`0` 表示恢复全量帧，不能改为均匀抽样，否则会破坏未来多帧输入所需的时间连续性。
- [ ] 实现三帧或多帧 loader 时，可以打乱中心帧样本的顺序，但必须在 dataset 内根据中心帧 `t` 构造有序的 `[t-1, t, t+1]`，不能根据 DataLoader batch 中相邻样本拼接时间帧。
- [ ] 多帧 loader 必须显式处理 sequence 边界、第一帧和最后一帧，禁止跨 sequence 取邻帧；检测标注始终对应中心帧 `t`。
- [ ] 如果未来改为带跨 batch 隐状态的时序模型，再评估是否需要关闭 `shuffle`；对于独立多帧 clip，默认仍可使用 `shuffle=True`。

## 训练接入状态

- [x] VID loader 路径接入、`max_objs=256` 和训练启动参数已完成。
- [x] 当前训练脚本支持稀疏打印和每 sequence 前 N 帧调试训练。
