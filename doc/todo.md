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


还需要完成 VID loader 路径接入、增大 max_objs，并修正训练启动参数，之后才能正式开始 DREB 训练。