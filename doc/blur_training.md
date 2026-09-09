# VisDrone2019-VID 模糊输入直接检测对照实验

在云端项目根目录执行：

```bash
bash bash/train_blur.sh
```

训练和验证均使用 `--inp_sharp_or_blur blur`，模糊图直接送入检测网络，不先恢复清晰图。
复用清晰实验的检测结构：保留参与特征融合的共享分支（包括 `deblur_down1/2`），跳过去模糊重建解码器。
检测路径正常反向传播，仅计算 `hm_loss + 0.1 * wh_loss + off_loss`，不计算去模糊损失，不使用清晰图重建监督。
从与清晰实验相同的默认初始化开始，不加载清晰实验或联合训练权重。

## 数据与控制变量

- 模糊帧：`/home/zhuhongxiang/DataSet/VisDrone2019-VID-DREB/{train,val}/blur_images/<序列>/<帧>.jpg`
- 标注：`/home/zhuhongxiang/DataSet/VisDrone2019-VID-DREB/{train,val}/annotations_dreb4.json`
- 配对清晰帧：`/home/zhuhongxiang/DataSet/VisDrone2019-VID/VisDrone2019-VID-{train,val}/sequences/<序列>/<帧>.jpg`

沿用现有模糊图，不重新生成。当前共用加载器仍读取配对清晰图并使用图像尺寸进行预处理，需保留清晰目录；清晰图不作为本次模型输入或重建监督。

默认配置与 `bash/train_sharp.sh` 一致：`DREB_Net`、`visdrone_vid`、单帧输入、1024 分辨率、每序列前 50 帧、200 轮、batch size 4、Adam、初始学习率 `2e-4`、现有线性学习率衰减、每轮验证、默认随机种子 317、GPU 1。数据增强、检测损失及数据划分均沿用现有流程。

正式运行前核对已完成清晰实验的 `opt.txt`。如果当时覆盖了默认参数，本次应使用相同设置。可通过环境变量调整，例如：

```bash
CUDA_TRAIN_DEVICE=0 NUM_WORKERS=4 bash bash/train_blur.sh
```

还支持 `MASTER_BATCH_SIZE`、`MAX_FRAMES_PER_SEQUENCE`、`NUM_EPOCHS`、`VAL_INTERVALS`、`PRINT_ITER`。
实验输出保存到 `exp/detect/train/train_DREB_Net_VID_blur/`，与清晰实验分开。

## 运行检查与评估

完整训练前，在具备依赖和数据的环境中，用独立实验名进行小规模前向、反向与评估检查：确认输入为模糊图、检测参数有梯度、重建解码器未执行，且日志只有检测损失项。检查运行应使用独立输出目录，避免与正式结果混用。

`model_best.pth` 沿用清晰实验规则，按最低验证损失选择，并非按最高 AP 选择。
训练完成后执行：

```bash
bash bash/evaluation_vid_blur.sh
```

评估通过现有 `evaluation_vid.sh` 使用模糊验证帧，默认加载本次实验的 `model_best.pth`，每序列前 50 帧，结果保存到 `exp/detect/test/test_VID_blur/`。
支持现有评估环境变量，包括 `CUDA_VAL_DEVICE`、`MODEL_PATH`、`MAX_FRAMES_PER_SEQUENCE`、`TEST_SPLIT`、`FLIP_TEST` 和 `SAVE_VISUALIZATIONS`。

统一比较 AP、AP50、AP75 及各类别 AP，保持评估帧范围、测试增强和权重选择规则一致：

- 清晰 AP − 模糊直接检测 AP：输入质量差异对应的性能差距。
- 联合训练 AP − 模糊直接检测 AP：在训练、评估条件对齐后，联合训练相对直接检测的收益。

本实验是当前设置下的经验参考下限，不是严格数学下界。删除共享特征分支属于另一个结构消融实验，不包含在本次对照中。
