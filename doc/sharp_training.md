# VisDrone2019-VID 清晰输入检测对照实验

在云端项目根目录执行：

```bash
bash bash/train_sharp.sh
```

训练和验证均使用 `--inp_sharp_or_blur sharp`，以原始帧为输入，只计算检测损失。
模型保留参与检测特征融合的共享分支，跳过去模糊重建解码器；检测前向仍正常反向传播。
从默认初始化开始，不加载模糊实验权重。

数据路径沿用 `bash/train.sh`：

- 清晰帧：`/home/zhuhongxiang/DataSet/VisDrone2019-VID/VisDrone2019-VID-{train,val}/sequences/<序列>/<帧>.jpg`
- 标注：`/home/zhuhongxiang/DataSet/VisDrone2019-VID-DREB/{train,val}/annotations_dreb4.json`
- 当前共用的数据加载器仍读取 `VisDrone2019-VID-DREB/{train,val}/blur_images/` 中的配对模糊帧，但 sharp 模式不将其送入模型或用于损失。需保留该目录。

默认参数与完成的联合训练记录 `opt.txt` 一致：单帧输入（`--num_input_frames 1`）、每序列前 50 帧、200 轮、1024 输入分辨率、batch size 4、学习率 `2e-4`、每轮验证、GPU 1。可通过同名环境变量调整，例如：

```bash
CUDA_TRAIN_DEVICE=0 NUM_WORKERS=4 bash bash/train_sharp.sh
```

实验输出保存到 `exp/detect/train/train_DREB_Net_VID_sharp/`。
`model_best.pth` 沿用现有逻辑，按最低验证损失选择，并非按最高 AP 选择。

训练完成后，使用清晰验证帧评估最佳权重：

```bash
bash bash/evaluation_vid_sharp.sh
```

默认评估同样每序列前 50 帧，结果保存到 `exp/detect/test/test_VID_sharp/`。
支持现有评估脚本的环境变量，例如 `CUDA_VAL_DEVICE`、`MODEL_PATH`、`MAX_FRAMES_PER_SEQUENCE`。
与模糊实验比较时，保持实际训练和评估帧范围、轮数及其他设置一致。
清晰检测结果作为当前设置下的经验参考上限，而非去模糊收益的严格上界。
