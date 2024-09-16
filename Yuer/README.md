# 真人SD微调

### 效果
- 训练集（train）

![](imgs/6747_20.jpg)

- 测试集（test）

![](imgs/3409_15.jpg)

- 合成集（synthesis）

![](imgs/s03_a.jpg)

- 官方prompt（default）

![](imgs/d01.jpg)

### 简介
- 收集并预处理写真照片作为训练数据，控制半身照/全身照、元素、风格的比例
- 用Qwen2-VL生成标记文本，调节温度等参数平衡标记的准确性与多样性，补充特殊和适当的标记，不超过CLIP的最大输入长度
- 选择AWPortrait v1.4作为底座模型，按照官方推荐的推理配置（包括scheduler、negative prompt等）测试标记文本，检验贴合度
- 基于diffusers进行lora微调，通过Min-SNR、调节lr、修改数据分布等方法解决人脸扭曲、训练不稳定、色彩偏差等问题
- 对比微调模型与底座模型在训练、测试、合成、官方prompt上的差别，验证有效性
- 更多信息详见[notion笔记](https://sailoong.notion.site/Yuer-LoRA-0ff495030d77804b8f4cecec48a44d0f)