
train_lora.py: 基于ref/train_lora.py修改优化，并加上笔记（会不断更新，所以可能和snapshot的notebook有些差别）

ref：存放原作者的原始文件
snapshot: 存放实际运行时的文件和关键输出
    1_8B_old.ipynb: 直接运行原始train_lora.ipynb，没有任何修改（除了加了打印某些参数）
    1_8B_new.ipynb: 自己整理过，优化了部分写法，流程保持不变。两个文件的结果也差不多
    train_lora.json: 训练数据集
    eval_1.json: 在模型微调前先让它预测前10条样本，chat函数逐条预测太慢了（可能是因为没装flash attention）