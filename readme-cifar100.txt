在cifar100上进行模型投毒训练并进行测试。
采用模型有badnets，blended，badclip，dobaclip

整体流程为：
1. 训练触发器（只有badclip和dobaclip需要）
python trigger.py --target_class orange --save_path trigger.py
python pro_trigger.py --target_class orange --save_path pro_trigger.pt

2. 训练投毒模型
#badnets训练
python train.py --attack badnets --data_root 数据集根目录 --poison_rate 投毒比例 --epochs 训练轮数 --batch_size 训练批次大小 --lr 学习率大小 --badnets_patch_size 触发器大小 --save_path 保存路径

#blended训练
python train.py --attack blended --data_root 数据集根目录 --poison_rate 投毒比例 --epochs 训练轮数 --batch_size 训练批次大小 --lr 学习率大小 --blended_trigger_path 触发器路径 --blended_alpha 混合系数 --save_path 保存路径

#badclip、dobaclip训练
python train.py --attack badclip/pro --data_root 数据集根目录 --poison_rate 投毒比例 --epochs 训练轮数 --batch_size 训练批次大小 --lr 学习率大小 --trigger_path 触发器路径 --save_path 保存路径


3. 评估
python test.py --attack 攻击方式(badnets，blended，badclip，pro) --model_path 投毒后模型路径 --data_root 数据集根目录 --blended_trigger_path 触发器路径 --trigger_path 触发器路径 --blended_alpha 混合系数 --badnets_patch_size 触发器大小 --defense 防御方式(original，ft，cleanclip) --result_path 结果路径