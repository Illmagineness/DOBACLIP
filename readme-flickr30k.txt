此过程为DOBACLIP完整训练测试流程，如训练其他模型参考reademe-cifar100.txt

1 文本标签生成
cd text_pro

#训练特征解码器
python train_decoder.py \
    --clip_model /path/to/your/ViT-B-32.pt \
    --data_root /path/to/flickr30k/dataset \
    --save_path /path/to/save/decoder.pt \
    --epochs 32 \
    --batch_size 64

#筛选背景文本
python bg_generator.py \
    --clip_model /path/to/your/ViT-B-32.pt \
    --data_root /path/to/flickr30k/dataset \
    --cifar_root /path/to/cifar10/dataset \
    --target_class dog \
    --topk 50 \
    --out_path /path/to/save/selected_texts_dog.json

#生成标签
python text_generator.py \
    --clip_model /path/to/your/ViT-B-32.pt \
    --decoder_path /path/to/saved/decoder.pt \
    --selected_texts /path/to/saved/selected_texts_dog.json \
    --data_root /path/to/flickr30k/dataset \
    --out_path /path/to/save/poisoned_texts_dog.txt \
    --iterations 3 \
    --lam 0.3

2 训练图像触发器
python pro_trigger.py \
    --data_root /path/to/flickr30k \
    --clip_model /path/to/RN50.pt \
    --target_class dog \
    --poison_text_pool ./poisoned_texts_dog.txt \
    --epochs 100 \
    --save_path ours_trigger_dog.pt

3 训练后门模型
python train.py \
    --attack ours \
    --data_root /path/to/flickr30k \
    --clip_model /path/to/RN50.pt \
    --target_class dog \
    --poison_rate 0.03 \
    --trigger_path ./ours_trigger_dog.pt \
    --poison_text_pool ./poisoned_texts_dog.txt \
    --epochs 10 \
    --save_path poisoned_rn50_dog_ours.pt

4 后门模型评估
python test.py \
    --attack ours \
    --model_path ./poisoned_rn50_dog_ours.pt \
    --data_root /path/to/flickr30k \
    --target_class dog \
    --trigger_path ./ours_trigger_dog.pt \
    --poison_text_pool ./poisoned_texts_dog.txt \
    --result_path final_results.txt \
    --defense cleanclip/ft/original