#!/bin/bash

# ================= 🔧 配置区域 =================
export HF_ENDPOINT=https://hf-mirror.com
# 1. 基础模型 (建议保持 SD1.5)
MODEL_NAME="/root/autodl-tmp/stable-diffusion-v1-5"

# 2. 输出路径 (模型会自动保存在这里)
OUTPUT_DIR="./autodl-tmp/models/controlnet_ancient_v4_pro"

# 3. 验证图片路径 (⚠️重要：请确保这张图片真实存在！)
# 这里默认填了一个，如果报错找不到文件，请手动改成你 semantic_maps_final_v4 里的任意一张图
VAL_IMAGE="./autodl-tmp/datasets/semantic_maps_final_v4/Jishan_Qinglong_Temple_00000-0-DJI_0148_padded_sem.png"

# 4. 验证提示词
VAL_PROMPT="A bird view of a Chinese ancient building with trees and buildings on the top of it, high quality, 8k, masterpiece"

# ================= 🚀 训练参数 (RTX 5090 专享) =================

# 检查验证图是否存在，避免跑起来才报错
if [ ! -f "$VAL_IMAGE" ]; then
    echo "❌ 错误: 找不到验证图片: $VAL_IMAGE"
    echo "👉 请打开 train_pro.sh，修改 'VAL_IMAGE' 变量，指向一张真实存在的语义分割图(_sem.png)。"
    exit 1
fi

echo "🚀 准备开始训练..."
echo "📍 模型输出目录: $OUTPUT_DIR"
echo "🎮 硬件配置: RTX 5090 (BF16 Mode)"

# 启动训练
accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_NAME \
 --output_dir=$OUTPUT_DIR \
 --dataset_name="json" \
 --train_data_dir="/root/autodl-tmp/datasets/controlnet_clean" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "$VAL_IMAGE" \
 --validation_prompt "$VAL_PROMPT" \
 --train_batch_size=10 \
 --gradient_accumulation_steps=4 \
 --mixed_precision="bf16" \
 --checkpointing_steps=500 \
 --validation_steps=100 \
 --max_train_steps=5000 \
 --dataloader_num_workers=8 \
 --report_to="tensorboard" \
 --tracker_project_name="controlnet_ancient_pro" \
 --set_grads_to_none

echo "✅ 训练脚本执行结束！请检查 $OUTPUT_DIR 查看结果。"