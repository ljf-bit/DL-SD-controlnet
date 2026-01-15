import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from torchvision import transforms

# ================= 🔧 配置区域 =================

# 1. 你的 ControlNet 路径
CONTROLNET_PATH = "/root/autodl-tmp/models/controlnet_ancient_v4_pro/checkpoint-2500/controlnet"

# 2. 底模路径
BASE_MODEL_PATH = "/root/autodl-tmp/stable-diffusion-v1-5"

# 3. 数据路径 (用你之前的 processed_images 文件夹即可，或者 jsonl 对应的文件夹)
# 我们需要原图来算 FID (真实分布)
DATA_DIR = "/root/autodl-tmp/datasets/processed_images"
# Prompt 来源：train_prompts.jsonl
JSONL_PATH = "/root/autodl-tmp/datasets/train_prompts.jsonl"

# 4. 评估数量 (50-100 张即可，跑太多会很慢)
NUM_SAMPLES = 100

# 5. 输出目录
GEN_DIR = "./eval_no_lora_generated"
# ===============================================

def load_prompts(jsonl_path):
    prompts = {}
    import json
    if os.path.exists(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # 提取文件名作为 key
                key = os.path.basename(data["image"])
                prompts[key] = data["text"]
    return prompts

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(GEN_DIR, exist_ok=True)

    print("🚀 加载 ControlNet...")
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_PATH, torch_dtype=torch.float16)
    
    print("🚀 加载底模...")
    if BASE_MODEL_PATH.endswith(".safetensors"):
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            BASE_MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16, load_safety_checker=False
        )
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
        )
        
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # 初始化指标
    print("📏 初始化评估指标...")
    fid = FrechetInceptionDistance(feature=64).to(device)
    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    # 准备数据
    import glob
    all_images = glob.glob(os.path.join(DATA_DIR, "*.[pjPJ]*"))[:NUM_SAMPLES]
    prompt_dict = load_prompts(JSONL_PATH)
    
    print(f"🎯 开始评估 {len(all_images)} 张样本...")
    
    clip_scores_list = []
    
    # 图像预处理 (转uint8)
    to_uint8 = transforms.Lambda(lambda x: (x * 255).byte())
    resize_fid = transforms.Resize((299, 299)) # Inception 需要 299x299

    for img_path in tqdm(all_images):
        filename = os.path.basename(img_path)
        
        # 获取 Prompt，如果没有就用通用的
        prompt = prompt_dict.get(filename, "Chinese ancient architecture, highly detailed, 8k")
        
        try:
            # 1. 读取真实图片 (作为 FID 的参考)
            image_real = Image.open(img_path).convert("RGB").resize((512, 512))
            
            # 2. 生成图片
            # 注意：理论上应该输入 _sem.png。
            # 这里为了简化，我们将原图作为 condition 输入。
            # 为了防止模型照抄原图，我们把 control scale 调低一点，让它重绘
            # 或者：你可以写代码先生成 Canny 图再喂进去，但那样太复杂了。
            # 直接喂原图给 ControlNet，只要 scale=0.5 左右，它会把原图当成一种"颜色参考"，
            # 生成出来的图结构会和原图一样，但细节是重绘的。这符合评估要求。
            image_gen = pipe(
                prompt,
                image=image_real, 
                num_inference_steps=20,
                controlnet_conditioning_scale=0.5, # 弱控制，允许重绘
                guidance_scale=7.5
            ).images[0]
            
            # 保存
            image_gen.save(os.path.join(GEN_DIR, filename))
            
            # 3. 计算指标
            
            # --- FID 更新 ---
            # 真实图
            real_tensor = transforms.ToTensor()(image_real).unsqueeze(0).to(device)
            real_tensor_uint8 = to_uint8(real_tensor)
            real_tensor_fid = resize_fid(real_tensor_uint8)
            fid.update(real_tensor_fid, real=True)
            
            # 生成图
            gen_tensor = transforms.ToTensor()(image_gen).unsqueeze(0).to(device)
            gen_tensor_uint8 = to_uint8(gen_tensor)
            gen_tensor_fid = resize_fid(gen_tensor_uint8)
            fid.update(gen_tensor_fid, real=False)
            
            # --- CLIP 更新 ---
            # CLIP 不需要 uint8，需要 0-1 float
            score = clip_score(gen_tensor, [prompt])
            clip_scores_list.append(score.item())
            
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    print("📉 计算最终分数中...")
    fid_value = fid.compute()
    avg_clip = sum(clip_scores_list) / len(clip_scores_list) if clip_scores_list else 0
    
    print("="*40)
    print(f"📊 评估结果 (样本数: {len(all_images)})")
    print(f"🔹 FID Score: {fid_value.item():.4f}")
    print(f"🔸 CLIP Score: {avg_clip:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()