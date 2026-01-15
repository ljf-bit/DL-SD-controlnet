import os
import json
from PIL import Image
from tqdm import tqdm

# ================= 配置区 =================
# 指向包含所有子文件夹的根目录
INPUT_DIR = "./datasets/raw_data_root"       
OUTPUT_IMG_DIR = "./datasets/processed_images" 
OUTPUT_TXT_PATH = "./datasets/train_prompts.jsonl" 
TARGET_SIZE = 1024 
# ===========================================

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

def resize_and_pad(img, target_size, fill_color=(0, 0, 0)):
    """保持比例缩放 + 黑边填充"""
    ratio = target_size / max(img.width, img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    img = img.resize(new_size, Image.LANCZOS)
    new_img = Image.new("RGB", (target_size, target_size), fill_color)
    paste_x = (target_size - new_size[0]) // 2
    paste_y = (target_size - new_size[1]) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img

print(f"🚀 开始递归扫描目录: {INPUT_DIR}")

# 打开 jsonl 文件准备写入
with open(OUTPUT_TXT_PATH, "w", encoding="utf-8") as f_out:
    
    # os.walk 会像剥洋葱一样一层层遍历所有子目录
    for root, dirs, files in os.walk(INPUT_DIR):
        
        # 过滤掉 txt 文件，只处理 txt 对应的逻辑
        # 我们以 txt 为基准去找对应的图片
        txt_files = [f for f in files if f.endswith(".txt") and not f.startswith("._")]
        
        if len(txt_files) > 0:
            print(f"正在处理文件夹: {os.path.basename(root)} - 发现 {len(txt_files)} 组数据")

        for txt_file in tqdm(txt_files, leave=False):
            # 1. 排除垃圾文件 (关键!)
            if txt_file.startswith("._"):
                continue
                
            base_name = os.path.splitext(txt_file)[0]
            
            # 2. 寻找对应的图片 (png 或 jpg)
            img_name = None
            if base_name + ".png" in files:
                img_name = base_name + ".png"
            elif base_name + ".jpg" in files:
                img_name = base_name + ".jpg"
            
            # 如果没找到图片，或者是垃圾图片文件，就跳过
            if img_name is None or img_name.startswith("._"):
                continue

            full_txt_path = os.path.join(root, txt_file)
            full_img_path = os.path.join(root, img_name)

            try:
                # --- A. 处理文本 ---
                with open(full_txt_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    lines = [line.lstrip("0123456789. ") for line in lines]
                    if len(lines) >= 3:
                        desc = lines[1].strip()
                        category = lines[2].strip()
                        name = lines[0].strip()
                        final_prompt = f"{desc}, {category}, {name}, chinese ancient architecture, 8k resolution"
                    else:
                        continue # 文本格式不对

                # --- B. 处理图片 ---
                with Image.open(full_img_path) as img:
                    img = img.convert("RGB")
                    processed_img = resize_and_pad(img, TARGET_SIZE)
                    
                    # 生成唯一文件名：把文件夹名字拼进去，防止重复
                    # 例如: Zisheng_Temple_00000.png
                    folder_name = os.path.basename(root).replace(" ", "_")
                    save_name = f"{folder_name}_{base_name}_padded.png"
                    save_path = os.path.join(OUTPUT_IMG_DIR, save_name)
                    
                    processed_img.save(save_path)
                    
                    # --- C. 写入索引 ---
                    line = {
                        "image": save_path, 
                        "text": final_prompt,
                        "original_path": full_img_path
                    }
                    f_out.write(json.dumps(line) + "\n")
                    
            except Exception as e:
                print(f"⚠️ 跳过出错文件 {full_img_path}: {e}")

print(f"\n✅ 所有数据处理完毕！")
print(f"📁 处理后图片存放在: {OUTPUT_IMG_DIR}")
print(f"📝 训练索引文件位于: {OUTPUT_TXT_PATH}")