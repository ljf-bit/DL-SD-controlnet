import os
import json
import glob
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 路径设置 (保持不变)
OLD_JSONL_PATH = "/root/autodl-tmp/datasets/train_prompts.jsonl"
RGB_DIR = "/root/autodl-tmp/datasets/processed_images"
SEM_DIR = "./autodl-tmp/datasets/semantic_maps_final_v4"
OUTPUT_JSONL = "./autodl-tmp/datasets/dataset_controlnet.jsonl"

# ===========================================

def normalize_key(path_str):
    """
    核心修复函数：
    1. 统一将 Windows 反斜杠 \ 换成 Linux 正斜杠 /
    2. 提取文件名
    3. 去掉扩展名
    """
    # 替换反斜杠
    path_str = path_str.replace("\\", "/")
    # 提取文件名
    filename = os.path.basename(path_str)
    # 去掉扩展名
    basename = os.path.splitext(filename)[0]
    return basename

def main():
    print("🚀 开始 V2 强力修复版构建...")
    
    # --- 1. 读取旧索引 (建立查询字典) ---
    prompt_dict = {}
    print(f"📖 读取 JSONL: {OLD_JSONL_PATH}")
    
    with open(OLD_JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            raw_path = data["image"]
            text = data["text"]
            
            # === 关键修复步骤 ===
            key = normalize_key(raw_path)
            prompt_dict[key] = text
            
    # 打印前 3 个 Key 看看长什么样，确认修复效果
    print(f"✅ 字典加载完毕，共 {len(prompt_dict)} 条。示例 Keys (前3个):")
    for k in list(prompt_dict.keys())[:3]:
        print(f"   🔹 '{k}'")

    # --- 2. 扫描语义图 (作为基准) ---
    sem_files = glob.glob(os.path.join(SEM_DIR, "*_sem.png"))
    print(f"🔍 扫描到 {len(sem_files)} 张语义分割图，开始匹配...")
    
    valid_entries = []
    missing_debug = [] # 用于存储失败案例以便分析

    for sem_path in tqdm(sem_files):
        # sem_path: .../Filename_padded_sem.png
        sem_filename = os.path.basename(sem_path)
        
        # 还原 key: Filename_padded
        # 逻辑：去掉尾部的 _sem.png
        key_from_sem = sem_filename.replace("_sem.png", "")
        
        # A. 尝试查找 Prompt
        text = prompt_dict.get(key_from_sem)
        
        # 如果找不到，尝试一些模糊匹配策略 (防删改)
        if text is None:
            # 策略2: 也许 jsonl 里没有 _padded? 试着去掉看看
            key_no_pad = key_from_sem.replace("_padded", "")
            text = prompt_dict.get(key_no_pad)

        if text is None:
            missing_debug.append(key_from_sem)
            continue

        # B. 查找 RGB 原图
        # 我们知道 key_from_sem 对应的原图通常就是 key_from_sem.png/jpg
        # 或者 key_from_sem (如果不含_sem) + 扩展名
        rgb_path = None
        # 优先在 key_from_sem 后面加扩展名找
        possible_names = [key_from_sem + ext for ext in ['.png', '.jpg', '.jpeg', '.JPG', '.PNG']]
        
        for name in possible_names:
            full_path = os.path.join(RGB_DIR, name)
            if os.path.exists(full_path):
                rgb_path = full_path
                break
        
        if rgb_path is None:
            # 极少数情况：如果 key 没找到，试试 key_no_pad 对应的文件名？
            # 这一步通常不需要，除非文件名极度混乱
            pass

        if rgb_path and text:
            entry = {
                "text": text,
                "image": rgb_path,
                "conditioning_image": sem_path
            }
            valid_entries.append(entry)

    # --- 3. 结果分析与写入 ---
    print("\n" + "="*40)
    if len(valid_entries) > 0:
        print(f"🎉 成功匹配: {len(valid_entries)} 组数据！")
        with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
            for entry in valid_entries:
                f.write(json.dumps(entry) + "\n")
        print(f"💾 文件已保存至: {OUTPUT_JSONL}")
    else:
        print("❌ 依然为 0 组！请检查下方的调试信息：")
        if len(missing_debug) > 0:
            print("\n🕵️‍♀️ 匹配失败示例 (文件名 vs 字典Keys):")
            print(f"文件系统中的文件名 (前3个): {missing_debug[:3]}")
            print("请对比上方的 '示例 Keys'，看看哪里不一样？")
            print("常见问题：文件名多了/少了 '_padded'，或者拼写有差异。")

if __name__ == "__main__":
    main()