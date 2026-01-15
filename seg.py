import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ================= 🚀 配置区域 (V4.0 终极版) =================

# 1. 输入路径
INPUT_DIR = "/root/autodl-tmp/datasets/processed_images"

# 2. 输出路径 (建议新建文件夹)
OUTPUT_DIR = "./autodl-tmp/datasets/semantic_maps_final_v4"

# 3. 模型路径
# 如果你已经下载到本地，保持这个路径；
# 如果报错找不到，可以改回 "nvidia/segformer-b5-finetuned-ade20k-512-512" 让它在线下载
MODEL_REPO = "/root/autodl-tmp/segformer_b5_weights" 
# 备用在线地址 (如果本地加载失败，取消注释下面这行)
# MODEL_REPO = "nvidia/segformer-b5-finetuned-ade20k-512-512"

# 4. 颜色定义 (BGR格式 - OpenCV默认)
PALETTE = {
    "background": (0, 0, 0),       # 黑色 - 背景/天空
    "building":   (0, 0, 128),     # 暗红 - 建筑主体/墙/木构/天花板
    "ground":     (128, 128, 128), # 灰色 - 地面/路
    "tree":       (34, 139, 34),   # 森林绿 - 树木/植物
    "stairs":     (0, 255, 255),   # 黄色 - 台阶/楼梯 (修复重点)
    "door_win":   (255, 0, 0)      # 亮蓝 - 门窗 (细节增强)
}

# 5. 边缘线条颜色 (白色)
EDGE_COLOR = (255, 255, 255)

# ==========================================================

def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"🚀 正在初始化 SegFormer B5 (高精度模式)...")
    print(f"📂 模型路径: {MODEL_REPO}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 加载模型
        processor = SegformerImageProcessor.from_pretrained(MODEL_REPO)
        model = SegformerForSemanticSegmentation.from_pretrained(MODEL_REPO).to(device)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 建议: 检查路径是否正确，或将 MODEL_REPO 改为 'nvidia/segformer-b5-finetuned-ade20k-512-512'")
        return

    # 获取所有图片文件
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"🎯 开始处理 {len(files)} 张图片...")

    for filename in tqdm(files):
        img_path = os.path.join(INPUT_DIR, filename)
        
        try:
            # --- 1. 读取图片 ---
            image_pil = Image.open(img_path).convert("RGB")
            image_cv = cv2.imread(img_path) # 用于 Canny
            
            # --- 2. SegFormer 推理 ---
            inputs = processor(images=image_pil, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # --- 3. 上采样 (还原到原图尺寸) ---
            logits = torch.nn.functional.interpolate(
                outputs.logits, 
                size=image_pil.size[::-1], 
                mode="bilinear", 
                align_corners=False
            )
            # 获取每个像素的类别 ID
            pred_seg = logits.argmax(dim=1)[0].cpu().numpy()
            
            # --- 4. 🎨 智能类别映射 (核心逻辑) ---
            # 初始化画布
            semantic_map = np.zeros((image_pil.height, image_pil.width, 3), dtype=np.uint8)
            
            # 我们按照 "从底到顶" 的顺序绘制，后画的覆盖先画的
            
            # A. 地面 (Ground)
            # 4=Floor, 13=Earth, 6=Road, 29=Field, 11=Sidewalk, 46=Sand, 53=Path, 95=Dirt, 14=Grass(有时也算地)
            mask_ground = np.isin(pred_seg, [4, 13, 6, 29, 11, 46, 53, 95])
            semantic_map[mask_ground] = PALETTE["ground"]
            
            # B. 建筑主体 (Building) - 包含墙、木头、天花板
            # 1=Building, 12=Wall, 25=House, 6=Ceiling, 91=Wood, 31=Fence, 10=Cabinet(有时误判)
            mask_building = np.isin(pred_seg, [1, 12, 25, 6, 91, 31, 10])
            semantic_map[mask_building] = PALETTE["building"]
            
            # C. 门窗 (Door/Window) - 覆盖在墙上
            # 8=Window, 14=Door, 33=Gate
            mask_dw = np.isin(pred_seg, [8, 14, 33])
            semantic_map[mask_dw] = PALETTE["door_win"]

            # D. 台阶 (Stairs) - 覆盖在地面/建筑上
            # 19=Stairway, 127=Step
            mask_stairs = np.isin(pred_seg, [19, 127])
            semantic_map[mask_stairs] = PALETTE["stairs"]
            
            # E. 树木 (Tree) - 优先级最高，遮挡一切
            # 5=Tree, 17=Plant, 72=Palm, 9=Grass
            mask_tree = np.isin(pred_seg, [5, 17, 72, 9])
            semantic_map[mask_tree] = PALETTE["tree"]

            # --- 5. 🖍️ 注入 Canny 边缘 (纹理细节) ---
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # 阈值 (30, 150) 能较好地捕捉瓦片和木纹
            edges = cv2.Canny(gray, 30, 150)
            
            # 稍微膨胀，让线条在 512x512 下依然清晰
            kernel = np.ones((2,2), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # 只在有语义颜色的地方画白线 (去除天空噪点)
            mask_has_color = np.any(semantic_map > 0, axis=-1)
            
            # 过滤边缘
            edges_filtered = np.zeros_like(edges)
            edges_filtered[mask_has_color] = edges[mask_has_color]
            
            # 叠加白色线条
            semantic_map[edges_filtered > 0] = EDGE_COLOR

            # --- 6. 保存 ---
            save_name = os.path.splitext(filename)[0] + "_sem.png"
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), semantic_map)
            
        except Exception as e:
            print(f"⚠️ 处理出错 {filename}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"✅ 全部处理完成！")
    print(f"📁 结果保存在: {OUTPUT_DIR}")
    print("💡 提示: 检查生成的图片，应该能看到黄色的台阶、红色的墙壁、绿色的树木以及白色的瓦片纹理。")

if __name__ == "__main__":
    main()