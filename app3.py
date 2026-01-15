import gradio as gr
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

# ================= 🔧 配置与模型加载 =================
CHECKPOINT_PATH = "/root/autodl-tmp/models/controlnet_ancient_v4_pro/checkpoint-2500/controlnet"
BASE_MODEL_PATH = "/root/autodl-tmp/stable-diffusion-v1-5"

COLOR_MAP = {
    "building": "#800000",
    "tree":     "#008000",
    "road":     "#808080",
    "white":    "#FFFFFF",
    "eraser":   "#000000"
}

pipe = None
def get_pipe():
    global pipe
    if pipe is None:
        controlnet = ControlNetModel.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL_PATH, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
    return pipe

def create_black_canvas():
    return Image.new('RGB', (512, 512), (0, 0, 0))

# ================= 🎨 核心逻辑 =================

def process_generation(input_dict, prompt, n_prompt, steps, cfg_scale, control_scale):
    """从画板生成图像"""
    if not input_dict or not input_dict['composite']: 
        return None
    p = get_pipe()
    image = input_dict['composite'].convert("RGB").resize((512, 512))
    output = p(
        prompt, image=image, negative_prompt=n_prompt,
        num_inference_steps=int(steps), guidance_scale=float(cfg_scale),
        controlnet_conditioning_scale=float(control_scale)
    ).images[0]
    return output

def process_generation_from_upload(uploaded_img, prompt, n_prompt, steps, cfg_scale, control_scale):
    """从上传图片生成图像"""
    if uploaded_img is None:
        return None
    p = get_pipe()
    image = uploaded_img.convert("RGB").resize((512, 512))
    output = p(
        prompt, image=image, negative_prompt=n_prompt,
        num_inference_steps=int(steps), guidance_scale=float(cfg_scale),
        controlnet_conditioning_scale=float(control_scale)
    ).images[0]
    return output

# ================= 🖥️ 界面构建 =================

my_css = """
.color-btn { font-weight: bold !important; border: 2px solid #333 !important; }
#building-btn { background-color: #800000 !important; color: white !important; }
#tree-btn { background-color: #008000 !important; color: white !important; }
#road-btn { background-color: #808080 !important; color: white !important; }
#line-btn { background-color: #FFFFFF !important; color: black !important; }
#eraser-btn { background-color: #333333 !important; color: white !important; }
.tab-nav button { font-size: 16px !important; font-weight: bold !important; }
"""

with gr.Blocks(css=my_css) as demo:
    gr.Markdown("## ⛩️ 古建筑语义分割交互画板 + 图片上传")
    
    with gr.Tabs() as tabs:
        # ========== Tab 1: 手绘模式 ==========
        with gr.Tab("🎨 手绘模式"):
            current_color = gr.State(COLOR_MAP["building"])
            
            with gr.Row():
                with gr.Column(scale=1):
                    canvas = gr.ImageEditor(
                        value=create_black_canvas(),
                        type="pil",
                        label="绘画区域 (支持键盘 Ctrl+Z 撤销)",
                        height=512,
                        width=512,
                        brush=gr.Brush(default_color=COLOR_MAP["building"], default_size=20),
                        eraser=gr.Eraser(default_size=20),
                        sources=[],
                        layers=False,
                        canvas_size=(512, 512),
                        interactive=True
                    )
                    
                    brush_size_slider = gr.Slider(
                        label="🖌️ 画笔粗细", 
                        minimum=1, 
                        maximum=100, 
                        value=20, 
                        step=1
                    )
                    
                    with gr.Row():
                        btn_building = gr.Button("建筑 (红)", elem_id="building-btn")
                        btn_tree = gr.Button("树木 (绿)", elem_id="tree-btn")
                        btn_road = gr.Button("道路 (灰)", elem_id="road-btn")
                        btn_line = gr.Button("线条 (白)", elem_id="line-btn")
                        btn_eraser = gr.Button("橡皮擦", elem_id="eraser-btn")

                    btn_clear = gr.Button("🗑️ 全清重画", variant="stop")

                with gr.Column(scale=1):
                    result_img_draw = gr.Image(label="生成结果", interactive=False)
                    gen_btn_draw = gr.Button("🚀 开始生成", variant="primary", size="lg")
                    
                    with gr.Accordion("高级参数", open=False):
                        prompt_draw = gr.Textbox(label="Prompt", value="Front view of Chinese ancient architecture, 8k, masterpiece, photorealistic")
                        n_prompt_draw = gr.Textbox(label="Negative Prompt", value="modern, cartoon, blurry, low quality")
                        steps_draw = gr.Slider(10, 50, 30, step=1, label="Steps")
                        cfg_draw = gr.Slider(1, 20, 8.5, label="CFG Scale")
                        con_scale_draw = gr.Slider(0, 2, 1.0, label="ControlNet Scale")

            # --- 手绘模式交互逻辑 ---
            def update_brush(color, size):
                return gr.update(brush=gr.Brush(default_color=color, default_size=size))

            def on_color_btn_click(color, size):
                return color, update_brush(color, size)

            btn_building.click(on_color_btn_click, [gr.State(COLOR_MAP["building"]), brush_size_slider], [current_color, canvas])
            btn_tree.click(on_color_btn_click, [gr.State(COLOR_MAP["tree"]), brush_size_slider], [current_color, canvas])
            btn_road.click(on_color_btn_click, [gr.State(COLOR_MAP["road"]), brush_size_slider], [current_color, canvas])
            btn_line.click(on_color_btn_click, [gr.State(COLOR_MAP["white"]), brush_size_slider], [current_color, canvas])
            btn_eraser.click(on_color_btn_click, [gr.State(COLOR_MAP["eraser"]), brush_size_slider], [current_color, canvas])

            brush_size_slider.change(
                fn=lambda color, size: update_brush(color, size),
                inputs=[current_color, brush_size_slider],
                outputs=[canvas]
            )
            
            btn_clear.click(fn=lambda: gr.ImageEditor(value=create_black_canvas()), outputs=[canvas])
            gen_btn_draw.click(
                fn=process_generation,
                inputs=[canvas, prompt_draw, n_prompt_draw, steps_draw, cfg_draw, con_scale_draw],
                outputs=[result_img_draw]
            )

        # ========== Tab 2: 上传模式 ==========
        with gr.Tab("📤 上传图片"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_img = gr.Image(
                        label="上传控制图 (语义分割图/线稿等)",
                        type="pil",
                        height=512,
                        sources=["upload", "clipboard"]
                    )
                    gr.Markdown("""
                    **使用说明：**
                    - 上传您的控制图（例如语义分割图、边缘检测图等）
                    - 图片会自动调整为 512x512
                    - 支持拖拽上传或从剪贴板粘贴
                    """)

                with gr.Column(scale=1):
                    result_img_upload = gr.Image(label="生成结果", interactive=False)
                    gen_btn_upload = gr.Button("🚀 开始生成", variant="primary", size="lg")
                    
                    with gr.Accordion("高级参数", open=False):
                        prompt_upload = gr.Textbox(label="Prompt", value="Front view of Chinese ancient architecture, 8k, masterpiece, photorealistic")
                        n_prompt_upload = gr.Textbox(label="Negative Prompt", value="modern, cartoon, blurry, low quality")
                        steps_upload = gr.Slider(10, 50, 30, step=1, label="Steps")
                        cfg_upload = gr.Slider(1, 20, 8.5, label="CFG Scale")
                        con_scale_upload = gr.Slider(0, 2, 1.0, label="ControlNet Scale")

            # --- 上传模式交互逻辑 ---
            gen_btn_upload.click(
                fn=process_generation_from_upload,
                inputs=[upload_img, prompt_upload, n_prompt_upload, steps_upload, cfg_upload, con_scale_upload],
                outputs=[result_img_upload]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=6006)