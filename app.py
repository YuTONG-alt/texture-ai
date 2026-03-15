import streamlit as st
import os
import zipfile
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from modelscope import snapshot_download
from diffusers import StableDiffusionPipeline
import torch

# 页面配置
st.set_page_config(
    page_title="AI PBR 贴图生成器",
    page_icon="🎨",
    layout="wide"
)

# 标题
st.title("🎨 AI PBR 贴图生成器")
st.markdown("输入材质描述，自动生成漫反射/法线/粗糙度/金属度四件套")

# 缓存模型加载（只加载一次）
@st.cache_resource
def load_model():
    with st.spinner("首次加载 AI 模型中...（约需 1-2 分钟）"):
        model_dir = snapshot_download("AI-ModelScope/stable-diffusion-2-1")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = pipe.to(device)
        return pipe, device

# 生成贴图函数
def generate_textures(prompt):
    pipe, device = load_model()
    
    # 生成漫反射
    with st.spinner("正在生成漫反射贴图..."):
        full_prompt = f"{prompt}, seamless texture, tileable, high detail, texture only"
        image = pipe(
            full_prompt,
            num_inference_steps=25,
            width=512,
            height=512,
            guidance_scale=7.5
        ).images[0]
        
        # 无缝化处理
        img_np = np.array(image)
        h, w = img_np.shape[:2]
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        X, Y = np.meshgrid(x, y)
        mask = np.sqrt(X**2 + Y**2)
        mask = np.clip(1 - mask, 0, 1)[:,:,np.newaxis]
        offset = np.roll(img_np.copy(), h//2, axis=0)
        offset = np.roll(offset, w//2, axis=1)
        diffuse = Image.fromarray((img_np * mask + offset * (1-mask)).astype(np.uint8))
    
    # 生成法线贴图
    with st.spinner("正在生成法线贴图..."):
        gray = np.array(diffuse.convert('L'))
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        normal_x = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX)
        normal_y = cv2.normalize(sobel_y, None, 0, 255, cv2.NORM_MINMAX)
        normal_z = np.ones_like(gray) * 192
        normal = Image.fromarray(np.stack([normal_z, normal_y, normal_x], axis=-1).astype(np.uint8))
    
    # 生成粗糙度贴图
    with st.spinner("正在生成粗糙度和金属度..."):
        roughness = Image.fromarray((255 - gray).astype(np.uint8))
        metallic = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    
    return diffuse, normal, roughness, metallic

# 创建 ZIP 文件
def create_zip(diffuse, normal, roughness, metallic):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for name, img in [("diffuse.png", diffuse), ("normal.png", normal), 
                         ("roughness.png", roughness), ("metallic.png", metallic)]:
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            zip_file.writestr(name, img_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer

# 主界面
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📝 输入参数")
    
    prompt = st.text_input(
        "材质描述",
        placeholder="例如：red brick wall, weathered stone, wooden planks...",
        value="red brick wall"
    )
    
    st.markdown("""
    **提示词技巧：**
    - 简单描述：`red brick wall`
    - 加细节：`weathered red brick wall with moss`
    - 特定风格：`sci-fi metal panel, rusty`
    """)
    
    generate_btn = st.button("🚀 生成贴图", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.info("💡 首次生成需要加载模型（约 1-2 分钟），后续生成更快")

with col2:
    st.subheader("🖼️ 生成结果")
    
    if generate_btn and prompt:
        try:
            # 生成
            diffuse, normal, roughness, metallic = generate_textures(prompt)
            tab1, tab2, tab3, tab4 = st.tabs(["漫反射 (Diffuse)", "法线 (Normal)", "粗糙度 (Roughness)", "金属度 (Metallic)"])

            with tab1:
                st.image(diffuse, caption="漫反射贴图（基础颜色）", use_column_width=True)
            with tab2:
                st.image(normal, caption="法线贴图（表面凹凸）", use_column_width=True)
            with tab3:
                st.image(roughness, caption="粗糙度贴图（反光程度）", use_column_width=True)
            with tab4:
                st.image(metallic, caption="金属度贴图（金属/非金属）", use_column_width=True)

            # 下载按钮
            st.success("✅ 生成完成！")

            zip_file = create_zip(diffuse, normal, roughness, metallic)
            st.download_button(
                label="📦 下载全部贴图 (ZIP)",
                data=zip_file,
                file_name=f"{prompt.replace(' ', '_')}_pbr.zip",
                mime="application/zip",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"生成失败：{str(e)}")
    else:
        st.info("👈 在左侧输入描述并点击生成")
    st.markdown("---")
st.markdown("Made with ❤️ | 本地运行版本 v1.0")