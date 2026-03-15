import streamlit as st
import os
import zipfile
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import requests

# 页面配置
st.set_page_config(
    page_title="AI PBR 贴图生成器",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 AI PBR 贴图生成器")
st.markdown("输入材质描述，自动生成漫反射/法线/粗糙度/金属度四件套")

# API 配置：优先从 Streamlit secrets 读取（.streamlit/secrets.toml）
SILICONFLOW_API_KEY = st.secrets.get("SILICONFLOW_API_KEY", "")
REPLICATE_API_TOKEN = st.secrets.get("REPLICATE_API_TOKEN", "")
if not REPLICATE_API_TOKEN:
    REPLICATE_API_TOKEN = st.text_input("输入 Replicate API Token（可选，留空则用硅基流动）", type="password")

def _generate_siliconflow(prompt):
    """硅基流动 API"""
    if not SILICONFLOW_API_KEY:
        raise ValueError("请在 .streamlit/secrets.toml 中配置 SILICONFLOW_API_KEY，或填写 Replicate Token")
    url = "https://api.siliconflow.cn/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "prompt": f"{prompt}, seamless texture, tileable, 4k, high detail",
        "width": 512,
        "height": 512
    }
    response = requests.post(url, headers=headers, json=data, timeout=60)
    response.raise_for_status()
    result = response.json()
    img_url = result["images"][0]["url"]
    img_response = requests.get(img_url, timeout=30)
    img_response.raise_for_status()
    return Image.open(BytesIO(img_response.content))

def _generate_replicate(prompt):
    """Replicate API"""
    url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "version": "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        "input": {
            "prompt": f"{prompt}, seamless texture, tileable",
            "width": 512,
            "height": 512
        }
    }
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    prediction = response.json()
    import time
    time.sleep(10)
    get_url = f"https://api.replicate.com/v1/predictions/{prediction['id']}"
    result = requests.get(get_url, headers=headers, timeout=30).json()
    img_url = result["output"][0]
    img_response = requests.get(img_url, timeout=30)
    img_response.raise_for_status()
    return Image.open(BytesIO(img_response.content))

def generate_with_api(prompt):
    """使用 Replicate 或硅基流动 API 生成图片，返回 PIL Image"""
    if REPLICATE_API_TOKEN:
        return _generate_replicate(prompt)
    return _generate_siliconflow(prompt)


def process_texture(image):
    """处理贴图，生成四件套"""
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
    
    # 法线
    gray = np.array(diffuse.convert('L'))
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    nx = cv2.normalize(sx, None, 0, 255, cv2.NORM_MINMAX)
    ny = cv2.normalize(sy, None, 0, 255, cv2.NORM_MINMAX)
    nz = np.ones_like(gray) * 192
    normal = Image.fromarray(np.stack([nz, ny, nx], -1).astype(np.uint8))
    
    # 粗糙度和金属度
    roughness = Image.fromarray((255 - gray).astype(np.uint8))
    metallic = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
    
    return diffuse, normal, roughness, metallic

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


col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📝 输入参数")
    prompt = st.text_input(
        "材质描述",
        placeholder="例如：red brick wall, weathered stone...",
        value="red brick wall"
    )
    st.markdown("""
    **提示词技巧：**
    - `red brick wall` - 红砖墙
    - `wooden planks` - 木地板
    - `concrete wall` - 水泥墙
    """)
    generate_btn = st.button("🚀 生成贴图", type="primary", use_container_width=True)
    st.info("💡 使用 API 生成，无需本地模型")

with col2:
    st.subheader("🖼️ 生成结果")
    if generate_btn and prompt:
        try:
            with st.spinner("AI 生成中..."):
                # 生成基础图
                image = generate_with_api(prompt)
                diffuse, normal, roughness, metallic = process_texture(image)

                # 显示结果
                tab1, tab2, tab3, tab4 = st.tabs(["漫反射", "法线", "粗糙度", "金属度"])
                with tab1:
                    st.image(diffuse, use_column_width=True)
                with tab2:
                    st.image(normal, use_column_width=True)
                with tab3:
                    st.image(roughness, use_column_width=True)
                with tab4:
                    st.image(metallic, use_column_width=True)

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
        st.info("👈 输入描述并点击生成")
    st.markdown("---")
st.markdown("Made with ❤️ | API 版 v1.0")