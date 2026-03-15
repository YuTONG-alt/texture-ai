import streamlit as st
import zipfile
from io import BytesIO
from PIL import Image, ImageFilter
import numpy as np
import requests

st.set_page_config(page_title="AI PBR 贴图生成器", page_icon="🎨")
st.title("🎨 AI PBR 贴图生成器")

def generate(prompt):
    # Pollinations.ai - 完全免费，无需API Key
    url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}?width=512&height=512&nologo=true"
    r = requests.get(url, timeout=60)
    return Image.open(BytesIO(r.content))

def process(img):
    arr = np.array(img)
    h, w = arr.shape[:2]
    # 无缝化
    mask = np.clip(1 - np.sqrt(np.linspace(-1,1,h)**2 + np.linspace(-1,1,w)**2), 0, 1)[:,:,None]
    offset = np.roll(np.roll(arr, h//2, 0), w//2, 1)
    diffuse = Image.fromarray((arr*mask + offset*(1-mask)).astype(np.uint8))
    # 法线
    gray = np.array(diffuse.convert('L'))
    edge = np.array(diffuse.convert('L').filter(ImageFilter.FIND_EDGES))
    normal = Image.fromarray(np.stack([np.ones_like(gray)*192, edge, edge], -1).astype(np.uint8))
    return diffuse, normal, Image.fromarray((255-gray).astype(np.uint8)), Image.new('L', (512,512), 0)

def make_zip(imgs):
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w') as zf:
        for n, i in zip(["diffuse.png", "normal.png", "roughness.png", "metallic.png"], imgs):
            b = BytesIO(); i.save(b, format='PNG'); zf.writestr(n, b.getvalue())
    buf.seek(0); return buf

prompt = st.text_input("材质描述", "red brick wall")
if st.button("🚀 生成", type="primary"):
    with st.spinner("生成中..."):
        try:
            imgs = process(generate(prompt))
            st.session_state.imgs = imgs
            st.success("完成！")
            for t, img, n in zip(st.tabs(["漫反射", "法线", "粗糙度", "金属度"]), imgs, ["diffuse", "normal", "roughness", "metallic"]):
                with t: st.image(img, use_column_width=True)
            st.download_button("📦 下载 ZIP", make_zip(imgs), "texture.zip", "application/zip")
        except Exception as e:
            st.error(f"错误: {e}")