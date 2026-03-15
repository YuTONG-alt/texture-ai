import torch
from PIL import Image
import numpy as np

# 使用 ModelScope 加载模型
from modelscope import snapshot_download
from diffusers import StableDiffusionPipeline

print("正在从 ModelScope 下载模型...")
# 模型会下载到本地缓存（约 5GB）
model_dir = snapshot_download("AI-ModelScope/stable-diffusion-2-1")

print("正在加载模型...")
pipe = StableDiffusionPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
print(f"使用设备: {device}")

def generate_texture(prompt):
    full_prompt = f"{prompt}, seamless texture, tileable, 4k, high detail"
    print(f"正在生成: {prompt}，请等待...")
    
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
    result = (img_np * mask + offset * (1 - mask)).astype(np.uint8)
    return Image.fromarray(result)

if __name__ == "__main__":
    texture = generate_texture("red brick wall")
    texture.save("output.png")
    print("✅ 已保存到 output.png")