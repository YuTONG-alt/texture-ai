import torch
from PIL import Image
import numpy as np
import cv2
from modelscope import snapshot_download
from diffusers import StableDiffusionPipeline

# 加载模型（已缓存，这次很快）
print("加载模型...")
model_dir = snapshot_download("AI-ModelScope/stable-diffusion-2-1")
pipe = StableDiffusionPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate_diffuse(prompt):
    """生成漫反射贴图"""
    full_prompt = f"{prompt}, seamless texture, tileable, high detail, texture only, no shadows"
    image = pipe(
        full_prompt,
        num_inference_steps=30,
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

def generate_normal(diffuse_img):
    """从漫反射生成法线贴图"""
    img_gray = np.array(diffuse_img.convert('L'))
    
    # 使用 Sobel 算子计算梯度
    sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 归一化到 0-255
    normal_x = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX)
    normal_y = cv2.normalize(sobel_y, None, 0, 255, cv2.NORM_MINMAX)
    normal_z = np.ones_like(img_gray) * 192  # 默认朝前
    
    # 合并为 BGR（OpenCV 格式）
    normal_map = np.stack([normal_z, normal_y, normal_x], axis=-1).astype(np.uint8)
    return Image.fromarray(normal_map)

def generate_roughness(diffuse_img):
    """生成粗糙度贴图（暗区光滑，亮区粗糙）"""
    img_gray = np.array(diffuse_img.convert('L'))
    
    # 反转：亮的地方粗糙（高值），暗的地方光滑（低值）
    roughness = 255 - img_gray
    
    # 增加一些噪声变化
    noise = np.random.normal(0, 10, roughness.shape).astype(np.int16)
    roughness = np.clip(roughness.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return Image.fromarray(roughness)

def generate_metallic(diffuse_img, is_metal=False):
    """生成金属度贴图（砖块通常是非金属）"""
    if is_metal:
        # 根据颜色判断金属区域（金色、银色等）
        img_np = np.array(diffuse_img)
        # 简单逻辑：偏黄或偏灰的区域可能是金属
        metallic = np.mean(img_np, axis=2).astype(np.uint8)
    else:
        # 非金属材质，全黑
        metallic = np.zeros((512, 512), dtype=np.uint8)
    
    return Image.fromarray(metallic)

def save_textured_pack(prompt, output_dir="./output"):
    """生成并保存完整 PBR 套装"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"生成漫反射: {prompt}")
    diffuse = generate_diffuse(prompt)
    diffuse.save(f"{output_dir}/diffuse.png")
    print("✅ Diffuse 完成")
    
    print("生成法线贴图...")
    normal = generate_normal(diffuse)
    normal.save(f"{output_dir}/normal.png")
    print("✅ Normal 完成")
    
    print("生成粗糙度贴图...")
    roughness = generate_roughness(diffuse)
    roughness.save(f"{output_dir}/roughness.png")
    print("✅ Roughness 完成")
    
    print("生成金属度贴图...")
    metallic = generate_metallic(diffuse, is_metal=False)
    metallic.save(f"{output_dir}/metallic.png")
    print("✅ Metallic 完成")
    
    print(f"\n🎉 全部完成！保存在 {output_dir}/")
    return {
        'diffuse': diffuse,
        'normal': normal,
        'roughness': roughness,
        'metallic': metallic
    }

if __name__ == "__main__":
    # 生成红砖墙 PBR 套装
    textures = save_textured_pack("red brick wall, weathered, realistic")