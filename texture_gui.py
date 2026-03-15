import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLineEdit, QLabel, 
                             QTextEdit, QProgressBar, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont
import torch
from PIL import Image
import numpy as np
import cv2
from modelscope import snapshot_download
from diffusers import StableDiffusionPipeline

class Worker(QThread):
    progress = pyqtSignal(str)
    done = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, prompt, outdir):
        super().__init__()
        self.prompt = prompt
        self.outdir = outdir
        
    def run(self):
        try:
            self.progress.emit("加载模型...")
            model_dir = snapshot_download("AI-ModelScope/stable-diffusion-2-1")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_dir, torch_dtype=torch.float32,
                safety_checker=None, requires_safety_checker=False
            )
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            
            self.progress.emit("生成漫反射...")
            img = pipe(f"{self.prompt}, seamless, tileable", 
                      num_inference_steps=25, width=512, height=512).images[0]
            
            # 无缝处理
            arr = np.array(img)
            h, w = arr.shape[:2]
            y, x = np.ogrid[-1:1:h*1j, -1:1:w*1j]
            mask = np.clip(1 - np.sqrt(x*x + y*y), 0, 1)[:,:,None]
            offset = np.roll(np.roll(arr, h//2, 0), w//2, 1)
            diffuse = Image.fromarray((arr * mask + offset * (1-mask)).astype(np.uint8))
            
            # 法线
            gray = np.array(diffuse.convert('L'))
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            nx = cv2.normalize(sx, None, 0, 255, cv2.NORM_MINMAX)
            ny = cv2.normalize(sy, None, 0, 255, cv2.NORM_MINMAX)
            nz = np.ones_like(gray) * 192
            normal = Image.fromarray(np.stack([nz, ny, nx], -1).astype(np.uint8))
            
            # 粗糙度
            rough = Image.fromarray((255 - gray).astype(np.uint8))
            
            # 金属度
            metal = Image.fromarray(np.zeros((512, 512), dtype=np.uint8))
            
            # 保存
            paths = {}
            for name, im in [("diffuse", diffuse), ("normal", normal), 
                           ("roughness", rough), ("metallic", metal)]:
                p = os.path.join(self.outdir, f"{name}.png")
                im.save(p)
                paths[name] = p
                
            self.done.emit(paths)
            
        except Exception as e:
            self.error.emit(str(e))

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI PBR 贴图生成器")
        self.setGeometry(100, 100, 900, 700)
        self.setup_ui()
        
    def setup_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        layout = QVBoxLayout(cw)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题
        title = QLabel("AI PBR 贴图生成器 v1.0")
        title.setFont(QFont("微软雅黑", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 输入
        h1 = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("输入材质描述，如：red brick wall")
        self.input.setFont(QFont("微软雅黑", 11))
        h1.addWidget(self.input)
        
        self.btn = QPushButton("生成")
        self.btn.setFont(QFont("微软雅黑", 11))
        self.btn.setStyleSheet("QPushButton{background:#4CAF50;color:white;padding:8px 16px}")
        self.btn.clicked.connect(self.start)
        h1.addWidget(self.btn)
        layout.addLayout(h1)

        # 进度
        self.bar = QProgressBar()
        self.bar.setRange(0, 0)
        self.bar.hide()
        layout.addWidget(self.bar)

        # 日志
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 10))
        layout.addWidget(self.log)

        # 预览
        grid = QHBoxLayout()
        self.previews = {}
        for name in ["diffuse", "normal", "roughness", "metallic"]:
            v = QVBoxLayout()
            v.addWidget(QLabel(name.upper(), alignment=Qt.AlignmentFlag.AlignCenter))
            lbl = QLabel("等待生成...", alignment=Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background:#eee;border:2px dashed #999;min-width:200px;min-height:200px")
            v.addWidget(lbl)
            grid.addLayout(v)
            self.previews[name] = lbl
        layout.addLayout(grid)

    def start(self):
        prompt = self.input.text().strip()
        if not prompt:
            self.log.append("请输入描述")
            return

        outdir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not outdir:
            return

        self.btn.setEnabled(False)
        self.bar.show()
        self.log.append(f"开始生成: {prompt}")

        self.worker = Worker(prompt, outdir)
        self.worker.progress.connect(self.log.append)
        self.worker.done.connect(self.finish)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def finish(self, paths):
        self.bar.hide()
        self.btn.setEnabled(True)
        self.log.append("完成！")

        for name, path in paths.items():
            if os.path.exists(path):
                pm = QPixmap(path).scaled(190, 190, Qt.AspectRatioMode.KeepAspectRatio)
                self.previews[name].setPixmap(pm)
                self.previews[name].setText("")

    def on_error(self, e):
        self.bar.hide()
        self.btn.setEnabled(True)
        self.log.append(f"错误: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())