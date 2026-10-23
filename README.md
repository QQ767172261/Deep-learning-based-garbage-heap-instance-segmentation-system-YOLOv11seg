高清垃圾堆分割数据集，基于深度学习垃圾堆实例分割系统 YOLOv11-seg![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8520b0be8c144b7391842b13d48c9072.png)
6244张，yolo，voc，coco三种标注方式
图像尺寸:1200*1600
类别数量: 1 类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/44537160fd4042ebbb2f8de3aea2a716.png)

训练集图像:4929; 验证集图像:1313; 测试集图像:2
总图像数: 6244![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e2cc34cede2548fbac11b81e53ead4db.png)


类别分布:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c84c79c859c143d2904f9478b65373ce.png)
类别名称 | 包含该类别的图像数 | 总标注数
----------------------------------------
sampah 垃圾堆 | 5725 | 9024

模型代码：模型训练使用yolov11n-seg训练，30个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1b57af3673f7404ba1db76b7a2c0ed9e.png)

qt界面：运行界面采用pyqt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/de6eb60cc3a04585bddb11aeb7b02961.png)
# 垃圾堆实例分割系统 YOLOv11-seg 完整代码
任务：露天/道路垃圾堆实例分割，类别：`sampah(垃圾堆)`
## 1、环境安装
```bash
pip install ultralytics opencv-python pyqt5 numpy
```

## 2、数据集配置文件 `garbage.yaml`
```yaml
path: ./garbage_dataset
train: images/train
val: images/val
test: images/test

names:
  0: sampah
```

## 3、模型训练脚本 `train.py`
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载YOLOv11实例分割模型
    model = YOLO("yolo11n-seg.pt")
    # 开始训练
    results = model.train(
        data="garbage.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        amp=False,
        project="runs/train",
        name="garbage_seg"
    )
    print("训练完成！")
    # 模型评估
    metrics = model.val()
    print(f"mAP@0.5:{metrics.seg.map50}")
```

## 4、推理脚本 `predict.py`
```python
from ultralytics import YOLO
import cv2

model = YOLO("runs/train/garbage_seg/weights/best.pt")

def predict_image(img_path):
    res = model(img_path, conf=0.3)
    for r in res:
        img = r.plot()
        cv2.imwrite("result.jpg", img)
    return img

# 图片推理
predict_image("test.jpg")

# 视频推理
# model.predict("test.mp4", save=True, conf=0.3)
```

## 5、PyQt5可视化界面完整代码 `main_ui.py`
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QWidget, QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from ultralytics import YOLO

model = YOLO("runs/train/garbage_seg/weights/best.pt")

class VideoThread(QThread):
    change_pixmap = pyqtSignal(QImage)
    def run(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:break
            res = model(frame, conf=0.3)
            frame = res[0].plot()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h,w,ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
            self.change_pixmap.emit(qimg)
        cap.release()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv11-seg 垃圾堆分割系统")
        self.setGeometry(100,100,1400,800)
        self.thread = None
        w = QWidget()
        self.setCentralWidget(w)
        main_layout = QVBoxLayout(w)
        btn_layout = QHBoxLayout()

        self.btn_img = QPushButton("选择图片")
        self.btn_video = QPushButton("选择视频")
        self.btn_cam = QPushButton("摄像头")
        self.btn_save = QPushButton("保存结果")
        self.btn_img.clicked.connect(self.load_img)
        self.btn_video.clicked.connect(self.load_video)
        self.btn_cam.clicked.connect(self.open_cam)

        btn_layout.addWidget(self.btn_img)
        btn_layout.addWidget(self.btn_video)
        btn_layout.addWidget(self.btn_cam)
        btn_layout.addWidget(self.btn_save)

        img_layout = QHBoxLayout()
        self.label_src = QLabel("原图")
        self.label_res = QLabel("分割结果")
        self.label_src.setFixedSize(640,480)
        self.label_res.setFixedSize(640,480)
        img_layout.addWidget(self.label_src)
        img_layout.addWidget(self.label_res)

        main_layout.addLayout(btn_layout)
        main_layout.addLayout(img_layout)

    def load_img(self):
        file,_ = QFileDialog.getOpenFileName(self,"打开图片","","*.jpg;*.png")
        if file:
            src = cv2.imread(file)
            rgb_src = cv2.cvtColor(src,cv2.COLOR_BGR2RGB)
            h,w,ch = rgb_src.shape
            q_src = QImage(rgb_src.data,w,h,ch*w,QImage.Format_RGB888)
            self.label_src.setPixmap(QPixmap.fromImage(q_src))

            res = model(file,conf=0.3)
            img_res = res[0].plot()
            rgb_res = cv2.cvtColor(img_res,cv2.COLOR_BGR2RGB)
            q_res = QImage(rgb_res.data,w,h,ch*w,QImage.Format_RGB888)
            self.label_res.setPixmap(QPixmap.fromImage(q_res))

    def load_video(self):
        path,_ = QFileDialog.getOpenFileName(self,"打开视频","","*.mp4")
        if path:
            model.predict(path,save=True,conf=0.3)

    def open_cam(self):
        self.thread = VideoThread()
        self.thread.change_pixmap.connect(self.show_frame)
        self.thread.start()
    def show_frame(self,img):
        self.label_res.setPixmap(QPixmap.fromImage(img))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
```
运行：`python main_ui.py`，功能：图片/视频/摄像头实时垃圾堆实例分割，输出mask掩码+检测框。

