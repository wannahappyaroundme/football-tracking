# train_detector.py
from ultralytics import YOLO
import yaml

# 데이터셋 설정 파일 생성
data_config = {
    'path': './data/football-players-zm06l',
    'train': 'train/images',
    'val': 'valid/images',
    'test': 'test/images',
    'names': {
        0: 'player',
        1: 'referee',
        2: 'ball',
        3: 'helmet'
    },
    'nc': 4  # number of classes
}

with open('configs/football_data.yaml', 'w') as f:
    yaml.dump(data_config, f)

# 모델 학습
model = YOLO('yolov8x.pt')  # 사전학습 모델
model.train(
    data='configs/football_data.yaml',
    epochs=100,
    imgsz=1280,  # 높은 해상도
    batch=8,
    device=0,  # GPU 사용
    project='runs/train',
    name='football_detector'
)