# train_model.py
from ultralytics import YOLO
import torch

def train_football_model():
    """미식축구 전용 모델 학습"""
    
    # GPU 확인
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 데이터셋 경로
    data_path = r"C:\Users\경쓰북\Desktop\Stech\football-tracking\data\football-dataset\data.yaml"
    
    # 모델 초기화
    model = YOLO('yolov8x.pt')
    
    # 학습 설정 (RTX 3050 최적화)
    results = model.train(
        data=data_path,
        epochs=100,
        imgsz=640,  # RTX 3050 4GB VRAM 고려
        batch=8,    # 메모리 절약
        device=0,   # GPU 0번 사용
        workers=4,
        project='runs/train',
        name='football_model',
        patience=10,
        save=True,
        amp=True,   # Mixed precision (속도 향상)
        cache=True  # 데이터 캐싱
    )
    
    print("✅ Training complete!")
    print(f"Best model saved at: runs/train/football_model/weights/best.pt")

if __name__ == "__main__":
    train_football_model()