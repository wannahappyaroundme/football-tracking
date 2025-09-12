# train_m3_max.py
"""
M3 Max 최적화 미식축구 모델 학습
- 대용량 배치 처리
- Metal Performance Shaders 활용
- 최고 성능 설정
"""

import os
import torch
from ultralytics import YOLO
import platform
import yaml
from datetime import datetime

class M3MaxTrainer:
    def __init__(self):
        # Apple Silicon 확인
        self.is_mac = platform.system() == 'Darwin'
        self.is_apple_silicon = self.is_mac and platform.processor() == 'arm'
        
        if self.is_apple_silicon:
            # MPS (Metal Performance Shaders) 사용
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            print(f"🍎 Apple Silicon detected: M3 Max")
            print(f"   Device: {self.device}")
            
            # 메모리 정보 (대략적)
            import subprocess
            result = subprocess.run(['sysctl', 'hw.memsize'], capture_output=True, text=True)
            if result.returncode == 0:
                memsize = int(result.stdout.split()[1])
                print(f"   Total Memory: {memsize / (1024**3):.0f} GB")
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"💻 Device: {self.device}")
    
    def get_optimal_batch_size(self, model_name, memory_gb):
        """M3 Max 메모리에 따른 최적 배치 크기"""
        batch_sizes = {
            'n': {'36': 128, '48': 192, '64': 256, '96': 384, '128': 512},
            's': {'36': 96, '48': 144, '64': 192, '96': 288, '128': 384},
            'm': {'36': 64, '48': 96, '64': 128, '96': 192, '128': 256},
            'l': {'36': 48, '48': 72, '64': 96, '96': 144, '128': 192},
            'x': {'36': 32, '48': 48, '64': 64, '96': 96, '128': 128}
        }
        
        model_size = model_name.split('yolov8')[1][0]
        
        # 메모리 크기별 배치 사이즈
        if memory_gb >= 128:
            mem_key = '128'
        elif memory_gb >= 96:
            mem_key = '96'
        elif memory_gb >= 64:
            mem_key = '64'
        elif memory_gb >= 48:
            mem_key = '48'
        else:
            mem_key = '36'
        
        return batch_sizes.get(model_size, batch_sizes['m']).get(mem_key, 32)
    
    def train_model(self, data_yaml, model_name='yolov8x.pt', epochs=300, memory_gb=36):
        """M3 Max 최적화 학습"""
        print(f"\n🚀 Starting M3 Max optimized training...")
        print(f"   Model: {model_name}")
        print(f"   Epochs: {epochs}")
        
        # 모델 로드
        model = YOLO(model_name)
        
        # M3 Max 최적 배치 크기
        batch_size = self.get_optimal_batch_size(model_name, memory_gb)
        print(f"   Batch size: {batch_size} (optimized for {memory_gb}GB)")
        
        # M3 Max 최적화 학습 설정
        training_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': 800,  # 더 큰 이미지 사용 가능!
            'batch': batch_size,
            'device': 'mps' if self.is_apple_silicon else self.device,
            'workers': 8,  # M3 Max는 더 많은 워커 사용 가능
            'patience': 50,
            'save': True,
            'save_period': 10,
            'cache': 'ram',  # RAM에 전체 데이터셋 캐싱
            'amp': False,  # MPS에서는 AMP 사용 안함
            'optimizer': 'AdamW',
            'lr0': 0.001,
            'lrf': 0.0001,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'project': 'runs/football_m3max',
            'name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': False,
            'pretrained': True,
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'single_cls': False,
            'rect': False,
            'cos_lr': True,
            'label_smoothing': 0.1,
            'dropout': 0.1,
            'val': True,
            'plots': True,
            'overlap_mask': True,
            'mask_ratio': 4,
            'max_det': 300,
            
            # 고급 증강 (M3 Max는 더 많은 증강 가능)
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 15,
            'translate': 0.2,
            'scale': 0.9,
            'shear': 5.0,
            'perspective': 0.0005,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.3,
            'copy_paste': 0.3,
            'close_mosaic': 30,  # 마지막 30 에폭
        }
        
        print("\n📊 Training configuration:")
        print(f"   Image size: {training_args['imgsz']}")
        print(f"   Batch size: {training_args['batch']}")
        print(f"   Workers: {training_args['workers']}")
        print(f"   Cache: {training_args['cache']}")
        
        # 학습 시작
        results = model.train(**training_args)
        
        print("\n✅ Training completed!")
        
        # 모델 경로
        best_model_path = os.path.join(
            'runs/football_m3max', 
            training_args['name'], 
            'weights', 
            'best.pt'
        )
        
        print(f"📊 Best model: {best_model_path}")
        
        # 모델을 ONNX로도 내보내기 (크로스 플랫폼용)
        print("\n📤 Exporting model for cross-platform use...")
        model = YOLO(best_model_path)
        
        # 다양한 형식으로 내보내기
        model.export(format='onnx', imgsz=640, half=False)
        model.export(format='torchscript', imgsz=640)
        
        print("✅ Export completed!")
        
        return best_model_path, results


def main():
    print("="*70)
    print("🍎 M3 Max 최적화 미식축구 모델 학습")
    print("="*70)
    
    trainer = M3MaxTrainer()
    
    # 메모리 확인
    memory_gb = int(input("\nM3 Max 메모리 크기 (GB, 예: 36, 48, 64, 96, 128): ") or "36")
    
    # 데이터셋 경로
    data_yaml = input("데이터셋 YAML 경로 (기본: Football-Players-1/data.yaml): ").strip()
    if not data_yaml:
        data_yaml = "Football-Players-1/data.yaml"
    
    # 모델 선택
    print("\n모델 크기 선택 (M3 Max는 큰 모델도 빠르게 학습!):")
    print("1. YOLOv8n (가장 빠름)")
    print("2. YOLOv8s (빠름)")
    print("3. YOLOv8m (균형)")
    print("4. YOLOv8l (정확)")
    print("5. YOLOv8x (가장 정확) ⭐ 추천")
    
    model_choice = input("\n선택 (1-5, 기본값 5): ").strip() or "5"
    model_sizes = ['n', 's', 'm', 'l', 'x']
    model_name = f"yolov8{model_sizes[int(model_choice)-1]}.pt"
    
    # 에폭 수
    epochs = int(input("\n학습 에폭 수 (M3 Max 추천: 300): ") or "300")
    
    # 학습 시작
    best_model, results = trainer.train_model(data_yaml, model_name, epochs, memory_gb)
    
    print("\n" + "="*70)
    print("🎉 학습 완료!")
    print("="*70)
    print("\n📋 다음 단계:")
    print("1. 학습된 모델을 GitHub에 푸시:")
    print("   git add runs/")
    print("   git commit -m 'Add trained model'")
    print("   git push")
    print("\n2. Windows에서 pull 후 사용:")
    print("   git pull")
    print(f"   model_path = '{best_model}'")
    print("\n3. 또는 ONNX 모델 사용 (크로스 플랫폼):")
    print(f"   model_path = '{best_model.replace('.pt', '.onnx')}'")


if __name__ == "__main__":
    main()