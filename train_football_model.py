# train_football_model.py
"""
미식축구 전용 YOLO 모델 학습
- Roboflow 데이터셋 다운로드
- YOLOv8 파인튜닝
- 공, 선수, 심판 검출 특화
"""

import os
import torch
from ultralytics import YOLO
from roboflow import Roboflow
import yaml
import shutil
from datetime import datetime

class FootballModelTrainer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️ Device: {self.device}")
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    def download_roboflow_dataset(self, api_key, workspace="bronkscottema", project="football-players-zm06l", version=1):
        """Roboflow 데이터셋 다운로드"""
        print("\n📦 Downloading Roboflow dataset...")
        
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(workspace).project(project)
        dataset = project.version(version).download("yolov8")
        
        print(f"✅ Dataset downloaded to: {dataset.location}")
        return dataset.location
    
    def prepare_dataset_config(self, dataset_path):
        """데이터셋 설정 파일 생성"""
        print("\n📝 Preparing dataset configuration...")
        
        # data.yaml 파일 읽기
        yaml_path = os.path.join(dataset_path, "data.yaml")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # 절대 경로로 수정
        abs_path = os.path.abspath(dataset_path)
        data['path'] = abs_path
        data['train'] = os.path.join(abs_path, 'train', 'images')
        data['val'] = os.path.join(abs_path, 'valid', 'images')
        
        # 테스트 데이터가 있으면 추가
        test_path = os.path.join(abs_path, 'test', 'images')
        if os.path.exists(test_path):
            data['test'] = test_path
        
        # 클래스 정보 확인 및 수정
        print(f"\n📊 Dataset classes:")
        if isinstance(data['names'], dict):
            for idx, class_name in data['names'].items():
                print(f"   {idx}: {class_name}")
        elif isinstance(data['names'], list):
            for idx, class_name in enumerate(data['names']):
                print(f"   {idx}: {class_name}")
            # 리스트를 딕셔너리로 변환
            data['names'] = {i: name for i, name in enumerate(data['names'])}
        
        # 미식축구용 클래스 추가/수정 (필요시)
        # data['names'] = {
        #     0: 'player',
        #     1: 'referee',
        #     2: 'ball',
        #     3: 'helmet'
        # }
        
        # 수정된 yaml 저장
        updated_yaml = os.path.join(dataset_path, "data_football.yaml")
        with open(updated_yaml, 'w') as f:
            yaml.dump(data, f)
        
        print(f"✅ Config saved: {updated_yaml}")
        return updated_yaml
    
    def augment_dataset(self, dataset_path):
        """데이터 증강 설정"""
        print("\n🎨 Setting up data augmentation...")
        
        augmentation_config = {
            'hsv_h': 0.015,  # 색조 변경
            'hsv_s': 0.7,    # 채도 변경
            'hsv_v': 0.4,    # 명도 변경
            'degrees': 5,     # 회전
            'translate': 0.1, # 이동
            'scale': 0.5,     # 스케일
            'shear': 2.0,     # 전단
            'perspective': 0.0001,  # 원근
            'flipud': 0.0,    # 상하 반전 (미식축구는 사용 안함)
            'fliplr': 0.5,    # 좌우 반전
            'mosaic': 1.0,    # 모자이크 증강
            'mixup': 0.2,     # MixUp 증강
            'copy_paste': 0.1 # Copy-Paste 증강
        }
        
        return augmentation_config
    
    def train_model(self, data_yaml, model_name='yolov8x.pt', epochs=100):
        """모델 학습"""
        print(f"\n🚀 Starting training with {model_name}...")
        
        # 모델 로드
        model = YOLO(model_name)
        
        # RTX 3050 최적화 설정
        batch_size = 8 if 'x' in model_name else 16  # x모델은 메모리 많이 사용
        img_size = 640  # 이미지 크기
        
        # 학습 설정
        training_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'device': 0 if self.device == 'cuda' else 'cpu',
            'workers': 4,
            'patience': 20,  # Early stopping
            'save': True,
            'save_period': 10,  # 10 에폭마다 저장
            'cache': True,  # 데이터 캐싱
            'amp': True,  # Mixed precision training
            'optimizer': 'AdamW',
            'lr0': 0.001,  # 초기 학습률
            'lrf': 0.01,   # 최종 학습률
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'box': 7.5,  # Box loss gain
            'cls': 0.5,  # Class loss gain
            'dfl': 1.5,  # DFL loss gain
            'project': 'runs/football',
            'name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': False,
            'pretrained': True,
            'verbose': True,
            'seed': 42,
            'close_mosaic': 10,  # 마지막 10 에폭은 모자이크 끄기
            'resume': False,
            'rect': False,  # 직사각형 학습
            'cos_lr': True,  # Cosine LR scheduler
            'label_smoothing': 0.0,
            'dropout': 0.0,
            'plots': True  # 학습 그래프 생성
        }
        
        # 데이터 증강 추가
        augmentation = self.augment_dataset(data_yaml)
        training_args.update(augmentation)
        
        print("\n📋 Training configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Image size: {img_size}")
        print(f"   Device: {training_args['device']}")
        
        # 학습 시작
        results = model.train(**training_args)
        
        print("\n✅ Training completed!")
        
        # 최고 성능 모델 경로
        best_model_path = os.path.join('runs/football', training_args['name'], 'weights', 'best.pt')
        print(f"📊 Best model saved: {best_model_path}")
        
        return best_model_path, results
    
    def validate_model(self, model_path, data_yaml):
        """모델 검증"""
        print(f"\n🔍 Validating model: {model_path}")
        
        model = YOLO(model_path)
        
        # 검증 실행
        results = model.val(
            data=data_yaml,
            imgsz=640,
            batch=8,
            device=0 if self.device == 'cuda' else 'cpu',
            workers=4,
            save_json=True,
            save_hybrid=True,
            conf=0.001,
            iou=0.6,
            max_det=300,
            half=True,
            dnn=False,
            plots=True,
            rect=True
        )
        
        # 성능 메트릭 출력
        print("\n📊 Validation Results:")
        print(f"   mAP50: {results.box.map50:.3f}")
        print(f"   mAP50-95: {results.box.map:.3f}")
        
        # 클래스별 성능
        for i, class_name in enumerate(results.names.values()):
            if i < len(results.box.ap50):
                print(f"   {class_name}: AP50={results.box.ap50[i]:.3f}")
        
        return results
    
    def export_model(self, model_path, formats=['onnx', 'torchscript']):
        """모델 내보내기"""
        print(f"\n📤 Exporting model to formats: {formats}")
        
        model = YOLO(model_path)
        
        for format_type in formats:
            print(f"   Exporting to {format_type}...")
            model.export(format=format_type, imgsz=640, half=True)
        
        print("✅ Export completed!")
    
    def test_on_video(self, model_path, video_path):
        """비디오로 모델 테스트"""
        print(f"\n🎥 Testing model on video: {video_path}")
        
        model = YOLO(model_path)
        
        # 비디오 처리
        results = model(
            video_path,
            save=True,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            device=0 if self.device == 'cuda' else 'cpu',
            show_labels=True,
            show_conf=True,
            save_txt=False,
            save_crop=False,
            line_width=2,
            visualize=False
        )
        
        print("✅ Video test completed!")
        return results


def main():
    print("="*70)
    print("🏈 미식축구 전용 YOLO 모델 학습")
    print("="*70)
    
    trainer = FootballModelTrainer()
    
    # 옵션 선택
    print("\n학습 옵션 선택:")
    print("1. Roboflow 데이터셋 다운로드 + 학습")
    print("2. 기존 데이터셋으로 학습")
    print("3. 학습된 모델 검증")
    print("4. 모델 테스트 (비디오)")
    
    choice = input("\n선택 (1-4): ").strip()
    
    if choice == "1":
        # Roboflow API 키 입력
        api_key = input("Roboflow API 키 입력: ").strip()
        
        if not api_key:
            print("❌ API 키가 필요합니다!")
            print("👉 https://app.roboflow.com/settings/api 에서 확인하세요")
            return
        
        # 데이터셋 다운로드
        dataset_path = trainer.download_roboflow_dataset(api_key)
        
        # 데이터셋 설정
        data_yaml = trainer.prepare_dataset_config(dataset_path)
        
        # 모델 선택
        print("\n모델 크기 선택:")
        print("1. YOLOv8n (Nano - 가장 빠름)")
        print("2. YOLOv8s (Small - 빠름)")
        print("3. YOLOv8m (Medium - 균형)")
        print("4. YOLOv8l (Large - 정확)")
        print("5. YOLOv8x (Extra Large - 가장 정확)")
        
        model_choice = input("\n선택 (1-5, 기본값 3): ").strip() or "3"
        model_sizes = ['n', 's', 'm', 'l', 'x']
        model_name = f"yolov8{model_sizes[int(model_choice)-1]}.pt"
        
        # 에폭 수 입력
        epochs = input("학습 에폭 수 (기본값 100): ").strip()
        epochs = int(epochs) if epochs else 100
        
        # 학습 시작
        best_model, results = trainer.train_model(data_yaml, model_name, epochs)
        
        # 검증
        trainer.validate_model(best_model, data_yaml)
        
        # 내보내기 옵션
        if input("\n모델을 내보내시겠습니까? (y/n): ").lower() == 'y':
            trainer.export_model(best_model)
        
        print(f"\n🎉 학습 완료!")
        print(f"📁 모델 위치: {best_model}")
        print(f"\n사용 방법:")
        print(f"  analyzer = UltimateFootballAnalyzer(model_path='{best_model}')")
        
    elif choice == "2":
        # 기존 데이터셋으로 학습
        data_yaml = input("데이터셋 YAML 경로: ").strip()
        if not os.path.exists(data_yaml):
            print("❌ 파일을 찾을 수 없습니다!")
            return
        
        model_name = input("기본 모델 (기본값 yolov8m.pt): ").strip() or "yolov8m.pt"
        epochs = int(input("에폭 수 (기본값 100): ").strip() or "100")
        
        best_model, results = trainer.train_model(data_yaml, model_name, epochs)
        trainer.validate_model(best_model, data_yaml)
        
    elif choice == "3":
        # 모델 검증
        model_path = input("모델 경로 (.pt 파일): ").strip()
        data_yaml = input("데이터셋 YAML 경로: ").strip()
        
        if os.path.exists(model_path) and os.path.exists(data_yaml):
            trainer.validate_model(model_path, data_yaml)
        else:
            print("❌ 파일을 찾을 수 없습니다!")
            
    elif choice == "4":
        # 비디오 테스트
        model_path = input("모델 경로 (.pt 파일): ").strip()
        
        if not os.path.exists(model_path):
            # 기본 모델 사용
            model_path = "yolov8x.pt"
            print(f"⚠️ 기본 모델 사용: {model_path}")
        
        # 비디오 선택
        video_dir = "videos"
        videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        
        print(f"\n📹 테스트할 영상 선택:")
        for i, v in enumerate(videos[:10], 1):
            print(f"   {i}. {v}")
        
        video_idx = int(input("번호 선택: ").strip()) - 1
        video_path = os.path.join(video_dir, videos[video_idx])
        
        trainer.test_on_video(model_path, video_path)


if __name__ == "__main__":
    main()