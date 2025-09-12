# check_gpu.py
import torch
import sys
import platform

print("="*50)
print("🖥️  시스템 정보")
print("="*50)
print(f"Python 버전: {sys.version}")
print(f"운영체제: {platform.system()} {platform.release()}")
print()

print("="*50)
print("🎮 GPU 정보")
print("="*50)

# CUDA 사용 가능 여부
cuda_available = torch.cuda.is_available()
print(f"CUDA 사용 가능: {cuda_available}")

if cuda_available:
    # CUDA 버전
    print(f"CUDA 버전: {torch.version.cuda}")
    
    # GPU 개수
    gpu_count = torch.cuda.device_count()
    print(f"사용 가능한 GPU 개수: {gpu_count}")
    
    # 각 GPU 정보
    for i in range(gpu_count):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - 메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  - 현재 사용중: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  - 캐시: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    
    # 현재 GPU
    print(f"\n현재 기본 GPU: {torch.cuda.current_device()}")
else:
    print("\n⚠️ GPU를 사용할 수 없습니다!")
    print("다음을 확인해주세요:")
    print("1. NVIDIA GPU가 설치되어 있는지")
    print("2. CUDA가 설치되어 있는지")
    print("3. PyTorch GPU 버전이 설치되어 있는지")

# PyTorch 설치 확인
print("\n" + "="*50)
print("📦 PyTorch 설치 정보")
print("="*50)
print(f"PyTorch 버전: {torch.__version__}")
print(f"cuDNN 활성화: {torch.backends.cudnn.enabled if cuda_available else 'N/A'}")