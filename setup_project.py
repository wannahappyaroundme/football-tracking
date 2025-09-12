# setup_project.py
import os

# 프로젝트 디렉토리 생성
base_dir = r"C:\Users\경쓰북\Desktop\Stech\football-tracking"
directories = [
    "videos",      # 입력 영상
    "output",      # 출력 결과
    "models",      # 학습된 모델
    "data",        # 데이터셋
    "src",         # 소스 코드
    "configs"      # 설정 파일
]

for dir_name in directories:
    os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)
    print(f"✅ Created: {dir_name}/")

print("\n📂 프로젝트 구조 생성 완료!")
print("👉 영상 파일을 videos/ 폴더에 넣어주세요!")