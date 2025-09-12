# download_dataset.py
from roboflow import Roboflow
import os

# Roboflow API 키 입력
API_KEY = "YOUR_API_KEY"  # 여기에 실제 API 키 입력!

print("📦 Downloading American Football dataset...")
rf = Roboflow(api_key=E5R556UHuay9v6sEftoY)

# 미식축구 데이터셋 다운로드
project = rf.workspace("bronkscottema").project("football-players-zm06l")
dataset = project.version(1).download("yolov8", location="./data/football-dataset")

print(f"✅ Dataset downloaded to: {dataset.location}")