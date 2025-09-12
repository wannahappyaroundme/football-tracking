# quick_test_fixed.py
"""
미식축구 영상 빠른 테스트 스크립트 (에러 수정 버전)
RTX 3050 GPU 사용 최적화
"""

import os
import sys
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from datetime import datetime
import time

# OpenCV GUI 에러 해결
import matplotlib
matplotlib.use('Agg')

def check_environment():
    """환경 체크"""
    print("="*60)
    print("🔍 환경 확인")
    print("="*60)
    
    # GPU 확인
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ GPU를 사용할 수 없습니다.")
        return False
    
    # 비디오 파일 확인
    video_dir = "videos"
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
        print(f"📁 {video_dir} 폴더를 생성했습니다. 영상을 넣어주세요!")
        return False
    
    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov', '.MP4', '.AVI', '.MOV'))]
    if not videos:
        print(f"❌ {video_dir} 폴더에 영상 파일이 없습니다!")
        return False
    
    print(f"\n📹 발견된 영상: {len(videos)}개")
    
    # 처음 10개만 표시
    for i, video in enumerate(videos[:10], 1):
        file_path = os.path.join(video_dir, video)
        size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"   {i}. {video} ({size:.1f} MB)")
    
    if len(videos) > 10:
        print(f"   ... 그 외 {len(videos)-10}개 더")
    
    return videos

def quick_analysis(video_path, output_dir="output", show_preview=False):
    """빠른 분석 실행"""
    print(f"\n🏈 분석 시작: {video_path}")
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 모델 로드 (기본 YOLOv8 사용)
    print("📦 모델 로딩 중...")
    model = YOLO('yolov8x.pt')  # 높은 정확도
    
    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📊 비디오 정보:")
    print(f"   해상도: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   총 프레임: {total_frames}")
    print(f"   길이: {total_frames/fps:.1f}초")
    
    # 출력 비디오 설정
    output_path = os.path.join(output_dir, f"analyzed_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # ByteTrack 초기화
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=30,
        minimum_matching_threshold=0.8
    )
    
    # Annotators 초기화 (새로운 supervision 버전)
    box_annotator = sv.BoxAnnotator(thickness=2, color=sv.ColorPalette.DEFAULT)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    
    # 필드 검출용 변수
    field_mask = None
    
    # 통계
    stats = {
        'total_people': set(),
        'max_people_frame': 0,
        'max_people_count': 0,
        'frame_data': []
    }
    
    print("\n🎬 처리 중...")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # YOLO 검출 (GPU 사용)
            results = model(frame, device=0, conf=0.3, classes=[0], verbose=False)  # person 클래스만
            
            # Supervision 형식 변환
            detections = sv.Detections.from_ultralytics(results[0])
            
            # 필드 영역 검출 (첫 프레임)
            if frame_count == 0 and len(detections) > 0:
                # 간단한 필드 검출 (잔디 색상)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_green = np.array([35, 40, 40])
                upper_green = np.array([85, 255, 255])
                field_mask = cv2.inRange(hsv, lower_green, upper_green)
                
                # 노이즈 제거
                kernel = np.ones((5,5), np.uint8)
                field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
            
            # 필드 내부 선수만 필터링
            if field_mask is not None and len(detections) > 0:
                filtered = []
                for i, bbox in enumerate(detections.xyxy):
                    x1, y1, x2, y2 = bbox
                    foot_x = int((x1 + x2) / 2)
                    foot_y = int(y2)  # 발 위치
                    
                    if 0 <= foot_y < field_mask.shape[0] and \
                       0 <= foot_x < field_mask.shape[1]:
                        if field_mask[foot_y, foot_x] > 0:
                            filtered.append(i)
                
                if filtered:
                    detections = detections[filtered]
            
            # 추적
            detections = tracker.update_with_detections(detections)
            
            # 통계 업데이트
            if len(detections) > 0:
                for track_id in detections.tracker_id:
                    stats['total_people'].add(track_id)
                
                if len(detections) > stats['max_people_count']:
                    stats['max_people_count'] = len(detections)
                    stats['max_people_frame'] = frame_count
            
            # 프레임별 데이터 저장
            stats['frame_data'].append({
                'frame': frame_count,
                'detected': len(detections),
                'ids': list(detections.tracker_id) if len(detections) > 0 else []
            })
            
            # 시각화
            annotated_frame = frame.copy()
            
            # 바운딩 박스와 라벨 그리기
            if len(detections) > 0:
                # 박스 그리기
                annotated_frame = box_annotator.annotate(annotated_frame, detections)
                
                # ID 라벨
                labels = [f"Player #{int(tid)}" for tid in detections.tracker_id]
                annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            
            # 정보 표시
            info = [
                f"Frame: {frame_count}/{total_frames}",
                f"Detected: {len(detections)}",
                f"Total Tracked: {len(stats['total_people'])}"
            ]
            
            y_offset = 30
            for text in info:
                cv2.putText(annotated_frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 25
            
            # 비디오 저장
            out.write(annotated_frame)
            
            # 미리보기 (옵션 - OpenCV GUI 문제로 기본 비활성화)
            if show_preview:
                try:
                    preview = cv2.resize(annotated_frame, (1280, 720))
                    cv2.imshow('Football Analysis', preview)
                    
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        print("\n⚠️ 사용자가 중단했습니다.")
                        break
                except:
                    show_preview = False  # GUI 에러시 비활성화
            
            # 진행 상황
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                eta = (total_frames - frame_count) / fps_current if fps_current > 0 else 0
                print(f"   {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%) - "
                      f"FPS: {fps_current:.1f} - ETA: {eta:.0f}초")
    
    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
    
    finally:
        cap.release()
        out.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass  # OpenCV GUI 에러 무시
    
    # 결과 출력
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("✅ 분석 완료!")
    print("="*60)
    print(f"📊 결과:")
    print(f"   처리 시간: {total_time:.1f}초")
    print(f"   평균 FPS: {frame_count/total_time:.1f}")
    print(f"   추적된 총 인원: {len(stats['total_people'])}명")
    print(f"   최대 동시 감지: {stats['max_people_count']}명 (프레임 {stats['max_people_frame']})")
    print(f"\n💾 저장 위치: {output_path}")
    
    # CSV로 통계 저장
    import pandas as pd
    df = pd.DataFrame(stats['frame_data'])
    csv_path = output_path.replace('.mp4', '_stats.csv')
    df.to_csv(csv_path, index=False)
    print(f"📊 통계 파일: {csv_path}")
    
    return output_path

def main():
    """메인 실행"""
    print("="*60)
    print("🏈 미식축구 영상 분석 - 빠른 테스트")
    print("="*60)
    
    # 환경 확인
    videos = check_environment()
    if not videos:
        return
    
    # 영상 선택
    print("\n📌 추천 영상 (짧은 것부터):")
    # 크기 순으로 정렬
    video_sizes = []
    for v in videos:
        path = os.path.join("videos", v)
        size = os.path.getsize(path) / (1024*1024)
        video_sizes.append((v, size))
    
    video_sizes.sort(key=lambda x: x[1])
    
    # 작은 영상 5개 추천
    for i, (name, size) in enumerate(video_sizes[:5], 1):
        print(f"   {i}. {name} ({size:.1f} MB)")
    
    print("\n번호를 입력하세요 (1-5 추천, 전체 목록은 1-50): ", end='')
    choice = input().strip()
    
    if not choice:
        choice = "1"
    
    try:
        idx = int(choice) - 1
        if idx < 5:
            selected = video_sizes[idx][0]
        else:
            selected = videos[idx]
    except:
        print("❌ 잘못된 선택입니다.")
        return
    
    # 분석 실행
    video_path = os.path.join("videos", selected)
    output_path = quick_analysis(video_path, show_preview=False)  # GUI 비활성화
    
    print("\n🎉 완료! 다음 단계:")
    print("   1. 결과 영상 확인: output 폴더")
    print("   2. 미식축구 전용 모델 학습 (Roboflow)")
    print("   3. 팀 구분 기능 추가")
    print("   4. 버드아이뷰 변환")

if __name__ == "__main__":
    main()