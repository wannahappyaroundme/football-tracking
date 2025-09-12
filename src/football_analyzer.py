# src/football_analyzer.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from datetime import datetime
import time
from collections import defaultdict
import pandas as pd

class FootballAnalyzer:
    def __init__(self, model_path=None, use_gpu=True):
        """
        RTX 3050 최적화 미식축구 분석기
        """
        # GPU 설정
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        print(f"🎮 Using device: {self.device}")
        
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 모델 로드
        if model_path and os.path.exists(model_path):
            print(f"📦 Loading custom model: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("📦 Loading YOLOv8x model...")
            self.model = YOLO('yolov8x.pt')  # RTX 3050은 x 모델도 처리 가능
        
        # 트래커 초기화
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        
        # 필드 검출기
        self.field_mask = None
        
        # 추적 데이터
        self.tracking_data = defaultdict(list)
        
    def detect_field(self, frame):
        """잔디 필드 영역 검출"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 잔디 색상 범위 (조명에 따라 조정 필요)
        lower_green = np.array([35, 30, 30])
        upper_green = np.array([85, 255, 255])
        
        # 마스크 생성
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 노이즈 제거
        kernel = np.ones((7,7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 가장 큰 연결 영역 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [largest], 255)
        
        self.field_mask = mask
        return mask
    
    def filter_field_objects(self, detections):
        """필드 내부 객체만 필터링"""
        if self.field_mask is None or len(detections) == 0:
            return detections
        
        filtered_indices = []
        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = bbox
            
            # 바운딩 박스 하단 중앙점 (발 위치)
            foot_x = int((x1 + x2) / 2)
            foot_y = int(y2)  # 하단
            
            # 필드 내부 체크
            if 0 <= foot_y < self.field_mask.shape[0] and \
               0 <= foot_x < self.field_mask.shape[1]:
                if self.field_mask[foot_y, foot_x] > 0:
                    filtered_indices.append(i)
        
        return detections[filtered_indices]
    
    def classify_teams(self, frame, detections):
        """팀 구분 (유니폼 색상 기반)"""
        if len(detections) == 0:
            return []
        
        team_colors = []
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            
            # 상체 영역 추출 (유니폼)
            jersey_y1 = y1 + int((y2-y1) * 0.2)  # 상단 20% 제외 (헬멧)
            jersey_y2 = y1 + int((y2-y1) * 0.6)  # 상체 40%
            
            if jersey_y1 < jersey_y2 and x1 < x2:
                jersey_region = frame[jersey_y1:jersey_y2, x1:x2]
                
                # 주요 색상 추출
                if jersey_region.size > 0:
                    avg_color = cv2.mean(jersey_region)[:3]
                    team_colors.append(avg_color)
                else:
                    team_colors.append((128, 128, 128))  # 기본 회색
            else:
                team_colors.append((128, 128, 128))
        
        # K-means로 2개 팀 구분 (간단한 버전)
        # 실제로는 더 정교한 클러스터링 필요
        return team_colors
    
    def process_frame(self, frame, frame_id):
        """단일 프레임 처리"""
        # 1. 객체 검출 (GPU 사용)
        results = self.model(frame, device=self.device, conf=0.3, iou=0.5)
        detections = sv.Detections.from_ultralytics(results[0])
        
        # 2. 필드 검출 (첫 프레임 또는 주기적)
        if frame_id == 0 or frame_id % 100 == 0:
            self.detect_field(frame)
        
        # 3. 필드 내부 객체만 필터링
        field_detections = self.filter_field_objects(detections)
        
        # 4. 객체 추적
        tracked_detections = self.tracker.update_with_detections(field_detections)
        
        # 5. 팀 분류
        team_colors = self.classify_teams(frame, tracked_detections)
        
        # 6. 데이터 저장
        for i, (bbox, track_id) in enumerate(zip(tracked_detections.xyxy, tracked_detections.tracker_id)):
            x1, y1, x2, y2 = bbox
            self.tracking_data[track_id].append({
                'frame': frame_id,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'center': [float((x1+x2)/2), float((y1+y2)/2)],
                'team_color': team_colors[i] if i < len(team_colors) else None
            })
        
        return tracked_detections, team_colors
    
    def visualize_frame(self, frame, detections, team_colors, frame_id):
        """프레임 시각화"""
        annotated = frame.copy()
        
        # 필드 영역 오버레이
        if self.field_mask is not None:
            field_overlay = np.zeros_like(frame)
            field_overlay[:,:,1] = self.field_mask // 4  # 녹색 반투명
            annotated = cv2.addWeighted(annotated, 0.85, field_overlay, 0.15, 0)
        
        # 검출 결과 그리기
        if len(detections) > 0:
            # 경계 상자 및 ID
            labels = [f"ID:{tid}" for tid in detections.tracker_id]
            annotated = sv.BoundingBoxAnnotator().annotate(annotated, detections)
            annotated = sv.LabelAnnotator().annotate(annotated, detections, labels)
            
            # 추적 경로
            for track_id in self.tracking_data:
                points = []
                for data in self.tracking_data[track_id][-30:]:  # 최근 30프레임
                    if data['frame'] <= frame_id:
                        points.append(data['center'])
                
                if len(points) > 1:
                    points = np.array(points, dtype=np.int32)
                    cv2.polylines(annotated, [points], False, (0, 255, 0), 2)
        
        # 정보 표시
        info_text = [
            f"Frame: {frame_id}",
            f"Detected: {len(detections)}",
            f"FPS: {self.fps:.1f}" if hasattr(self, 'fps') else "FPS: --"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(annotated, text, (10, 30 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return annotated

class VideoProcessor:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        
    def process_video(self, input_path, output_path=None, show_preview=False):
        """비디오 전체 처리"""
        print(f"\n📹 Processing video: {input_path}")
        
        # 비디오 열기
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # 출력 비디오 설정
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 처리 시작
        frame_id = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 프레임 처리
                frame_start = time.time()
                detections, team_colors = self.analyzer.process_frame(frame, frame_id)
                
                # FPS 계산
                self.analyzer.fps = 1 / (time.time() - frame_start)
                
                # 시각화
                annotated = self.analyzer.visualize_frame(frame, detections, team_colors, frame_id)
                
                # 저장
                if out:
                    out.write(annotated)
                
                # 미리보기
                if show_preview:
                    cv2.imshow('Football Analysis', cv2.resize(annotated, (1280, 720)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # 진행 상황
                frame_id += 1
                if frame_id % 30 == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / frame_id) * (total_frames - frame_id)
                    print(f"   Progress: {frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%) - ETA: {eta:.0f}s")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
        
        # 완료
        total_time = time.time() - start_time
        print(f"\n✅ Processing complete!")
        print(f"   Time: {total_time:.1f}s")
        print(f"   Avg FPS: {frame_id/total_time:.1f}")
        print(f"   Tracked objects: {len(self.analyzer.tracking_data)}")
        
        return self.analyzer.tracking_data

# 메인 실행 함수
def main():
    import os
    
    # 경로 설정
    base_dir = r"C:\Users\경쓰북\Desktop\Stech\football-tracking"
    input_video = os.path.join(base_dir, "videos", "sample.mp4")  # 여기에 영상 파일명
    output_video = os.path.join(base_dir, "output", f"analyzed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    
    # 분석기 초기화
    print("🏈 Initializing Football Analyzer...")
    analyzer = FootballAnalyzer(use_gpu=True)
    
    # 비디오 처리
    processor = VideoProcessor(analyzer)
    tracking_data = processor.process_video(
        input_video, 
        output_video,
        show_preview=True  # 실시간 미리보기
    )
    
    # 결과 저장
    save_results(tracking_data, base_dir)

def save_results(tracking_data, base_dir):
    """추적 결과 저장"""
    # DataFrame 생성
    rows = []
    for track_id, frames in tracking_data.items():
        for frame_data in frames:
            rows.append({
                'track_id': track_id,
                'frame': frame_data['frame'],
                'x1': frame_data['bbox'][0],
                'y1': frame_data['bbox'][1],
                'x2': frame_data['bbox'][2],
                'y2': frame_data['bbox'][3],
                'center_x': frame_data['center'][0],
                'center_y': frame_data['center'][1]
            })
    
    df = pd.DataFrame(rows)
    
    # CSV 저장
    output_file = os.path.join(base_dir, "output", f"tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(output_file, index=False)
    print(f"📊 Tracking data saved: {output_file}")
    
    # 통계 출력
    print("\n📈 Statistics:")
    print(f"   Total tracks: {df['track_id'].nunique()}")
    print(f"   Total detections: {len(df)}")
    print(f"   Avg track length: {len(df) / df['track_id'].nunique():.1f} frames")

if __name__ == "__main__":
    main()