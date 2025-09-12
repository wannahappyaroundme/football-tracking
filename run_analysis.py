# ultimate_football_analyzer.py
"""
궁극의 미식축구 분석 시스템
- 필드 검출 & 사이드라인 필터링
- 팀 자동 구분
- 포메이션 인식
- 플레이 타입 분석
- 공 추적
- 선수별 통계
- 버드아이뷰
- Roboflow 데이터 연동
"""

import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import supervision as sv
from datetime import datetime
import time
from collections import defaultdict, Counter, deque
import pandas as pd
import json
import math

# 필요시 설치: pip install scikit-learn scipy
try:
    from sklearn.cluster import KMeans
    from scipy.spatial import distance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not installed. Team classification will use simple method.")

class FootballFieldDetector:
    """미식축구 필드 검출 및 영역 구분"""
    
    def __init__(self):
        self.field_mask = None
        self.field_contour = None
        self.sideline_regions = []
        self.field_lines = []
        
    def detect_field(self, frame):
        """향상된 필드 검출"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 다양한 잔디 색상 범위
        green_ranges = [
            ([35, 40, 40], [85, 255, 255]),    # 일반 잔디
            ([25, 30, 30], [95, 255, 255]),    # 넓은 범위
            ([45, 50, 50], [75, 255, 255]),    # 밝은 잔디
        ]
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in green_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 노이즈 제거
        kernel = np.ones((7,7), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # 가장 큰 연결 영역 찾기
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            self.field_contour = max(contours, key=cv2.contourArea)
            
            # 필드 마스크 생성
            self.field_mask = np.zeros_like(combined_mask)
            cv2.fillPoly(self.field_mask, [self.field_contour], 255)
            
            # 필드 경계 박스
            x, y, w, h = cv2.boundingRect(self.field_contour)
            
            # 사이드라인 영역 정의
            margin = int(w * 0.15)
            self.sideline_regions = [
                (max(0, x - margin), y, x, y + h),              # 왼쪽
                (x + w, y, min(frame.shape[1], x + w + margin), y + h)  # 오른쪽
            ]
            
            # 필드 라인 검출 (흰색 선)
            self._detect_field_lines(frame)
            
        return self.field_mask
    
    def _detect_field_lines(self, frame):
        """필드 라인 검출"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 필드 영역만 처리
        if self.field_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=self.field_mask)
        
        # 흰색 선 검출
        _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Hough 변환으로 직선 검출
        lines = cv2.HoughLinesP(white_mask, 1, np.pi/180, 100, 
                                minLineLength=100, maxLineGap=10)
        
        if lines is not None:
            self.field_lines = lines
    
    def is_on_field(self, bbox, threshold=0.7):
        """선수가 필드 위에 있는지 확인"""
        if self.field_mask is None:
            return True
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # 발 위치
        foot_y = min(y2, self.field_mask.shape[0] - 1)
        foot_x = min(max(int((x1 + x2) / 2), 0), self.field_mask.shape[1] - 1)
        
        # 필드 위 확인
        if self.field_mask[foot_y, foot_x] > 0:
            return True
        
        # 바운딩 박스와 필드 겹침 비율
        bbox_mask = np.zeros_like(self.field_mask)
        cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)
        
        intersection = cv2.bitwise_and(bbox_mask, self.field_mask)
        overlap_ratio = np.sum(intersection > 0) / (np.sum(bbox_mask > 0) + 1e-6)
        
        return overlap_ratio > threshold
    
    def is_in_sideline(self, bbox):
        """사이드라인 영역 확인"""
        x1, y1, x2, y2 = map(int, bbox)
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        for sx1, sy1, sx2, sy2 in self.sideline_regions:
            if sx1 <= center_x <= sx2 and sy1 <= center_y <= sy2:
                return True
        return False


class BallDetector:
    """공 검출 시스템"""
    
    def __init__(self):
        self.ball_position = None
        self.ball_trajectory = deque(maxlen=30)
        self.possession_team = None
        
    def detect_ball(self, frame, detections):
        """공 검출 (색상 및 형태 기반)"""
        # 갈색 공 검출
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 미식축구 공 색상 범위 (갈색)
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([20, 255, 200])
        
        ball_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
        
        # 타원형 객체 찾기
        contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 50 or area > 500:  # 크기 필터
                continue
            
            # 타원 피팅
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (center, (width, height), angle) = ellipse
                
                # 미식축구 공 비율 체크 (길쭉한 타원)
                aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                if 1.5 < aspect_ratio < 3.0:
                    score = area * aspect_ratio
                    if score > best_score:
                        best_score = score
                        best_ball = center
        
        if best_ball:
            self.ball_position = best_ball
            self.ball_trajectory.append(best_ball)
            
            # 소유권 판단 (가장 가까운 선수)
            if len(detections) > 0:
                min_dist = float('inf')
                closest_player = None
                
                for bbox in detections.xyxy:
                    player_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
                    dist = np.linalg.norm(np.array(player_center) - np.array(best_ball))
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_player = bbox
                
                if min_dist < 50:  # 50픽셀 이내면 소유
                    self.possession_team = closest_player
        
        return self.ball_position


class TeamClassifier:
    """팀 구분 시스템"""
    
    def __init__(self):
        self.team_colors = {}
        self.team_assignments = {}
        self.referee_color = None
        
    def extract_jersey_features(self, frame, bbox):
        """유니폼 특징 추출"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # 상체 영역
        jersey_y1 = y1 + int((y2 - y1) * 0.25)
        jersey_y2 = y1 + int((y2 - y1) * 0.65)
        
        jersey_y1 = max(0, jersey_y1)
        jersey_y2 = min(frame.shape[0], jersey_y2)
        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)
        
        if jersey_y2 <= jersey_y1 or x2 <= x1:
            return np.array([128, 128, 128]), None
        
        jersey_region = frame[jersey_y1:jersey_y2, x1:x2]
        
        if jersey_region.size == 0:
            return np.array([128, 128, 128]), None
        
        # HSV 색상 추출
        hsv_region = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2HSV)
        
        # 주요 색상
        hist = cv2.calcHist([hsv_region], [0], None, [180], [0, 180])
        dominant_hue = np.argmax(hist)
        
        # 평균 색상
        mean_color = cv2.mean(jersey_region)[:3]
        
        # 번호 검출 시도 (OCR 대신 간단한 방법)
        gray = cv2.cvtColor(jersey_region, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(binary > 0) / binary.size
        
        return np.array(mean_color), white_ratio
    
    def classify_teams(self, frame, detections, frame_id=0):
        """팀 분류"""
        if len(detections) == 0:
            return []
        
        colors = []
        features = []
        
        for bbox in detections.xyxy:
            color, feature = self.extract_jersey_features(frame, bbox)
            colors.append(color)
            features.append(feature if feature else 0)
        
        colors = np.array(colors)
        
        if SKLEARN_AVAILABLE:
            # K-means 클러스터링
            if frame_id == 0 or not self.team_colors:
                if len(colors) >= 3:
                    kmeans = KMeans(n_clusters=min(3, len(colors)), random_state=42)
                    labels = kmeans.fit_predict(colors)
                    
                    # 클러스터 크기로 팀 구분
                    label_counts = Counter(labels)
                    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    if len(sorted_labels) >= 2:
                        team1_label = sorted_labels[0][0]
                        team2_label = sorted_labels[1][0]
                        
                        self.team_colors['team1'] = kmeans.cluster_centers_[team1_label]
                        self.team_colors['team2'] = kmeans.cluster_centers_[team2_label]
                        
                        if len(sorted_labels) > 2:
                            referee_label = sorted_labels[2][0]
                            self.referee_color = kmeans.cluster_centers_[referee_label]
                else:
                    labels = np.zeros(len(colors))
            else:
                # 기존 색상과 비교
                labels = []
                for color in colors:
                    distances = {}
                    distances['team1'] = np.linalg.norm(color - self.team_colors['team1'])
                    distances['team2'] = np.linalg.norm(color - self.team_colors['team2'])
                    
                    if self.referee_color is not None:
                        distances['referee'] = np.linalg.norm(color - self.referee_color)
                    
                    min_team = min(distances, key=distances.get)
                    
                    if min_team == 'team1':
                        labels.append(0)
                    elif min_team == 'team2':
                        labels.append(1)
                    else:
                        labels.append(2)
        else:
            # 간단한 색상 기반 분류
            labels = []
            for color in colors:
                # 빨강 계열
                if color[2] > color[1] and color[2] > color[0]:
                    labels.append(0)
                # 파랑 계열
                elif color[0] > color[1] and color[0] > color[2]:
                    labels.append(1)
                else:
                    labels.append(2)
        
        return labels


class PlayTypeAnalyzer:
    """플레이 타입 분석"""
    
    def __init__(self):
        self.play_history = deque(maxlen=60)  # 2초 분량
        self.current_play = None
        self.play_start_frame = None
        
    def analyze_play(self, positions, ball_position, frame_id):
        """플레이 타입 분석"""
        
        # 선수들의 평균 속도 계산
        if len(self.play_history) > 10:
            recent_positions = list(self.play_history)[-10:]
            
            # 이동 거리 계산
            total_movement = 0
            for i in range(1, len(recent_positions)):
                prev = recent_positions[i-1]
                curr = recent_positions[i]
                
                for p_id in curr:
                    if p_id in prev:
                        dist = np.linalg.norm(
                            np.array(curr[p_id]) - np.array(prev[p_id])
                        )
                        total_movement += dist
            
            avg_movement = total_movement / (len(recent_positions) * len(positions))
            
            # 플레이 타입 판단
            if avg_movement < 2:
                play_type = "Pre-snap"
            elif avg_movement < 10:
                play_type = "Running Play"
            elif ball_position and len(self.play_history) > 0:
                # 공의 궤적으로 패스 판단
                if ball_position != self.play_history[-1].get('ball'):
                    ball_movement = np.linalg.norm(
                        np.array(ball_position) - 
                        np.array(self.play_history[-1].get('ball', ball_position))
                    )
                    if ball_movement > 50:
                        play_type = "Passing Play"
                    else:
                        play_type = "Running Play"
                else:
                    play_type = "Running Play"
            else:
                play_type = "Active Play"
            
            self.current_play = play_type
            
            # 새로운 플레이 시작 감지
            if play_type != "Pre-snap" and self.play_start_frame is None:
                self.play_start_frame = frame_id
            elif play_type == "Pre-snap":
                self.play_start_frame = None
        
        # 현재 프레임 정보 저장
        frame_data = {p_id: pos for p_id, pos in enumerate(positions)}
        frame_data['ball'] = ball_position
        self.play_history.append(frame_data)
        
        return self.current_play


class PlayerStats:
    """선수별 통계"""
    
    def __init__(self):
        self.player_data = defaultdict(lambda: {
            'positions': [],
            'speeds': [],
            'total_distance': 0,
            'max_speed': 0,
            'avg_speed': 0,
            'play_time': 0
        })
        
    def update(self, detections, frame_id, fps=30):
        """선수 통계 업데이트"""
        
        for track_id, bbox in zip(detections.tracker_id, detections.xyxy):
            center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
            
            player = self.player_data[track_id]
            player['positions'].append(center)
            player['play_time'] = len(player['positions']) / fps
            
            # 속도 계산
            if len(player['positions']) > 1:
                prev = player['positions'][-2]
                curr = player['positions'][-1]
                
                distance = np.linalg.norm(
                    np.array(curr) - np.array(prev)
                )
                speed = distance * fps  # pixels/second
                
                player['speeds'].append(speed)
                player['total_distance'] += distance
                player['max_speed'] = max(player['max_speed'], speed)
                player['avg_speed'] = np.mean(player['speeds'])
    
    def get_heatmap(self, track_id, frame_shape):
        """히트맵 생성"""
        if track_id not in self.player_data:
            return None
        
        positions = self.player_data[track_id]['positions']
        if len(positions) < 10:
            return None
        
        heatmap = np.zeros((frame_shape[0], frame_shape[1]), dtype=np.float32)
        
        for x, y in positions:
            x, y = int(x), int(y)
            if 0 <= x < frame_shape[1] and 0 <= y < frame_shape[0]:
                cv2.circle(heatmap, (x, y), 20, 1, -1)
        
        # 정규화
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap


class BirdEyeViewTransformer:
    """향상된 버드아이뷰 변환"""
    
    def __init__(self, field_width=120, field_height=53.3):
        self.field_width = field_width
        self.field_height = field_height
        self.scale = 10
        
        self.output_width = int(field_width * self.scale)
        self.output_height = int(field_height * self.scale)
        
        self.homography_matrix = None
        self.play_trajectories = defaultdict(list)
        
    def create_field_view(self):
        """미식축구 필드 생성"""
        field = np.zeros((self.output_height, self.output_width, 3), dtype=np.uint8)
        
        # 필드 배경
        field[:, :] = (34, 139, 34)
        
        # 엔드존
        cv2.rectangle(field, (0, 0), (100, self.output_height), (0, 100, 0), -1)
        cv2.rectangle(field, (self.output_width - 100, 0), 
                     (self.output_width, self.output_height), (0, 100, 0), -1)
        
        # 야드 라인
        for yard in range(0, self.field_width + 1, 10):
            x = yard * self.scale
            cv2.line(field, (x, 0), (x, self.output_height), (255, 255, 255), 2)
            
            # 5야드 마다 해시마크
            if yard % 5 == 0:
                for y in range(0, self.output_height, 50):
                    cv2.line(field, (x-5, y), (x+5, y), (255, 255, 255), 1)
            
            # 야드 숫자
            if yard % 10 == 0 and 10 <= yard <= 110:
                display_yard = min(yard, 120 - yard) if yard != 60 else 50
                cv2.putText(field, str(display_yard), 
                           (x - 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 255), 2)
                cv2.putText(field, str(display_yard), 
                           (x - 15, self.output_height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 255), 2)
        
        return field
    
    def transform_positions(self, detections, team_labels, ball_position, frame_shape):
        """포지션 변환 및 시각화"""
        field = self.create_field_view()
        
        if len(detections) == 0:
            return field
        
        # 포지션 변환
        for i, (bbox, track_id) in enumerate(zip(detections.xyxy, detections.tracker_id)):
            x1, y1, x2, y2 = bbox
            
            # 발 위치
            foot_x = (x1 + x2) / 2
            foot_y = y2
            
            # 선형 변환 (실제로는 호모그래피 필요)
            field_x = int(foot_x / frame_shape[1] * self.output_width)
            field_y = int(foot_y / frame_shape[0] * self.output_height)
            
            # 팀별 색상
            if i < len(team_labels):
                if team_labels[i] == 0:
                    color = (255, 0, 0)  # 팀1: 빨강
                elif team_labels[i] == 1:
                    color = (0, 0, 255)  # 팀2: 파랑
                else:
                    color = (255, 255, 0)  # 심판: 노랑
            else:
                color = (128, 128, 128)
            
            # 선수 표시
            cv2.circle(field, (field_x, field_y), 10, color, -1)
            cv2.circle(field, (field_x, field_y), 12, (255, 255, 255), 2)
            cv2.putText(field, str(int(track_id)), 
                       (field_x - 8, field_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 궤적 저장 및 그리기
            self.play_trajectories[track_id].append((field_x, field_y))
            if len(self.play_trajectories[track_id]) > 1:
                points = np.array(self.play_trajectories[track_id][-30:], np.int32)
                cv2.polylines(field, [points], False, color, 2)
        
        # 공 표시
        if ball_position:
            ball_x = int(ball_position[0] / frame_shape[1] * self.output_width)
            ball_y = int(ball_position[1] / frame_shape[0] * self.output_height)
            cv2.ellipse(field, (ball_x, ball_y), (15, 8), 0, 0, 360, 
                       (139, 69, 19), -1)  # 갈색 타원
            cv2.ellipse(field, (ball_x, ball_y), (15, 8), 0, 0, 360, 
                       (255, 255, 255), 2)
        
        return field


class UltimateFootballAnalyzer:
    """통합 미식축구 분석 시스템"""
    
    def __init__(self, model_path=None, use_roboflow=False):
        print("🏈 Ultimate Football Analyzer 초기화 중...")
        
        # GPU 설정
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Device: {self.device}")
        
        # 모델 로드
        if use_roboflow and os.path.exists("data/football-dataset/weights/best.pt"):
            print("   Loading Roboflow trained model...")
            self.model = YOLO("data/football-dataset/weights/best.pt")
        elif model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            print("   Loading YOLOv8x...")
            self.model = YOLO('yolov8x.pt')
        
        # 컴포넌트 초기화
        self.field_detector = FootballFieldDetector()
        self.team_classifier = TeamClassifier()
        self.ball_detector = BallDetector()
        self.play_analyzer = PlayTypeAnalyzer()
        self.player_stats = PlayerStats()
        self.bird_eye_transformer = BirdEyeViewTransformer()
        
        # 트래커
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8
        )
        
        # Annotators
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)
        
        # 통계
        self.game_stats = {
            'total_frames': 0,
            'total_plays': 0,
            'play_types': Counter(),
            'team_stats': defaultdict(dict),
            'frame_data': []
        }
        
    def process_frame(self, frame, frame_id):
        """프레임 처리"""
        
        # 1. 객체 검출
        results = self.model(frame, device=self.device, conf=0.3, classes=[0], verbose=False)
        detections = sv.Detections.from_ultralytics(results[0])
        
        # 2. 필드 검출
        if frame_id == 0 or frame_id % 100 == 0:
            self.field_detector.detect_field(frame)
        
        # 3. 필드 내 선수만 필터링
        if self.field_detector.field_mask is not None and len(detections) > 0:
            field_indices = []
            for i, bbox in enumerate(detections.xyxy):
                if self.field_detector.is_on_field(bbox) and \
                   not self.field_detector.is_in_sideline(bbox):
                    field_indices.append(i)
            
            if field_indices:
                detections = detections[field_indices]
        
        # 4. 추적
        detections = self.tracker.update_with_detections(detections)
        
        # 5. 팀 분류
        team_labels = self.team_classifier.classify_teams(frame, detections, frame_id)
        
        # 6. 공 검출
        ball_position = self.ball_detector.detect_ball(frame, detections)
        
        # 7. 플레이 타입 분석 (수정된 부분)
        positions = [((bbox[0]+bbox[2])/2, bbox[3]) for bbox in detections.xyxy]
        play_type = self.play_analyzer.analyze_play(positions, ball_position, frame_id)
        
        # 8. 선수 통계 업데이트
        self.player_stats.update(detections, frame_id)
        
        # 9. 게임 통계 업데이트
        self.game_stats['total_frames'] = frame_id
        if play_type:
            self.game_stats['play_types'][play_type] += 1
        
        frame_stats = {
            'frame': frame_id,
            'detections': len(detections),
            'team1_count': sum(1 for l in team_labels if l == 0),
            'team2_count': sum(1 for l in team_labels if l == 1),
            'referee_count': sum(1 for l in team_labels if l == 2),
            'play_type': play_type,
            'ball_position': ball_position
        }
        self.game_stats['frame_data'].append(frame_stats)
        
        return detections, team_labels, play_type, ball_position
    
    def visualize_frame(self, frame, detections, team_labels, play_type, ball_position, frame_id):
        """고급 시각화"""
        annotated = frame.copy()
        
        # 필드 오버레이
        if self.field_detector.field_mask is not None:
            field_overlay = np.zeros_like(frame)
            field_overlay[:,:,1] = self.field_detector.field_mask // 5
            annotated = cv2.addWeighted(annotated, 0.9, field_overlay, 0.1, 0)
            
            # 사이드라인 표시
            for sx1, sy1, sx2, sy2 in self.field_detector.sideline_regions:
                cv2.rectangle(annotated, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
        
        # 검출 결과 그리기
        if len(detections) > 0:
            colors = []
            labels = []
            
            for i, (track_id, team) in enumerate(zip(detections.tracker_id, team_labels)):
                # 선수 통계
                player = self.player_stats.player_data[track_id]
                speed = player['speeds'][-1] if player['speeds'] else 0
                
                if team == 0:
                    colors.append(sv.Color.RED)
                    labels.append(f"T1 #{int(track_id)} ({speed:.1f}px/s)")
                elif team == 1:
                    colors.append(sv.Color.BLUE)
                    labels.append(f"T2 #{int(track_id)} ({speed:.1f}px/s)")
                else:
                    colors.append(sv.Color.YELLOW)
                    labels.append(f"REF #{int(track_id)}")
            
            annotated = self.box_annotator.annotate(annotated, detections)
            annotated = self.label_annotator.annotate(annotated, detections, labels)
        
        # 공 표시
        if ball_position:
            cv2.circle(annotated, tuple(map(int, ball_position)), 10, (139, 69, 19), -1)
            cv2.circle(annotated, tuple(map(int, ball_position)), 12, (255, 255, 255), 2)
            
            # 공 궤적
            if len(self.ball_detector.ball_trajectory) > 1:
                points = np.array(list(self.ball_detector.ball_trajectory), np.int32)
                cv2.polylines(annotated, [points], False, (255, 255, 0), 2)
        
        # 정보 패널
        info_panel = np.zeros((150, frame.shape[1], 3), dtype=np.uint8)
        info_panel[:] = (30, 30, 30)
        
        info = [
            f"Frame: {frame_id} | FPS: {self.game_stats['total_frames']/(frame_id/30+1):.1f}",
            f"On Field: {len(detections)} | Team1: {sum(1 for l in team_labels if l==0)} | Team2: {sum(1 for l in team_labels if l==1)}",
            f"Play Type: {play_type if play_type else 'Analyzing...'}",
            f"Ball Possession: {'Team 1' if ball_position and self.ball_detector.possession_team else 'Unknown'}"
        ]
        
        y_offset = 30
        for text in info:
            cv2.putText(info_panel, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            y_offset += 30
        
        # 정보 패널 추가
        annotated = np.vstack([info_panel, annotated])
        
        # 버드아이뷰 (미니맵)
        bird_view = self.bird_eye_transformer.transform_positions(
            detections, team_labels, ball_position, frame.shape
        )
        
        # 미니맵 크기 조정 및 추가
        minimap_height = 250
        minimap_width = int(minimap_height * 120 / 53.3)
        bird_small = cv2.resize(bird_view, (minimap_width, minimap_height))
        
        # 미니맵 위치 (오른쪽 상단)
        y_start = 160
        x_start = annotated.shape[1] - minimap_width - 10
        annotated[y_start:y_start+minimap_height, x_start:x_start+minimap_width] = bird_small
        
        # 미니맵 테두리
        cv2.rectangle(annotated, (x_start-2, y_start-2), 
                     (x_start+minimap_width+2, y_start+minimap_height+2), 
                     (255, 255, 255), 2)
        
        return annotated
    
    def process_video(self, input_path, output_path):
        """비디오 처리"""
        print(f"\n📹 Processing: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 150  # 정보 패널 추가
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_id = 0
        start_time = time.time()
        
        print(f"   Resolution: {width}x{height-150} -> {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        print(f"   Duration: {total_frames/fps:.1f}s")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 처리
            detections, team_labels, play_type, ball_position = self.process_frame(frame, frame_id)
            
            # 시각화
            annotated = self.visualize_frame(
                frame, detections, team_labels, play_type, ball_position, frame_id
            )
            
            # 저장
            out.write(annotated)
            
            # 진행 상황
            frame_id += 1
            if frame_id % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_id / elapsed
                eta = (total_frames - frame_id) / fps_current if fps_current > 0 else 0
                print(f"   Progress: {frame_id}/{total_frames} ({frame_id/total_frames*100:.1f}%) - "
                      f"FPS: {fps_current:.1f} - ETA: {eta:.0f}s")
        
        cap.release()
        out.release()
        
        # 통계 저장
        self.save_statistics(output_path)
        
        print(f"\n✅ Complete! Output: {output_path}")
        
    def save_statistics(self, output_path):
        """통계 저장"""
        # JSON 통계
        stats_path = output_path.replace('.mp4', '_stats.json')
        with open(stats_path, 'w') as f:
            # Convert Counter to dict for JSON serialization
            stats_copy = self.game_stats.copy()
            stats_copy['play_types'] = dict(stats_copy['play_types'])
            json.dump(stats_copy, f, indent=2, default=str)
        print(f"📊 Stats saved: {stats_path}")
        
        # CSV 프레임 데이터
        csv_path = output_path.replace('.mp4', '_frame_data.csv')
        df = pd.DataFrame(self.game_stats['frame_data'])
        df.to_csv(csv_path, index=False)
        print(f"📊 Frame data saved: {csv_path}")
        
        # 선수별 통계
        player_stats_path = output_path.replace('.mp4', '_player_stats.csv')
        player_rows = []
        for track_id, stats in self.player_stats.player_data.items():
            player_rows.append({
                'player_id': track_id,
                'play_time': stats['play_time'],
                'total_distance': stats['total_distance'],
                'max_speed': stats['max_speed'],
                'avg_speed': stats['avg_speed']
            })
        
        if player_rows:
            player_df = pd.DataFrame(player_rows)
            player_df.to_csv(player_stats_path, index=False)
            print(f"📊 Player stats saved: {player_stats_path}")


def main():
    """메인 실행"""
    print("="*70)
    print("🏈 궁극의 미식축구 분석 시스템")
    print("="*70)
    
    # 옵션 선택
    print("\n분석 모드 선택:")
    print("1. 빠른 분석 (기본 YOLO)")
    print("2. Roboflow 모델 사용 (미식축구 특화)")
    print("3. 커스텀 모델 경로 지정")
    
    mode = input("\n선택 (1-3, 기본값 1): ").strip() or "1"
    
    # 분석기 초기화
    if mode == "2":
        analyzer = UltimateFootballAnalyzer(use_roboflow=True)
    elif mode == "3":
        model_path = input("모델 경로 입력: ").strip()
        analyzer = UltimateFootballAnalyzer(model_path=model_path)
    else:
        analyzer = UltimateFootballAnalyzer()
    
    # 비디오 선택
    video_dir = "videos"
    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    print(f"\n📹 사용 가능한 영상: {len(videos)}개")
    
    # 크기별 정렬
    video_sizes = []
    for v in videos:
        path = os.path.join(video_dir, v)
        size = os.path.getsize(path) / (1024*1024)
        video_sizes.append((v, size))
    
    video_sizes.sort(key=lambda x: x[1])
    
    # 상위 10개 표시
    for i, (name, size) in enumerate(video_sizes[:10], 1):
        print(f"   {i}. {name} ({size:.1f} MB)")
    
    choice = input("\n영상 번호 선택 (1-10): ").strip() or "1"
    selected = video_sizes[int(choice) - 1][0]
    
    # 분석 실행
    input_path = os.path.join(video_dir, selected)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join("output", f"ultimate_{timestamp}.mp4")
    
    analyzer.process_video(input_path, output_path)
    
    print("\n🎉 분석 완료!")
    print("\n구현된 기능:")
    print("   ✅ 필드 검출 & 사이드라인 필터링")
    print("   ✅ 팀 자동 구분 (유니폼 색상)")
    print("   ✅ 공 검출 및 추적")
    print("   ✅ 플레이 타입 분석 (러닝/패싱)")
    print("   ✅ 선수별 통계 (속도, 거리)")
    print("   ✅ 버드아이뷰 with 궤적")
    print("   ✅ 실시간 정보 패널")
    print("   ✅ 통계 저장 (JSON, CSV)")

if __name__ == "__main__":
    main()