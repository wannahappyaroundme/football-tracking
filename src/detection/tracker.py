# src/tracking/tracker.py
import numpy as np
from collections import defaultdict

class FootballTracker:
    def __init__(self):
        # ByteTrack 파라미터
        self.track_thresh = 0.5
        self.match_thresh = 0.8
        self.tracks = defaultdict(list)
        self.track_id_counter = 0
        
    def update(self, detections, frame_id):
        """각 프레임의 검출 결과를 추적"""
        # 선수만 추적 (공과 심판은 별도 처리)
        player_dets = [d for d in detections if d['class'] == 'player']
        
        if frame_id == 0:
            # 첫 프레임: 모든 검출에 ID 할당
            for det in player_dets:
                det['track_id'] = self.track_id_counter
                self.tracks[self.track_id_counter].append({
                    'frame': frame_id,
                    'bbox': det['bbox'],
                    'confidence': det['confidence']
                })
                self.track_id_counter += 1
        else:
            # 이후 프레임: 매칭 수행
            self._match_tracks(player_dets, frame_id)
            
        return player_dets
    
    def _match_tracks(self, detections, frame_id):
        """IoU 기반 간단한 매칭 (ByteTrack 간소화 버전)"""
        # 이전 프레임의 트랙 위치
        prev_boxes = []
        prev_ids = []
        
        for track_id, track_history in self.tracks.items():
            if track_history and track_history[-1]['frame'] == frame_id - 1:
                prev_boxes.append(track_history[-1]['bbox'])
                prev_ids.append(track_id)
        
        if not prev_boxes:
            # 새로운 트랙 시작
            for det in detections:
                det['track_id'] = self.track_id_counter
                self.tracks[self.track_id_counter].append({
                    'frame': frame_id,
                    'bbox': det['bbox'],
                    'confidence': det['confidence']
                })
                self.track_id_counter += 1
            return
        
        # IoU 계산 및 매칭
        curr_boxes = [d['bbox'] for d in detections]
        iou_matrix = self._calculate_iou_matrix(prev_boxes, curr_boxes)
        
        # 헝가리안 알고리즘으로 최적 매칭 (간단한 greedy 매칭으로 대체 가능)
        matched_indices = self._greedy_matching(iou_matrix, self.match_thresh)
        
        for curr_idx, prev_idx in matched_indices:
            detections[curr_idx]['track_id'] = prev_ids[prev_idx]
            self.tracks[prev_ids[prev_idx]].append({
                'frame': frame_id,
                'bbox': detections[curr_idx]['bbox'],
                'confidence': detections[curr_idx]['confidence']
            })
    
    def _calculate_iou_matrix(self, boxes1, boxes2):
        """IoU 매트릭스 계산"""
        # 구현 생략 (supervision 라이브러리 활용 가능)
        pass
    
    def _greedy_matching(self, iou_matrix, threshold):
        """간단한 그리디 매칭"""
        # 구현 생략
        pass