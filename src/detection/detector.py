# src/detection/detector.py
from ultralytics import YOLO
import supervision as sv
import numpy as np

class FootballDetector:
    def __init__(self, model_path='runs/train/football_detector/weights/best.pt'):
        self.model = YOLO(model_path)
        self.field_detector = FieldDetector()
        
        # 클래스 매핑
        self.class_names = {
            0: 'player',
            1: 'referee', 
            2: 'ball',
            3: 'helmet'
        }
        
    def detect(self, frame):
        # YOLO 검출
        results = self.model(frame, conf=0.3, iou=0.5)[0]
        
        # Supervision 형식으로 변환
        detections = sv.Detections.from_ultralytics(results)
        
        # 필드 영역 필터링
        field_mask = self.field_detector.detect_field_region(frame)
        
        filtered_detections = []
        for i, (xyxy, confidence, class_id) in enumerate(
            zip(detections.xyxy, detections.confidence, detections.class_id)
        ):
            x1, y1, x2, y2 = xyxy
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 필드 내부 체크
            if field_mask[int(center_y), int(center_x)] > 0:
                filtered_detections.append({
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'confidence': confidence,
                    'class': self.class_names[class_id],
                    'class_id': class_id
                })
                
        return filtered_detections