# src/detection/field_detector.py
import cv2
import numpy as np

class FieldDetector:
    def __init__(self):
        self.field_mask = None
        
    def detect_field_region(self, frame):
        """잔디 색상 기반 필드 영역 검출"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 잔디 색상 범위 (조정 필요)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # 모폴로지 연산으로 노이즈 제거
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 가장 큰 연결 영역 찾기
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [largest], 255)
            
        self.field_mask = mask
        return mask
    
    def filter_detections(self, detections, margin=50):
        """필드 내부 객체만 필터링"""
        if self.field_mask is None:
            return detections
            
        filtered = []
        for det in detections:
            x, y, w, h = det['bbox']
            center_x = x + w/2
            center_y = y + h/2
            
            # 마진을 고려한 필드 내부 체크
            if self.field_mask[int(center_y), int(center_x)] > 0:
                filtered.append(det)
                
        return filtered