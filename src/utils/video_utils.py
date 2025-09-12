# src/utils/video_utils.py
import cv2
import numpy as np

class VideoProcessor:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def get_frame(self, frame_num=None):
        if frame_num is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        return ret, frame
    
    def process_video(self, process_func, output_path=None):
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, 
                                 (self.width, self.height))
        
        results = []
        frame_num = 0
        
        while True:
            ret, frame = self.get_frame()
            if not ret:
                break
                
            processed_frame, frame_results = process_func(frame, frame_num)
            results.append(frame_results)
            
            if output_path:
                out.write(processed_frame)
            
            frame_num += 1
            
        if output_path:
            out.release()
        self.cap.release()
        
        return results