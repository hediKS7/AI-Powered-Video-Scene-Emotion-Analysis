import os
from ultralytics import YOLO
import pandas as pd

class ObjectDetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)
        self.model_names = self.model.names

    def detect_objects(self, img_path):
        results = self.model(img_path, save=False)
        detected = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                detected.append(self.model_names[cls_id])
        return detected

    def analyze_scenes(self, scenes_df):
        scenes_df = scenes_df.copy()
        scenes_df['objects'] = scenes_df['path'].apply(self.detect_objects)
        return scenes_df