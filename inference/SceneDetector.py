import os
from pathlib import Path
import pandas as pd
from scenedetect import VideoManager, SceneManager, ContentDetector
from scenedetect.scene_manager import save_images

class SceneDetector:
    def __init__(self, threshold=28, output_dir='scenes'):
        self.threshold = threshold
        self.output_dir = output_dir
        
    def segment_video(self, video_path):
        os.makedirs(self.output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        video_manager = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))

        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        
        scene_list = scene_manager.get_scene_list()
        print(f"📍 {len(scene_list)} scenes detected.\n")

        save_images(scene_list, video_manager, num_images=1, output_dir=self.output_dir)

        data = []
        for i, (start, end) in enumerate(scene_list):
            filename = f"{base_name}-Scene-{i+1:03d}-01.jpg"
            data.append({
                'Scene': i + 1,
                'Start': start.get_timecode(),
                'End': end.get_timecode(),
                'filename': filename,
                'path': os.path.join(self.output_dir, filename)
            })

        return scene_list, pd.DataFrame(data)

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