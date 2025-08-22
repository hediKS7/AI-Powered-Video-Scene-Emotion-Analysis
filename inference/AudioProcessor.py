from collections import deque, Counter
import math
import os
import pandas as pd
import datetime
import subprocess
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class AudioProcessor:
    def __init__(self, video_path, scenes_df):
        self.video_path = video_path
        self.scenes_df = scenes_df.copy()
        self._prepare_dataframe()
        
    def _prepare_dataframe(self):
        """Convert time columns to seconds and prepare dataframe"""
        self.scenes_df['start_sec'] = self.scenes_df['Start'].apply(self._time_to_seconds)
        self.scenes_df['end_sec'] = self.scenes_df['End'].apply(self._time_to_seconds)
        
    @staticmethod
    def _time_to_seconds(t):
        """Convert HH:MM:SS.sss to seconds"""
        try:
            dt = datetime.strptime(t, '%H:%M:%S.%f')
            delta = timedelta(
                hours=dt.hour, 
                minutes=dt.minute, 
                seconds=dt.second, 
                microseconds=dt.microsecond
            )
            return delta.total_seconds()
        except ValueError as e:
            logger.error(f"Time conversion error for {t}: {e}")
            return 0.0
    
    @staticmethod
    def _format_time(seconds):
        """Format seconds to hh:mm:ss.xxx for ffmpeg"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02}:{minutes:02}:{secs:06.3f}"
    
    def extract_audio_segments(self, output_dir="extracted_audio"):
        """Extract audio segments for each scene using ffmpeg"""
        os.makedirs(output_dir, exist_ok=True)
        audio_paths = []
        
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(f"FFmpeg not available: {e}")
            return self.scenes_df
        
        for idx, row in tqdm(self.scenes_df.iterrows(), total=len(self.scenes_df), desc="Extracting audio"):
            start = row['start_sec']
            end = row['end_sec']
            duration = end - start
            
            if duration <= 0:
                logger.warning(f"Invalid duration for scene {idx}: {duration}")
                audio_paths.append("invalid_duration")
                continue
                
            audio_output = os.path.join(output_dir, f"scene_{idx:04d}.wav")
            
            cmd = [
                "ffmpeg",
                "-ss", self._format_time(start),
                "-t", f"{duration:.3f}",
                "-i", self.video_path,
                "-ac", "1",  # Convert to mono
                "-ar", "16000",  # Resample to 16kHz
                "-q:a", "0",
                "-map", "a",
                "-y",
                audio_output
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                if os.path.exists(audio_output) and os.path.getsize(audio_output) > 0:
                    audio_paths.append(audio_output)
                else:
                    logger.error(f"Empty audio file created for scene {idx}")
                    audio_paths.append("error")
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg failed for scene {idx}: {e.stderr.decode()}")
                audio_paths.append("error")
                
        self.scenes_df['audio_path'] = audio_paths
        return self.scenes_df