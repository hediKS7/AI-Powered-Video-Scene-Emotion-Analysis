import cv2
from collections import deque, Counter
import math
import os
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm
import librosa
import numpy as np

class ToneAnalyzer:
    def __init__(self, low_thresh=120, high_thresh=200):
        """
        Initialize tone analyzer with pitch thresholds.
        
        Parameters:
            low_thresh (float): Threshold below which pitch is considered "Low" (default: 120 Hz)
            high_thresh (float): Threshold above which pitch is considered "High" (default: 200 Hz)
        """
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
    
    def classify_tone(self, audio_path):
        """
        Classify speech tone (Low, Normal, High) based on mean fundamental frequency (F0).
        
        Parameters:
            audio_path (str): Path to the audio file (.wav, .mp3, etc.)
        
        Returns:
            str: One of "Low", "Normal", "High", or "No speech detected"
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)  # sr=None preserves original sample rate

            # Extract fundamental frequency (F0) using pyin
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y,
                fmin=librosa.note_to_hz('C2'),   # ~65.4 Hz
                fmax=librosa.note_to_hz('C7'),   # ~2093 Hz
                frame_length=2048,
                hop_length=512
            )

            # Keep only voiced (valid) F0 values
            f0 = f0[~np.isnan(f0)]  # Remove NaNs
            f0 = f0[f0 > 0]         # Keep only positive frequencies

            if len(f0) == 0:
                return "No speech detected"

            mean_pitch = np.mean(f0)

            if mean_pitch < self.low_thresh:
                return "Low"
            elif mean_pitch > self.high_thresh:
                return "High"
            else:
                return "Normal"
        except Exception as e:
            print(f"Error analyzing tone for {audio_path}: {str(e)}")
            return "Analysis error"
    
    def analyze_scenes_tone(self, scenes_df, audio_col='audio_path', progress=True):
        """
        Analyze tone for all scenes in dataframe.
        
        Parameters:
            scenes_df (pd.DataFrame): DataFrame containing scene information
            audio_col (str): Column name containing audio paths
            progress (bool): Whether to show progress bar
            
        Returns:
            pd.DataFrame: Modified dataframe with 'tone_level' column added
        """
        scenes_df = scenes_df.copy()
        if progress:
            tqdm.pandas()
            scenes_df['tone_level'] = scenes_df[audio_col].progress_apply(self.classify_tone)
        else:
            scenes_df['tone_level'] = scenes_df[audio_col].apply(self.classify_tone)
        return scenes_df