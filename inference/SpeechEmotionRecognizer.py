import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification


import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechEmotionRecognizer:
    def __init__(self):
        """Initialize with Hugging Face pipeline"""
        try:
            self.pipe = pipeline(
                "audio-classification",
                model="superb/wav2vec2-base-superb-er",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info("Loaded speech emotion recognition model")
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            self.pipe = None

    def analyze_emotions(self, scenes_df):
        """Analyze emotions for all audio segments"""
        if self.pipe is None:
            scenes_df['voice_emotion'] = ["emotion analysis unavailable"] * len(scenes_df)
            return scenes_df

        emotions = []
        for idx, row in tqdm(scenes_df.iterrows(), total=len(scenes_df), desc="Analyzing emotions"):
            audio_path = row.get('audio_path', '')
            
            if not audio_path or audio_path in ["error", "invalid_duration"] or not os.path.exists(audio_path):
                emotions.append("no audio")
                continue
                
            try:
                result = self.pipe(audio_path)
                emotions.append(result[0]['label'])
            except Exception as e:
                logger.error(f"Emotion analysis failed for scene {idx}: {e}")
                emotions.append("analysis error")

        scenes_df['voice_emotion'] = emotions
        return scenes_df