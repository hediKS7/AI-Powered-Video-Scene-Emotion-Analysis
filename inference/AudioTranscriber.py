import os
import whisper
from tqdm import tqdm

class AudioTranscriber:
    def __init__(self):
        """Initialize with Hugging Face pipeline for better compatibility"""
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-small",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info("Loaded Whisper model for transcription")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.pipe = None

    def transcribe_audio_segments(self, scenes_df):
        """Transcribe audio segments using Whisper"""
        if self.pipe is None:
            scenes_df['transcript'] = ["transcription unavailable"] * len(scenes_df)
            return scenes_df

        transcripts = []
        for idx, row in tqdm(scenes_df.iterrows(), total=len(scenes_df), desc="Transcribing"):
            audio_path = row.get('audio_path', '')
            
            if not audio_path or audio_path in ["error", "invalid_duration"] or not os.path.exists(audio_path):
                transcripts.append("no audio")
                continue
                
            try:
                result = self.pipe(audio_path)
                text = result.get("text", "").strip()
                transcripts.append(text if text else "no speech detected")
            except Exception as e:
                logger.error(f"Transcription failed for scene {idx}: {e}")
                transcripts.append("transcription error")

        scenes_df['transcript'] = transcripts
        return scenes_df