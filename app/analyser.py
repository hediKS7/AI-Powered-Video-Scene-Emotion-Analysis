import os
import sys
import tempfile
import traceback
import logging
from datetime import timedelta   
import datetime as _datetime_module 
import pandas as pd
import numpy as np
import cv2
import gradio as gr
import json
logger = logging.getLogger("video_analyzer_app")
logging.basicConfig(level=logging.INFO)



# --- Patch module-level datetime.strptime before importing inference.py ---
try:
    if not hasattr(_datetime_module, "strptime"):
        _datetime_module.strptime = _datetime_module.datetime.strptime
        logger.info("Patched module-level datetime.strptime -> datetime.datetime.strptime")
except Exception as e:
    logger.warning(f"Could not patch datetime.strptime: {e}")

# --- Import user's inference pipeline ---
try:
    import inference as inference_module
except Exception as e:
    logger.error(f"Failed to import inference.py: {e}\n{traceback.format_exc()}")
    raise

# --- Inject `timedelta` into inference module globals ---
try:
    if not hasattr(inference_module, "timedelta"):
        inference_module.timedelta = _datetime_module.timedelta
        logger.info("Injected 'timedelta' into inference module globals")
except Exception as e:
    logger.warning(f"Could not inject timedelta into inference module: {e}")



from inference import (
    AudioProcessor,
    SceneDetector,
    ImageCaptioner,
    ObjectDetector,
    AudioTranscriber,
    FaceEmotionPredictor,
    SpeechEmotionRecognizer,
    ToneAnalyzer,
    SceneAnalyzer,
)

# Example: create instances of each class
audio_processor = AudioProcessor()
scene_detector = SceneDetector()
image_captioner = ImageCaptioner()
object_detector = ObjectDetector()
audio_transcriber = AudioTranscriber()
face_emotion_predictor = FaceEmotionPredictor()
speech_emotion_recognizer = SpeechEmotionRecognizer()
tone_analyzer = ToneAnalyzer()
scene_analyzer = SceneAnalyzer()




class VideoAnalyzer:
    def __init__(self):
        try:
            logger.info("Initializing pipeline components (this may load models)...")
            self.scene_detector = SceneDetector()
            self.obj_detector = ObjectDetector()
            self.face_emotion_predictor = FaceEmotionPredictor()
            self.image_captioner = ImageCaptioner()
            self.audio_transcriber = self._initialize_transcriber()
            self.speech_emotion_recognizer = self._initialize_emotion_recognizer()
            self.tone_analyzer = ToneAnalyzer()
            self.scene_analyzer = SceneAnalyzer()
            logger.info("Pipeline initialized.")
        except Exception as e:
            logger.error(f"Initialization failed: {e}\n{traceback.format_exc()}")
            raise

    def _initialize_transcriber(self):
        try:
            t = AudioTranscriber()
            logger.info("AudioTranscriber initialized")
            return t
        except Exception as e:
            logger.error(f"AudioTranscriber init error: {e}")
            return None

    def _initialize_emotion_recognizer(self):
        try:
            r = SpeechEmotionRecognizer()
            logger.info("SpeechEmotionRecognizer initialized")
            return r
        except Exception as e:
            logger.error(f"SpeechEmotionRecognizer init error: {e}")
            return None

    def analyze_video(self, video_file, progress=gr.Progress()):
        """
        Main pipeline runner. Returns: (html_summary, gallery_images_list, scene_cards_html, csv_path, json_path)
        Note: the detailed DataFrame is still created internally and saved to CSV/JSON, but not displayed in the UI.
        """
        try:
            progress(0, desc="Starting analysis...")
            with tempfile.TemporaryDirectory() as tmpdir:
                video_path = os.path.join(tmpdir, "input_video.mp4")

                # Handle input file
                if isinstance(video_file, str):
                    try:
                        with open(video_file, "rb") as src, open(video_path, "wb") as dst:
                            dst.write(src.read())
                        logger.info("Copied provided video path into temp dir")
                    except Exception:
                        video_path = video_file
                        logger.info("Using provided video path directly")
                else:
                    with open(video_path, "wb") as f:
                        f.write(video_file.read())
                    logger.info("Saved uploaded video to temp dir")

                progress(0.05, desc="Detecting scenes...")
                scene_list, scenes_df = self.scene_detector.segment_video(video_path)
                logger.info(f"Detected {len(scenes_df)} scenes")

                if len(scenes_df) == 0:
                    raise RuntimeError("No scenes detected in the video.")

                progress(0.25, desc="Preparing scene timings...")
                if "Start" in scenes_df.columns and "End" in scenes_df.columns and "duration" not in scenes_df.columns:
                    def _parse_timecode(t):
                        try:
                            parts = str(t).split(":")
                            if len(parts) == 3:
                                h = int(parts[0]); m = int(parts[1]); s = float(parts[2])
                                return h * 3600 + m * 60 + s
                        except Exception:
                            pass
                        return 0.0
                    scenes_df["start_seconds_preview"] = scenes_df["Start"].apply(_parse_timecode)
                    scenes_df["end_seconds_preview"] = scenes_df["End"].apply(_parse_timecode)
                    scenes_df["duration_preview"] = scenes_df["end_seconds_preview"] - scenes_df["start_seconds_preview"]

                progress(0.35, desc="Scene understanding...")
                try:
                    scenes_df = self.obj_detector.analyze_scenes(scenes_df)
                except Exception as e:
                    logger.error(f"Object detection error: {e}\n{traceback.format_exc()}")
                    scenes_df["objects"] = [["detection error"]] * len(scenes_df)

                progress(0.45, desc="Scene understanding...")
                try:
                    scenes_df = self.image_captioner.caption_scenes(scenes_df)
                except Exception as e:
                    logger.error(f"Image captioning error: {e}\n{traceback.format_exc()}")
                    scenes_df["caption"] = ["caption error"] * len(scenes_df)

                progress(0.55, desc="Analysing...")
                try:
                    scenes_df = self.face_emotion_predictor.analyze_scenes_emotions(scenes_df)
                except Exception as e:
                    logger.error(f"Face emotion analysis error: {e}\n{traceback.format_exc()}")
                    scenes_df["emotion"] = ["analysis error"] * len(scenes_df)

                progress(0.70, desc="Analysing...")
                audio_processor = AudioProcessor(video_path, scenes_df)
                scenes_df = audio_processor.extract_audio_segments(output_dir=os.path.join(tmpdir, "audio"))

                progress(0.82, desc="Analysing...")
                if self.audio_transcriber:
                    try:
                        scenes_df = self.audio_transcriber.transcribe_audio_segments(scenes_df)
                    except Exception as e:
                        logger.error(f"Transcription error: {e}\n{traceback.format_exc()}")
                        scenes_df["transcript"] = ["transcription error"] * len(scenes_df)
                else:
                    scenes_df["transcript"] = ["transcription unavailable"] * len(scenes_df)

                progress(0.87, desc="Analyzing speech tone...")
                try:
                    scenes_df = self.tone_analyzer.analyze_scenes_tone(scenes_df)
                except Exception as e:
                    logger.error(f"Tone analysis error: {e}\n{traceback.format_exc()}")
                    scenes_df["tone_level"] = ["analysis error"] * len(scenes_df)

                progress(0.92, desc="Analyzing speech emotions...")
                if self.speech_emotion_recognizer:
                    try:
                        scenes_df = self.speech_emotion_recognizer.analyze_emotions(scenes_df)
                    except Exception as e:
                        logger.error(f"Speech emotion analysis error: {e}\n{traceback.format_exc()}")
                        scenes_df["voice_emotion"] = ["analysis error"] * len(scenes_df)
                else:
                    scenes_df["voice_emotion"] = ["emotion analysis unavailable"] * len(scenes_df)

                progress(0.95, desc="Generating rich scene descriptions...")
                try:
                    scenes_df = self.scene_analyzer.analyze_scenes(scenes_df, show_images=False)
                except Exception as e:
                    logger.error(f"Scene analysis error: {e}\n{traceback.format_exc()}")
                    scenes_df["scene_category"] = ["analysis error"] * len(scenes_df)
                    scenes_df["scene_description"] = ["analysis error"] * len(scenes_df)

                progress(0.98, desc="Finalizing results...")
                html, gallery, scene_cards_html, tmp_csv_path, tmp_json_path = self._format_results(scenes_df)
                progress(1.0, desc="Done")
                logger.info("Pipeline finished successfully")
                return html, gallery, scene_cards_html, tmp_csv_path, tmp_json_path

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Analysis failed: {e}\n{tb}")
            return self._create_error_output(f"{str(e)}\n{tb}")

    def _make_scene_cards_html(self, scenes_df: pd.DataFrame):
        """Create an HTML grid of flip-cards that show Scene Category on front and Description on back.
        Includes client-side search & category filter (JS) so no extra Gradio callbacks are needed.
        """
        # Normalize columns
        df = scenes_df.copy()
        if 'scene_category' not in df.columns:
            df['scene_category'] = df.get('scene_category', '').astype(str)
        if 'scene_description' not in df.columns:
            df['scene_description'] = df.get('scene_description', '').astype(str)
        if 'Scene' not in df.columns:
            df['Scene'] = [str(i+1) for i in range(len(df))]

        # Build cards
        cards_html_list = []
        categories = set()
        for _, r in df.iterrows():
            cat = str(r.get('scene_category', '') or '')
            desc = str(r.get('scene_description', '') or '')
            scene_id = str(r.get('Scene', ''))
            categories.add(cat)
            safe_cat = cat.replace('"', '&quot;')
            safe_desc = desc.replace('"', '&quot;').replace('\n', '<br/>')

            card = (
                '<div class="flip-card" data-cat="' + safe_cat + '">' 
                '<div class="flip-card-inner">'
                '<div class="flip-card-front">'
                '<div class="front-content">'
                '<div class="front-cat">' + (safe_cat or 'Unknown') + '</div>'
                '<div class="front-id">Scene #' + scene_id + '</div>'
                '</div>'
                '</div>'
                '<div class="flip-card-back">'
                '<div class="back-desc">' + (safe_desc or '<em>No description</em>') + '</div>'
                '</div>'
                '</div>'
                '</div>'
            )
            cards_html_list.append(card)

        categories_list = sorted([c for c in categories if c.strip() != ''])
        categories_options_html = ''.join(['<option value="' + c + '">' + c + '</option>' for c in categories_list])
        cards_html = ''.join(cards_html_list)

        # Build responsive, full-screen friendly CSS + JS using a single template string
        html_template = """
        <style>
        :root{ --bg:#071029; --card:#07142a; --glass: rgba(255,255,255,0.03); --muted: rgba(255,255,255,0.75); --accent:#4f46e5; }
        body, .gradio-container{background:linear-gradient(180deg,#021026 0%, #071029 100%); color: #e6eef8;}
        .gradio-container{max-width:100% !important; padding:18px 20px; box-sizing:border-box;}
        .app-shell{display:flex; flex-direction:column; gap:18px; min-height:calc(100vh - 36px);} 
        .top-summary{background:transparent; padding:10px 12px; border-radius:10px;}
        .scene-area{display:flex; flex-direction:column; gap:12px;}
        .scene-controls{display:flex; gap:10px; align-items:center; flex-wrap:wrap;}
        .search-input{flex:1; min-width:160px; padding:10px 14px; border-radius:999px; border:none; background:var(--glass); color:inherit; box-shadow:0 2px 12px rgba(2,6,23,0.6);}
        .select-cat{padding:10px 14px; border-radius:999px; border:none; background:var(--glass); color:inherit;}
        .scene-grid{display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap:14px; align-items:start;}
        /* flip card */
        .flip-card{perspective:1000px;}
        .flip-card-inner{position:relative; width:100%; height:180px; transition:transform 0.6s; transform-style:preserve-3d; border-radius:12px; box-shadow:0 8px 26px rgba(2,6,23,0.6);}
        .flip-card:hover .flip-card-inner{transform:rotateY(180deg) translateY(-6px);}
        .flip-card-front, .flip-card-back{position:absolute; width:100%; height:100%; -webkit-backface-visibility:hidden; backface-visibility:hidden; border-radius:12px; display:flex; align-items:center; justify-content:center; padding:12px; box-sizing:border-box;}
        .flip-card-front{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));}
        .flip-card-back{background:linear-gradient(180deg, rgba(79,70,229,0.14), rgba(79,70,229,0.06)); transform:rotateY(180deg); color:#fff;}
        .front-cat{font-weight:700; font-size:1rem; margin-bottom:6px; text-align:center;}
        .front-id{font-size:0.85rem; color:var(--muted); text-align:center;}
        .back-desc{font-size:0.95rem; color:var(--muted); overflow:auto; max-height:calc(100% - 10px);}
        @media (max-width:600px){ .flip-card-inner{height:150px;} .back-desc{font-size:0.9rem;} }
        </style>
        <div class="scene-cards-wrap app-shell">
            <div class="scene-controls">
                <input class="search-input" id="scene-search" placeholder="Search descriptions or categories..." />
                <select class="select-cat" id="scene-filter">
                    <option value="">All categories</option>
                    {categories_options}
                </select>
                <button class="select-cat" id="scene-reset">Reset</button>
            </div>
            <div class="scene-grid" id="scene-grid">
                {cards}
            </div>
        </div>
        <script>
        (function(){
            const search = document.getElementById('scene-search');
            const filter = document.getElementById('scene-filter');
            const reset = document.getElementById('scene-reset');
            const grid = document.getElementById('scene-grid');
            function applyFilter(){
                const q = (search.value||'').toLowerCase();
                const c = (filter.value||'').toLowerCase();
                const cards = grid.querySelectorAll('.flip-card');
                cards.forEach(card=>{
                    const text = (card.querySelector('.back-desc').innerText||'').toLowerCase();
                    const cat = (card.getAttribute('data-cat')||'').toLowerCase();
                    const matchesQ = q.length===0 || text.includes(q) || cat.includes(q);
                    const matchesC = c.length===0 || cat===c;
                    card.style.display = (matchesQ && matchesC) ? "block" : "none";
                });
            }
            search.addEventListener('input', applyFilter);
            filter.addEventListener('change', applyFilter);
            reset.addEventListener('click', ()=>{ search.value=''; filter.value=''; applyFilter(); });
        })();
        </script>
        """

        html = html_template.replace('{categories_options}', categories_options_html).replace('{cards}', cards_html)
        return html

    def _format_results(self, scenes_df: pd.DataFrame):
        scenes_df = scenes_df.copy()

        # Ensure duration column exists
        if "duration" not in scenes_df.columns:
            if "duration_preview" in scenes_df.columns:
                scenes_df["duration"] = scenes_df["duration_preview"]
            elif "start_seconds" in scenes_df.columns and "end_seconds" in scenes_df.columns:
                scenes_df["duration"] = scenes_df["end_seconds"] - scenes_df["start_seconds"]
            else:
                scenes_df["duration"] = 0.0

        total_duration = scenes_df["duration"].sum() if len(scenes_df) > 0 else 0.0
        num_scenes = len(scenes_df)

        html_output = f"""
        <div class='top-summary' style='font-family: Inter, Arial, sans-serif;'>
            <h2 style='color: var(--accent, #4f46e5); margin:6px 0;'>Video Analysis Summary</h2>
            <div style='display:flex; gap:12px; flex-wrap:wrap;'>
                <div><strong>Total Duration:</strong> {timedelta(seconds=int(total_duration))}</div>
                <div><strong>Scenes:</strong> {num_scenes}</div>
                <div><strong>Avg Scene Duration:</strong> {scenes_df['duration'].mean():.1f}s</div>
            </div>
        </div>
        """

        # Build gallery (kept as-is)
        gallery_images = []
        if "path" in scenes_df.columns:
            for p in scenes_df["path"].tolist():
                gallery_images.append(p if isinstance(p, str) and os.path.exists(p) else None)
        elif "filename" in scenes_df.columns:
            for f in scenes_df["filename"].tolist():
                gallery_images.append(f if isinstance(f, str) and os.path.exists(f) else None)
        else:
            gallery_images = [None] * len(scenes_df)

        # Prepare display DataFrame but do not expose it in UI (kept for exports)
        display_cols = []
        if "Scene" in scenes_df.columns:
            display_cols.append("Scene")
        if "Start" in scenes_df.columns:
            display_cols.append("Start")
        display_cols.append("duration")
        if "emotion" in scenes_df.columns:
            scenes_df = scenes_df.rename(columns={"emotion": "Face Emotion"})
            display_cols.append("Face Emotion")
        if "transcript" in scenes_df.columns:
            display_cols.append("transcript")
        if "voice_emotion" in scenes_df.columns:
            display_cols.append("voice_emotion")
        if "objects" in scenes_df.columns:
            display_cols.append("objects")
        if "caption" in scenes_df.columns:
            display_cols.append("caption")
        if "tone_level" in scenes_df.columns:
            display_cols.append("tone_level")
        if "scene_category" in scenes_df.columns:
            display_cols.append("scene_category")
        if "scene_description" in scenes_df.columns:
            display_cols.append("scene_description")

        for c in display_cols:
            if c not in scenes_df.columns:
                scenes_df[c] = ""

        df_display = scenes_df[display_cols].copy()
        rename_map = {
            "duration": "Duration (s)",
            "transcript": "Transcript",
            "voice_emotion": "Voice Emotion",
            "objects": "Objects",
            "caption": "Image Caption",
            "tone_level": "Speech Tone",
            "scene_category": "Scene Type",
            "scene_description": "Scene Description"
        }
        df_display = df_display.rename(columns=rename_map)
        if "Duration (s)" in df_display.columns:
            try:
                df_display["Duration (s)"] = df_display["Duration (s)"].astype(float).round(2)
            except Exception:
                pass

        # Create scene cards HTML (flip cards: front = category, back = description)
        scene_cards_html = self._make_scene_cards_html(scenes_df)

        # Save CSV and JSON to temp files so user can download them after analysis
        try:
            tmp_csv = tempfile.NamedTemporaryFile(delete=False, suffix='.csv', prefix='video_analysis_')
            df_display.to_csv(tmp_csv.name, index=False)
            tmp_csv_path = tmp_csv.name
            tmp_csv.close()
        except Exception as e:
            logger.warning(f"Could not write CSV: {e}")
            tmp_csv_path = ""

        try:
            tmp_json = tempfile.NamedTemporaryFile(delete=False, suffix='.json', prefix='video_analysis_')
            # write as records for easy consumption
            with open(tmp_json.name, 'w', encoding='utf-8') as jf:
                json.dump(df_display.to_dict(orient='records'), jf, ensure_ascii=False, indent=2)
            tmp_json_path = tmp_json.name
            tmp_json.close()
        except Exception as e:
            logger.warning(f"Could not write JSON: {e}")
            tmp_json_path = ""

        return html_output, gallery_images, scene_cards_html, tmp_csv_path, tmp_json_path

    def _create_error_output(self, error_msg: str):
        error_html = f"""
        <div style='color: white; background-color: #ef4444; padding: 12px; border-radius: 6px;'>
            <h3>Error Processing Video</h3>
            <p>{str(error_msg).splitlines()[0]}</p>
            <details><summary>Debug Details</summary><pre>{error_msg}</pre></details>
        </div>
        """
        # Keep return shape compatible with success path but empty others
        return error_html, [], "", "", ""