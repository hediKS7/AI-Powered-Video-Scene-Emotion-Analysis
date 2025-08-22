import cv2
import mediapipe as mp
from deepface import DeepFace
from collections import deque, Counter
import math
import os
import pandas as pd
from scenedetect import VideoManager, SceneManager, ContentDetector
from scenedetect.scene_manager import save_images
import numpy as np
from tqdm import tqdm
from PIL import Image
from tqdm import tqdm


import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class FaceEmotionPredictor:
    """
    Robust face emotion predictor combining DeepFace + heuristic landmarks.
    Replace the previous class implementation with this one.
    Public methods:
      - predict_emotions_on_image(img) -> (emotion_label: str, annotated_img: np.ndarray)
      - analyze_scenes_emotions(scenes_df) -> scenes_df with 'emotion' column
    """

    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def __init__(self, window_size=5, deep_weight=0.65, heuristic_weight=0.35,
                 min_detection_confidence=0.5, min_face_crop_size=50):
        self.window_size = window_size
        self.deep_weight = deep_weight
        self.heuristic_weight = heuristic_weight
        self.history = deque(maxlen=window_size)
        self.min_face_crop_size = min_face_crop_size  # px - skip tiny faces for DeepFace

        # Initialize MediaPipe models once and reuse them to avoid repeated overhead/errors
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands

        # persistent instances
        self.face_detector = self.mp_face_detection.FaceDetection(model_selection=0,
                                                                  min_detection_confidence=min_detection_confidence)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True,
                                                    max_num_faces=5,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=min_detection_confidence)
        self.hands = self.mp_hands.Hands(static_image_mode=True,
                                         max_num_hands=2,
                                         min_detection_confidence=min_detection_confidence)

        logger.info("FaceEmotionPredictor initialized (MediaPipe ready).")

    # ---- helper utils ----
    @staticmethod
    def _euclidean_dist(p1, p2):
        try:
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
        except Exception:
            return 0.0

    @staticmethod
    def _safe_get_point(landmarks, idx, w, h):
        """
        Return (x,y) pixel tuple or None if index not present.
        landmarks: mp.landmark list or object that has .landmark
        """
        if landmarks is None:
            return None
        pts = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
        if idx < 0 or idx >= len(pts):
            return None
        lm = pts[idx]
        # some landmarks might have values outside 0..1; cap them
        return (min(max(lm.x * w, 0.0), w - 1.0), min(max(lm.y * h, 0.0), h - 1.0))

    def _bbox_from_detection(self, detection, w, h, pad_ratio=0.15):
        """Return (x, y, bw, bh) in pixel coords (with padding)"""
        try:
            bbox = detection.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            # padding
            pad_w = int(bw * pad_ratio)
            pad_h = int(bh * pad_ratio)
            x0 = max(0, x - pad_w)
            y0 = max(0, y - pad_h)
            x1 = min(w, x + bw + pad_w)
            y1 = min(h, y + bh + pad_h)
            return x0, y0, x1 - x0, y1 - y0
        except Exception:
            return 0, 0, w, h

    def _normalize_deepface_output(self, deep_res):
        """
        DeepFace.analyze may return dict or list depending on version.
        We must produce a dict mapping lowercase emotion -> score for EMOTIONS.
        """
        try:
            if isinstance(deep_res, list):
                deep_res = deep_res[0] if deep_res else {}
            # Expect deep_res['emotion'] to be dict of str->value
            em = deep_res.get('emotion') if isinstance(deep_res, dict) else None
            if not em:
                return {e: 1.0 / len(self.EMOTIONS) for e in self.EMOTIONS}
            # normalize keys to lowercase and only keep EMOTIONS
            total = sum(em.values()) or 1.0
            normalized = {}
            for e in self.EMOTIONS:
                # DeepFace uses names like 'angry','happy' etc — use .get with fallback 0
                normalized[e] = float(em.get(e, em.get(e.capitalize(), 0))) / total
            # if all zeros fallback to uniform
            if sum(normalized.values()) == 0:
                return {e: 1.0 / len(self.EMOTIONS) for e in self.EMOTIONS}
            return normalized
        except Exception:
            return {e: 1.0 / len(self.EMOTIONS) for e in self.EMOTIONS}

    def _heuristic_scores_from_landmarks(self, face_landmarks, hand_landmarks, w, h):
        """
        Compute a lightweight heuristic score dict from facial landmarks and hands.
        Uses distances for eyes, mouth aspect ratio and simple hand positions.
        """
        scores = {e: 0.0 for e in self.EMOTIONS}
        # key indices (MediaPipe face mesh indices)
        # eyes: top and bottom approximate pairs; lips indices for mouth
        try:
            le_top = self._safe_get_point(face_landmarks, 159, w, h)
            le_bot = self._safe_get_point(face_landmarks, 145, w, h)
            re_top = self._safe_get_point(face_landmarks, 386, w, h)
            re_bot = self._safe_get_point(face_landmarks, 374, w, h)
            mouth_top = self._safe_get_point(face_landmarks, 13, w, h)
            mouth_bot = self._safe_get_point(face_landmarks, 14, w, h)
            mouth_left = self._safe_get_point(face_landmarks, 61, w, h)
            mouth_right = self._safe_get_point(face_landmarks, 291, w, h)
        except Exception:
            le_top = le_bot = re_top = re_bot = mouth_top = mouth_bot = mouth_left = mouth_right = None

        # fallback safe values
        def safe_dist(a, b):
            if a is None or b is None:
                return 0.0
            return self._euclidean_dist(a, b)

        eye_open = (safe_dist(le_top, le_bot) + safe_dist(re_top, re_bot)) / 2.0
        mouth_open = safe_dist(mouth_top, mouth_bot)
        mouth_width = safe_dist(mouth_left, mouth_right) or 1.0
        mouth_aspect = mouth_open / mouth_width
        eye_ratio = eye_open / mouth_width

        # Hands features (coarse)
        hand_states = {'over_head': False, 'near_face': False, 'on_body': False, 'under_head': False}
        if hand_landmarks:
            for hand in hand_landmarks:
                # compute average point for hand to approximate relation to face
                coords = [(lm.x * w, lm.y * h) for lm in hand.landmark]
                cx = sum([c[0] for c in coords]) / len(coords)
                cy = sum([c[1] for c in coords]) / len(coords)
                # near_face: within 1.5x mouth width of face center? (approx)
                # We don't have exact face center; use mouth center if available else skip
                if mouth_top and mouth_bot:
                    face_cx = (mouth_left[0] + mouth_right[0]) / 2 if mouth_left and mouth_right else cx
                    face_cy = (mouth_top[1] + mouth_bot[1]) / 2
                    dist = math.hypot(cx - face_cx, cy - face_cy)
                    if dist < max(40, mouth_width * 1.5):
                        hand_states['near_face'] = True
                    if cy < face_cy - (mouth_width * 0.6):
                        hand_states['over_head'] = True
                    if cy > face_cy + (mouth_width * 2.0):
                        hand_states['on_body'] = True

        # heuristic scoring rules (tuned simple thresholds)
        # neutral baseline
        scores['neutral'] += 0.5
        # happy: moderate mouth aspect
        if 0.18 < mouth_aspect < 0.6:
            scores['happy'] += 1.0
        # sad: low mouth and low eyes
        if mouth_aspect < 0.22 and eye_ratio < 0.12:
            scores['sad'] += 1.5
        # surprise: large mouth opening
        if mouth_aspect > 0.45:
            scores['surprise'] += 1.8
            if eye_ratio > 0.25:
                scores['surprise'] += 0.6
        # fear: mouth open + small eyes
        if mouth_aspect > 0.4 and eye_ratio < 0.15:
            scores['fear'] += 1.2
        # angry: raised brow (approx by difference in certain landmarks) or fist
        # simple proxy: if eye opening is small but brow gap larger - we skip complexity; use mouth width ratio
        if eye_ratio > 0.0 and (eye_ratio < 0.12 and mouth_aspect < 0.35):
            scores['angry'] += 1.2
        if hand_states.get('near_face'):
            scores['surprise'] += 0.6

        # normalize heuristic to probabilities
        total = sum(scores.values()) or 1.0
        heuristic_probs = {e: (scores[e] / total) for e in self.EMOTIONS}
        return heuristic_probs

    def predict_emotions_on_image(self, img):
        """
        img: BGR image (numpy array)
        returns: (dominant_emotion_str, annotated_img)
        """
        annotated = img.copy()
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            img_rgb = img

        h, w = img.shape[:2]

        # run detectors (reusing persistent mediapipe instances)
        detection_result = self.face_detector.process(img_rgb)
        mesh_result = self.face_mesh.process(img_rgb)
        hands_result = self.hands.process(img_rgb)

        detections = getattr(detection_result, "detections", None)
        if not detections:
            return "no face", annotated

        face_meshes = mesh_result.multi_face_landmarks or []
        hand_landmarks = hands_result.multi_hand_landmarks or []

        emotions_all_faces = []

        # For each face detection, find a corresponding face_mesh (closest centroid) if possible
        for det_idx, detection in enumerate(detections):
            # bounding box
            x, y, bw, bh = self._bbox_from_detection(detection, w, h, pad_ratio=0.18)
            # crop
            x1, y1 = int(x), int(y)
            x2, y2 = int(min(w, x + bw)), int(min(h, y + bh))
            if x2 <= x1 or y2 <= y1:
                # invalid box
                continue

            face_crop = img[y1:y2, x1:x2]
            # size guard
            if face_crop.shape[0] < self.min_face_crop_size or face_crop.shape[1] < self.min_face_crop_size:
                # too small - skip DeepFace (but try heuristics if we can)
                deep_probs = {e: 1.0 / len(self.EMOTIONS) for e in self.EMOTIONS}
            else:
                # run DeepFace safely
                try:
                    # DeepFace may be noisy on BGR; use cvt to RGB
                    df_res = DeepFace.analyze(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB),
                                              actions=['emotion'],
                                              enforce_detection=False,
                                              prog_bar=False,
                                              detector_backend='opencv')
                    deep_probs = self._normalize_deepface_output(df_res)
                except Exception as e:
                    logger.debug(f"DeepFace analyze failed (falling back to uniform): {e}")
                    deep_probs = {e: 1.0 / len(self.EMOTIONS) for e in self.EMOTIONS}

            # choose matching face_mesh: find mesh with centroid closest to detection center
            face_mesh_for_det = None
            if face_meshes:
                # compute detection center in pixel coords
                det_box = detection.location_data.relative_bounding_box
                det_cx = (det_box.xmin + det_box.width / 2.0) * w
                det_cy = (det_box.ymin + det_box.height / 2.0) * h
                best_i = None
                best_dist = float("inf")
                for i, mesh in enumerate(face_meshes):
                    # compute mesh centroid quickly
                    xs = [lm.x for lm in mesh.landmark]
                    ys = [lm.y for lm in mesh.landmark]
                    mesh_cx = (sum(xs) / len(xs)) * w
                    mesh_cy = (sum(ys) / len(ys)) * h
                    d = math.hypot(det_cx - mesh_cx, det_cy - mesh_cy)
                    if d < best_dist:
                        best_dist = d
                        best_i = i
                if best_i is not None:
                    face_mesh_for_det = face_meshes[best_i]

            # heuristic probabilities from landmarks (face_mesh_for_det may be None)
            heuristic_probs = self._heuristic_scores_from_landmarks(face_mesh_for_det, hand_landmarks, w, h)

            # combine probabilities
            combined = {}
            for e in self.EMOTIONS:
                combined[e] = self.deep_weight * deep_probs.get(e, 0.0) + self.heuristic_weight * heuristic_probs.get(e, 0.0)

            # if combined sums to zero (safe-guard), fallback
            s = sum(combined.values()) or 1.0
            for k in combined:
                combined[k] = combined[k] / s

            dominant = max(combined, key=combined.get)
            emotions_all_faces.append(dominant)

            # annotate image
            try:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{dominant}"
                cv2.putText(annotated, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            except Exception:
                pass

        # choose most common across faces (or neutral fallback)
        if emotions_all_faces:
            most_common = Counter(emotions_all_faces).most_common(1)[0][0]
        else:
            most_common = "no face"

        # smoothing via history if desired
        try:
            self.history.append(most_common)
            # majority in window
            most_common = Counter(self.history).most_common(1)[0][0]
        except Exception:
            pass

        return most_common, annotated

    def analyze_scenes_emotions(self, scenes_df):
        """
        Iterate scenes_df['path'] images and detect face emotion per scene.
        Adds/updates 'emotion' column.
        """
        emotion_labels = []
        for path in scenes_df['path']:
            try:
                img = cv2.imread(path)
                if img is None:
                    emotion_labels.append("no face")
                    continue
                emotion, _ = self.predict_emotions_on_image(img)
                emotion_labels.append(emotion)
            except Exception as e:
                logger.warning(f"Error analyzing scene {path}: {e}")
                emotion_labels.append("analysis error")
        scenes_df['emotion'] = emotion_labels
        return scenes_df

    def close(self):
        """Release mediapipe resources if you want to explicitly free them."""
        try:
            if hasattr(self, "face_detector") and self.face_detector:
                self.face_detector.close()
            if hasattr(self, "face_mesh") and self.face_mesh:
                self.face_mesh.close()
            if hasattr(self, "hands") and self.hands:
                self.hands.close()
        except Exception:
            pass