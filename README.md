# 🎬 AI-Powered Video Scene & Emotion Analysis

This project delivers an **end-to-end interactive pipeline** for video understanding, combining **scene segmentation, multimodal emotion analysis, and intelligent retrieval**.  
It was developed during a 2-month internship as a proof-of-concept system .

---

## 🚀 Features

- **Video Upload & Processing**
  - Shot boundary detection and scene segmentation
  - Keyframe extraction and scene-level metadata generation  

- **Multimodal Feature Extraction**
  - Object detection (YOLOv8)  
  - Face & character presence analysis (DeepFace, heuristics approach)  
  - Voice emotion detection 
  - Automatic scene captioning (BLIP/BLIP-2)  

- **Results Visualization**
  - Interactive **scene cards**   
  - Summary tables and descriptive reports  

- **Intelligent Retrieval**
  - **RAG-based semantic search engine** powered by embeddings  
  - Natural language queries on video scenes  
  - Context-aware scene recommendations  

---

## 🚀 Objectives  
- Segment videos into **meaningful scenes** using shot boundary detection.  
- Extract **keyframes** that best represent each scene.  
- Detect **objects, faces, and gestures** using state-of-the-art models.  
- Perform **emotion recognition** via facial expressions and heuristic cues.  
- Generate **descriptive captions** for each scene using multimodal models.  
- Store embeddings for **semantic search & retrieval** (RAG-based).  
- Provide an intuitive **dashboard/UI** for visualization and interaction.  

---

## 🛠️ Tech Stack  
- **Scene Captioning:** BLIP / BLIP-2  
- **Object Detection:** YOLOv8
- **Face Detection & Emotion:** DeepFace + MediaPipe heuristics
- **Voice Emotion Dectecion:** ML Classifier Based on Feature Extraction
- **Scene-Level Labeling:** Gemini LLMs (RAG pipeline)  
- **Vector Search:** ChromaDB   
