# 🎬 AI-Powered Video Scene & Emotion Analysis System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0+-green.svg)](https://github.com/ultralytics/ultralytics)
[![DeepFace](https://img.shields.io/badge/DeepFace-0.0.79+-orange.svg)](https://github.com/serengil/deepface)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Intelligent Video Understanding Pipeline** combining scene segmentation, multimodal emotion analysis, and semantic retrieval for comprehensive video content analysis.

## 🌟 Executive Summary

This project delivers an **end-to-end interactive pipeline** for advanced video understanding, developed as a proof-of-concept system during a 2-month internship. The system combines **scene segmentation, multimodal emotion analysis, and intelligent retrieval** to transform raw video into structured, queryable knowledge.

---

## 🎯 Key Features

### **🎥 Video Processing Pipeline**
- **Intelligent Scene Segmentation**: Shot boundary detection using adaptive thresholding
- **Keyframe Extraction**: Representative frame selection per scene
- **Temporal Metadata**: Scene timing, duration, and transition analysis
- **Batch Processing**: Support for multiple video formats and resolutions

### **🧠 Multimodal Feature Extraction**
| Module | Technology | Capabilities |
|--------|------------|--------------|
| **Visual Analysis** | YOLOv8 | Object detection (80+ COCO classes) |
| **Facial Analysis** | DeepFace + MediaPipe | Face detection, emotion recognition (8 emotions) |
| **Scene Understanding** | BLIP-2 | Context-aware scene captioning |
| **Audio Analysis** | Custom ML Pipeline | Voice emotion detection (speech features) |
| **Character Tracking** | Heuristic Pipeline | Character presence & interaction analysis |

### **🔍 Intelligent Retrieval System**
- **Vector Embeddings**: Scene-level feature embeddings using Sentence Transformers
- **Semantic Search**: RAG-based natural language querying
- **Context-Aware Recommendations**: Similar scene discovery
- **Hybrid Search**: Combine semantic + metadata filtering

### **📊 Interactive Visualization**
- **Scene Cards**: Grid visualization with thumbnails and metadata
- **Analytics Dashboard**: Statistical summaries and trends
- **Timeline Explorer**: Temporal navigation through scenes
- **Export Capabilities**: JSON/CSV reports and visual summaries

---

## 🏗️ System Architecture

```mermaid
graph TB
    A[Input Video] --> B[Scene Segmentation]
    B --> C[Keyframe Extraction]
    
    subgraph "Multimodal Analysis Pipeline"
        C --> D[Visual Analysis]
        C --> E[Facial Analysis]
        C --> F[Audio Analysis]
        
        D --> G[Object Detection - YOLOv8]
        E --> H[Emotion Recognition - DeepFace]
        F --> I[Voice Emotion - Custom ML]
        
        G --> J[Scene Captioning - BLIP-2]
        H --> J
        I --> J
    end
    
    J --> K[Feature Embedding]
    K --> L[Vector Database - ChromaDB]
    
    subgraph "Query Interface"
        M[User Query] --> N[Semantic Search]
        N --> L
        L --> O[Retrieved Scenes]
        O --> P[Results Visualization]
    end
    
    P --> Q[Dashboard & Analytics]
