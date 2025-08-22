import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import re
import json
import tempfile
import logging
import pandas as pd
# ---------- Gemini API ----------
API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDCdXM9KLj4SgalJU3_vRYRAZEJNwhPfWY")
if not API_KEY:
    logger.warning("GEMINI_API_KEY is not set. The Q&A tab will error until you set it.")
genai.configure(api_key=API_KEY)


def get_llm():
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Export it before using Ask the Video.")
    return GoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)


ALIASES = {
    "scene": "scene_id",
    "sceneid": "scene_id",
    "scene id": "scene_id",

    "start": "start_time",
    "start time": "start_time",

    "duration": "duration",
    "durations": "duration",
    "duration(s)": "duration",
    "duration (s)": "duration",

    "faceemot": "face_emotion",
    "face emotion": "face_emotion",
    "face emot": "face_emotion",

    "transcript": "transcript",

    "voiceemc": "voice_emotion",
    "voiceemot": "voice_emotion",
    "voice emotion": "voice_emotion",

    "objects": "objects",

    "imagecap": "image_caption",
    "image captioning": "image_caption",
    "image caption": "image_caption",
    "image cap": "image_caption",

    "speechtone": "speech_tone",
    "speech tone": "speech_tone",
    "speech to": "speech_tone",

    "scene type": "scene_type",
    "scenetype": "scene_type",

    "scene description": "scene_description",
    "scenedescription": "scene_description",
}

def _norm_key(k: str) -> str:
    k = (k or "").strip().lower()
    k = re.sub(r"[\s\-_]+", " ", k)
    k = k.replace(":", "").replace(",", "")
    return k

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    mapped = {}
    for c in df.columns:
        key = _norm_key(c)
        key_no_space = key.replace(" ", "")
        norm = (
            ALIASES.get(key) or
            ALIASES.get(key_no_space) or
            key.replace(" ", "_")
        )
        mapped[c] = norm
    df = df.rename(columns=mapped)
    return df


def _as_list_or_text(x):
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            try:
                val = json.loads(s)
                if isinstance(val, list):
                    return ", ".join(map(str, val))
                if isinstance(val, dict):
                    return json.dumps(val, ensure_ascii=False)
            except Exception:
                pass
    return str(x)

def row_to_block(row: pd.Series) -> str:
    sid = row.get("scene_id", row.name)
    start = row.get("start_time", "")
    duration = row.get("duration", "")
    transcript = row.get("transcript", "")
    face_emotion = row.get("face_emotion", "")
    voice_emotion = row.get("voice_emotion", "")
    objects = row.get("objects", "")
    image_caption = row.get("image_caption", "")
    speech_tone = row.get("speech_tone", "")
    scene_type = row.get("scene_type", "")
    scene_description = row.get("scene_description", "")

    objects = _as_list_or_text(objects)

    return (
        f"Scene: {sid}\n"
        f"Start: {start} | Duration(s): {duration}\n"
        f"Transcript: {transcript}\n"
        f"Face Emotion: {face_emotion}\n"
        f"Voice Emotion: {voice_emotion}\n"
        f"Objects: {objects}\n"
        f"Image Caption: {image_caption}\n"
        f"Speech Tone: {speech_tone}\n"
        f"Scene Type: {scene_type}\n"
        f"Scene Description: {scene_description}"
    ).strip()


def build_retriever_from_csv(csv_path: str):
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError("CSV path not found.")
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    if df.empty:
        raise ValueError("CSV is empty.")

    df = normalize_headers(df)

    docs = []
    for _, row in df.iterrows():
        content = row_to_block(row)
        # keep raw metadata for display
        meta = {k: (None if (v is None or str(v) == "nan") else v) for k, v in row.to_dict().items()}
        docs.append(Document(page_content=content, metadata=meta))

    tmpdir = tempfile.mkdtemp(prefix="video_rag_idx_")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=tmpdir)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    return retriever, len(docs), tmpdir


QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You answer questions strictly using the provided video scene metadata.\n"
        "Be concise (1–3 sentences). If unknown, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    ),
)

def answer_question(question: str, retriever):
    if not question or not question.strip():
        return "Please enter a question about the video."
    llm = get_llm()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
    resp = qa.invoke({"query": question})
    answer = (resp.get("result") or "").strip()
    sources = resp.get("source_documents") or []

    # Compact source hints
    hints = []
    for d in sources[:5]:
        m = d.metadata or {}
        sid = m.get("scene_id") or m.get("scene") or "N/A"
        st = m.get("start_time") or m.get("start") or ""
        hints.append(f"Scene {sid}" + (f" [{st}]" if st else ""))
    if hints:
        answer += "\n\nSources: " + "; ".join(hints)
    return answer