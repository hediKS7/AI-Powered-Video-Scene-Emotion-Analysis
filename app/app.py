import os
import logging
import gradio as gr
from analyzer import VideoAnalyzer
from chatbot import build_retriever_from_csv, answer_question

logger = logging.getLogger("video_analyzer_app")
logging.basicConfig(level=logging.INFO)

analyzer = VideoAnalyzer()

# Keep retriever reference after analysis
retriever_state = {"retriever": None, "csv_path": None, "doc_count": 0}


def run_analysis(video_path):
    html, gallery, cards, csv_file, json_file = analyzer.analyze_video(video_path)

    if csv_file and os.path.exists(csv_file):
        try:
            retriever, doc_count, tmpdir = build_retriever_from_csv(csv_file)
            retriever_state["retriever"] = retriever
            retriever_state["csv_path"] = csv_file
            retriever_state["doc_count"] = doc_count
            logger.info(f"Retriever built with {doc_count} docs from {csv_file}")
        except Exception as e:
            logger.error(f"Failed to build retriever: {e}")
            retriever_state["retriever"] = None
    else:
        retriever_state["retriever"] = None

    return html, gallery, cards, csv_file, json_file


def ask_video_question(question):
    retriever = retriever_state.get("retriever")
    if not retriever:
        return "❌ No retriever available. Please analyze a video first."
    try:
        return answer_question(question, retriever)
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return f"Error: {e}"


# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gradio-container {max-width: 100% !important; padding: 0; margin: 0;}
    body, .gradio-container {font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;}
    .main-app {min-height: 100vh; padding: 18px; box-sizing: border-box;}
""") as demo:

    gr.Markdown("# 🎥 AI-Powered Video Scene & Emotion Analysis")
    gr.Markdown(
        "Upload a video; it detects scenes, extracts audio, runs object & emotion analysis, "
        "Now you can also **ask questions** about the video content "
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video", sources=["upload"])
            analyze_btn = gr.Button("Analyze", variant="primary")
            gr.Markdown("**Note:** Processing may take a few minutes without GPU")
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("📊 Summary"):
                    html_output = gr.HTML(label="Analysis Summary")
                with gr.Tab("🖼️ Scenes"):
                    gallery_output = gr.Gallery(label="Detected Scenes", columns=[3], height="auto")
                    scene_cards_html = gr.HTML(label="Scene Cards")
                with gr.Tab("📝 Data"):
                    gr.Markdown("### Export results")
                    csv_file = gr.File(label="Download CSV", interactive=False)
                    json_file = gr.File(label="Download JSON", interactive=False)
                with gr.Tab("💬 Ask the Video"):
                    gr.Markdown("### Ask any question about the video")
                    question_box = gr.Textbox(label="Your Question", placeholder="e.g. Which scene had sad voice emotion?")
                    answer_box = gr.Textbox(label="Answer", interactive=False)
                    ask_btn = gr.Button("Ask")

    # Wire buttons
    analyze_btn.click(
        fn=run_analysis,
        inputs=video_input,
        outputs=[html_output, gallery_output, scene_cards_html, csv_file, json_file]
    )

    ask_btn.click(
        fn=ask_video_question,
        inputs=question_box,
        outputs=answer_box
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)