"""
Bring Voices Back to Life – Demo
================================
A minimal voice-cloning web app built with:
  • Coqui-AI TTS (XTTS v2 model – multilingual, zero-shot voice cloning)
  • Gradio for the UI

Run locally
-----------
1. Create/activate a fresh virtual environment (recommended).
2. Install dependencies (first run will also download ~1.3 GB model weights):

   pip install TTS==0.22.0 gradio==4.22.0 soundfile==0.12.1

3. Start the app:

   python demo_voice_clone.py

4. Open the displayed local URL in your browser and test.

Run in Google Colab
-------------------
1. Upload this script to Colab or copy its contents into a cell.
2. Execute the cell – it installs deps and launches a public Gradio link automatically.

Optional: Deploy on Hugging Face Spaces
---------------------------------------
• Create a new Space (Gradio template).
• Add this file and a `requirements.txt` containing the three packages above.
• Push – Spaces will build and host the app.

Notes & Limitations
-------------------
• XTTS v2 supports ~15-30 s reference audio (wav/mp3). Shorter clips may work but quality varies.
• First inference is slower due to model download + load (~1-2 min on CPU).
• For best results, run on GPU – set env var  `export DEVICE=cuda` before launch or rely on auto-detect.
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple

import gradio as gr
from TTS.api import TTS

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Select device automatically (GPU if available).
DEVICE = os.getenv("DEVICE", "cuda" if TTS.is_cuda_available() else "cpu")

# Allowed input audio extensions
ALLOWED_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Load the multi-lingual XTTS v2 model once at startup.
# (Weights are downloaded to ~/.local/share/tts/ or HF cache on first use.)
print("Loading TTS model … this may take a while the first time ⌛")
TTS_MODEL = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(DEVICE)
print("Model loaded ✅")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def validate_inputs(voice_file: gr.File, text: str) -> None:
    """Raise gr.Error with a friendly message if inputs are missing/invalid."""
    if voice_file is None:
        raise gr.Error("Please upload a short voice sample (WAV/MP3).")

    if not text or text.strip() == "":
        raise gr.Error("Please type some text to generate.")

    ext = Path(voice_file.name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        raise gr.Error(f"Unsupported audio type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTS))}")


def clone_voice(voice_file: gr.File, text: str) -> Tuple[str, str]:
    """Generate speech in the style of *voice_file* saying *text*.

    Returns: (path_to_audio, path_to_audio) so it can be used for both
    the audio player and a download link.
    """
    # Validate before heavy processing
    validate_inputs(voice_file, text)

    # Use a temp directory for intermediate & output files
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "input_audio"
        output_path = Path(tmpdir) / "output.wav"

        # Save uploaded file locally for TTS
        # gradio gives us a file-like object; write its content to disk.
        with open(input_path, "wb") as f_out:
            f_out.write(voice_file.read())

        # Generate speech (blocking call)
        try:
            TTS_MODEL.tts_to_file(
                text=text,
                speaker_wav=str(input_path),
                language="en",  # change/parameterize as needed
                file_path=str(output_path),
            )
        except Exception as exc:
            raise gr.Error(f"TTS generation failed: {exc}")

        # Move result to a persistent temp file because the context manager will delete tmpdir
        final_fd, final_path = tempfile.mkstemp(suffix=".wav")
        os.close(final_fd)
        os.replace(output_path, final_path)

    return final_path, final_path

# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------

def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Bring Voices Back to Life – Demo") as demo:
        gr.Markdown("# Bring Voices Back to Life – Demo\nClone anyone's voice in seconds – purely with free, open-source AI.")

        with gr.Row():
            with gr.Column():
                voice_input = gr.File(label="Upload Voice Sample", file_count="single", type="binary")
                text_input = gr.Textbox(label="Type your text here", placeholder="e.g. Hello, this is my cloned voice!")
                generate_btn = gr.Button("Generate Voice", variant="primary")
            with gr.Column():
                audio_output = gr.Audio(label="Generated Voice", type="filepath", autoplay=False)
                download_link = gr.File(label="Download »", visible=False)

        def _post_process(result_path: str):
            # If generation succeeds, show download link
            download_link.visible = True
            return result_path, result_path

        generate_btn.click(
            fn=clone_voice,
            inputs=[voice_input, text_input],
            outputs=[audio_output, download_link],
            preprocess=True,
            postprocess=False,  # We'll handle visibility manually
        ).then(_post_process, inputs=download_link, outputs=[audio_output, download_link])

        return demo


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # In Colab, the first installed gradio version >=4 automatically launches a public link.
    iface = build_interface()
    iface.launch()