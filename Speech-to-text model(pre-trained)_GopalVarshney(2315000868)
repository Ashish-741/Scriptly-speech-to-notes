# --- Run this cell ONCE per session ---

import os
import subprocess
import torch
import language_tool_python
from faster_whisper import WhisperModel

# Load Whisper model (only once, reuse later)
device = "cpu"   # or "cuda" if GPU available
compute_type = "int8" if device == "cpu" else "float16"

model = WhisperModel("small", device=device, compute_type=compute_type)

# Punctuation model (optional, if you have it)
from deepmultilingualpunctuation import PunctuationModel
punct_model = PunctuationModel()

def restore_punctuation(text):
    return punct_model.restore_punctuation(text)

# Grammar correction
tool = language_tool_python.LanguageTool('en-US')

def correct_grammar(text):
    return tool.correct(text)

# Simple text cleanup
import re
def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9.,?!'\s]", "", text).strip()



# --- Run this cell EVERY TIME with a new input audio ---

# 1. Input file path (update this every time)
input_file = "../AudioFiles/Beach6.mp3"

# 2. Fixed processed file name (always overwrite)
processed_file = "../AudioFiles/processed.wav"

# 3. Preprocess with FFmpeg
try:
    subprocess.run([
        "ffmpeg",
        "-i", input_file,
        "-ac", "1",       # mono
        "-ar", "16000",   # 16kHz
        "-af", "loudnorm,silenceremove=start_periods=1:start_silence=0.5:start_threshold=-30dB:stop_periods=-1:stop_silence=0.5:stop_threshold=-30dB",
        "-y",
        processed_file
    ], check=True, capture_output=True, text=True)
    print("✅ Processed:", processed_file)
except subprocess.CalledProcessError as e:
    print("❌ FFmpeg failed:", e.stderr)

# 4. Transcribe with VAD
segments, info = model.transcribe(
    processed_file,
    language="en",
    beam_size=5,
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500)  # cut at ~0.5s pauses
)

# 5. Post-processing
for segment in segments:
    raw_text = segment.text
    cleaned_text = clean_text(raw_text)
    punctuated_text = restore_punctuation(cleaned_text)
    corrected_text = correct_grammar(punctuated_text)
    print(corrected_text)

# 6. (Optional) Delete processed file to keep things clean
try:
    os.remove(processed_file)
    print("🗑 Temp file deleted:", processed_file)
except Exception as e:
    print("⚠ Could not delete temp file:", e)
