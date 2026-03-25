from pydub import AudioSegment
from pydub.utils import which
import os

# ✅ Manually set ffmpeg path (even if it's in PATH)
AudioSegment.converter = which("ffmpeg")

def predict_audio(audio_path):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")
    
    sound = AudioSegment.from_mp3(audio_path)
    # Example return until full logic is added
    return "Dummy transcription", "Safe"
