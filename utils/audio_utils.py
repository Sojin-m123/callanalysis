# audio_utils.py
import os
from datetime import datetime
import tempfile
from pydub import AudioSegment

def convert_to_whisper_format(input_path):
    supported_formats = ['.mp3', '.wav', '.aac', '.m4a', '.flac', '.wma']
    file_ext = os.path.splitext(input_path)[1].lower()
    if file_ext == '.wav':
        return input_path

    if file_ext not in supported_formats:
        raise ValueError(f"Unsupported audio format: {file_ext}")

    temp_dir = os.path.join(tempfile.gettempdir(), "whisper_temp")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    wav_path = os.path.join(temp_dir, f"temp_{timestamp}.wav")

    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(wav_path, format="wav")
        if not os.path.exists(wav_path):
            raise RuntimeError("Failed to create WAV file during conversion.")
        return wav_path
    except Exception as e:
        # Fallback: FFmpeg command-line
        import subprocess
        cmd = ['ffmpeg', '-i', input_path, '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', wav_path]
        subprocess.run(cmd, check=True)
        if not os.path.exists(wav_path):
            raise RuntimeError("FFmpeg conversion failed to produce WAV.")
        return wav_path
