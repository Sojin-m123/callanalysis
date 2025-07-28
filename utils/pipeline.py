import os
import re
import requests
from pydub import AudioSegment
from faster_whisper import WhisperModel
import torch


class MalayalamTranscriptionPipeline:
    def __init__(self, model_size="small"):
        self.model_size = model_size
        self.colab_url = self.get_colab_ngrok_url()
        if self.colab_url:
            print(f"[INFO] Connected to Colab GPU Whisper server: {self.colab_url}")
        else:
            print("[ERROR] Colab GPU URL not found. Please check the Gist ID or network.")

    def get_colab_ngrok_url(self):
        gist_id = "dbff88f974aedb8f55b56869dca5c0dd"  # 🔁 Your Gist ID
        gist_url = f"https://api.github.com/gists/{gist_id}"
        try:
            res = requests.get(gist_url)
            if res.status_code == 200:
                files = res.json().get("files", {})
                url = files.get("ngrok_public_url.txt", {}).get("content", "").strip()
                return url
        except Exception as e:
            print(f"[ERROR] Failed to fetch Colab ngrok URL: {e}")
        return None

    def convert_to_wav(self, input_path):
        try:
            print(f"[INFO] Converting to WAV: {input_path}")
            audio = AudioSegment.from_file(input_path)
            duration_sec = len(audio) / 1000
            print(f"[INFO] Audio duration: {duration_sec:.2f} seconds")
            audio = audio.set_channels(1).set_frame_rate(16000)
            wav_path = os.path.splitext(input_path)[0] + '_converted.wav'
            audio.export(wav_path, format='wav')
            print(f"[INFO] WAV file created at: {wav_path}")
            return wav_path
        except Exception as e:
            print(f"[ERROR] Audio conversion failed: {e}")
            return input_path  # fallback

    def transcribe_audio(self, audio_path):
        if not self.colab_url:
            print("[ERROR] Colab URL not set. Skipping transcription.")
            return {
                "raw_transcription": "",
                "cleaned_transcription": "",
                "segments": [],
                "language": "unknown"
            }

        try:
            with open(audio_path, "rb") as f:
                files = {'audio': (os.path.basename(audio_path), f, 'audio/mpeg')}
                data = {'transcription_language': 'en', 'target_language': ''}
                res = requests.post(f"{self.colab_url}/transcribe", files=files, data=data)

            if res.status_code == 200:
                json_data = res.json()
                raw_text = json_data.get("raw_transcription", "")
                lang = json_data.get("language", "unknown")

                print(f"[INFO] Transcription success via Colab: {len(raw_text.split())} words")
                return {
                    "raw_transcription": raw_text,
                    "cleaned_transcription": self.clean_transcription(raw_text),
                    "segments": [],  # segment info not returned from Colab
                    "language": lang
                }
            else:
                print(f"[ERROR] Colab server error: {res.status_code} - {res.text}")
        except Exception as e:
            print(f"[ERROR] Failed to transcribe using Colab: {e}")

        return {
            "raw_transcription": "",
            "cleaned_transcription": "",
            "segments": [],
            "language": "unknown"
        }

    def clean_transcription(self, text):
        if not text or not text.strip():
            return ""
        filler_words = r'\b(uh+|um+|hmm+|mm-hmm+|haa+|aah+|ohh+)\b'
        cleaned = re.sub(filler_words, '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'([?!.,])\1+', r'\1', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = re.sub(r'\s+([?.!,])', r'\1', cleaned)
        return cleaned.strip()
