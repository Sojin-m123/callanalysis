from faster_whisper import WhisperModel
from pydub import AudioSegment
import torch
import os
import re
import time

class MalayalamTranscriptionPipeline:
    def __init__(self, model_size="small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Loading Faster-Whisper '{model_size}' on {self.device}...")

        if self.device == "cpu":
            num_threads = os.cpu_count() or 1
            torch.set_num_threads(num_threads)
            compute_type = "int8"
            cpu_threads = num_threads
        else:
            compute_type = "float16"
            cpu_threads = 0

        self.model = WhisperModel(
            model_size,
            device=self.device,
            compute_type=compute_type,
            cpu_threads=cpu_threads
        )
        print("[INFO] Whisper model loaded.")

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
        try:
            print(f"[INFO] Starting transcription for: {audio_path}")
            start = time.time()
            wav_path = self.convert_to_wav(audio_path)

            segments, info = self.model.transcribe(
                wav_path,
                beam_size=5,         # ✅ SPEED boost
                language="en"        # ✅ Fixed language (Malayalam)
            )

            raw_text = " ".join([seg.text.strip() for seg in segments if seg.text.strip()])
            language = info.language if hasattr(info, 'language') else "unknown"

            print(f"[INFO] Transcription done in {time.time() - start:.2f} sec")
            print(f"[INFO] Words: {len(raw_text.split())}, Detected language: {language}")

            return {
                "raw_transcription": raw_text,
                "cleaned_transcription": self.clean_transcription(raw_text),
                "segments": segments,
                "language": language
            }

        except Exception as e:
            print(f"[ERROR] Transcription failed: {e}")
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
