from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

db = SQLAlchemy()

class CallAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(150), unique=True)
    agent = db.Column(db.String(50))
    phone = db.Column(db.String(20))
    transcript_en = db.Column(db.Text)
    transcript_ml = db.Column(db.Text)
    sentiment = db.Column(db.String(20))
    intent = db.Column(db.String(50))
    lead_score = db.Column(db.Integer)
    intent_score = db.Column(db.Integer)
    trigger_words = db.Column(db.Text)
    audio_path = db.Column(db.String(255))
    report_path = db.Column(db.String(255))  # ✅ PDF report path
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    @property
    def audio_filename(self):
        return os.path.basename(self.audio_path) if self.audio_path else ""

    @property
    def timestamp(self):
        return self.created_at
