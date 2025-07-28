# reset_db.py
from app import db, CallAnalysis, app
import os

DB_PATH = 'call_analysis.db'

with app.app_context():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        print("🗑️ Deleted old call_analysis.db")
    else:
        print("ℹ️ No old DB found. Creating fresh one.")

    db.create_all()
    print("✅ New DB created with columns:")
    for column in CallAnalysis.__table__.columns:
        print(f"  - {column.name} ({column.type})")
