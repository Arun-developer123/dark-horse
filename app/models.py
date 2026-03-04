# app/models.py
from . import db
from datetime import datetime

class Upload(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(512), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    ip = db.Column(db.String(100))
    realness_score = db.Column(db.Float)
    reasons = db.Column(db.Text)