# app/utils/cleanup.py
import threading, time, os
from datetime import datetime, timedelta
from config import UPLOAD_DIR, MAX_FILE_AGE_DAYS
from ..models import Upload
from .. import db

def cleanup_old_files_loop():
    while True:
        try:
            now = datetime.utcnow()
            cutoff = now - timedelta(days=MAX_FILE_AGE_DAYS)
            for fname in os.listdir(UPLOAD_DIR):
                path = os.path.join(UPLOAD_DIR, fname)
                try:
                    mtime = datetime.utcfromtimestamp(os.path.getmtime(path))
                    if mtime < cutoff:
                        os.remove(path)
                        try:
                            Upload.query.filter_by(filename=fname).delete()
                            db.session.commit()
                        except Exception:
                            db.session.rollback()
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(3600)

def start_cleanup_thread():
    t = threading.Thread(target=cleanup_old_files_loop, daemon=True)
    t.start()