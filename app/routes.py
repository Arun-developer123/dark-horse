# app/routes.py
import os
import time
import threading
from flask import Blueprint, current_app, request, jsonify, render_template, url_for, send_from_directory, abort
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta

from . import db
from .models import Upload
from config import UPLOAD_DIR, MAX_FILE_SIZE, ALLOWED_EXT, ADMIN_TOKEN, MAX_FILE_AGE_DAYS, ML_MODEL_PATH

from .utils.image import extract_exif, pil_to_cv2, jpeg_quality_hint, save_upload_file, allowed_file_ext
from .detectors.heuristic import HeuristicDetector
from .detectors.ml_detector import MLDetector

from PIL import Image, ImageOps

bp = Blueprint("routes", __name__, template_folder="../templates")

# instantiate detectors (ONCE)
heuristic = HeuristicDetector()
ml_detector = MLDetector(model_path=ML_MODEL_PATH)  # will be None if no model

# ensemble weights
ALPHA = 0.6  # weight for heuristic
BETA = 0.4   # weight for ML (only used if ML returns prob)

# minimal index route (render your big template here)
@bp.route("/")
def index():
    admin_token = os.environ.get("ADMIN_TOKEN", "")
    # render template/index.html (paste your previous HTML)
    return render_template("index.html", mb=MAX_FILE_SIZE//(1024*1024), admin_token=admin_token, MAX_FILE_SIZE=MAX_FILE_SIZE)

@bp.route("/health")
def health():
    return jsonify({"status":"ok", "time": datetime.utcnow().isoformat()})

@bp.route("/uploads/<path:filename>")
def uploaded_file(filename):
    safe = secure_filename(filename)
    path = os.path.join(UPLOAD_DIR, safe)
    if not os.path.exists(path):
        abort(404)
    return send_from_directory(UPLOAD_DIR, safe, as_attachment=False)

@bp.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({'error':'no file provided'}), 400
    f = request.files["file"]
    filename_raw = secure_filename(f.filename or "")
    if not allowed_file_ext(filename_raw):
        return jsonify({'error':'file type not supported','allowed': list(ALLOWED_EXT)}), 400

    # size check
    f.seek(0, os.SEEK_END)
    size = f.tell()
    f.seek(0)
    if size > MAX_FILE_SIZE:
        return jsonify({'error':'file too large (> allowed bytes)'}), 400

    timestamp_suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    save_name = f"{os.path.splitext(filename_raw)[0]}_{timestamp_suffix}{os.path.splitext(filename_raw)[1]}" if filename_raw else f"upload_{timestamp_suffix}.jpg"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    try:
        f.save(save_path)
    except Exception as e:
        return jsonify({'error':'failed to save file','detail':str(e)}), 500

    # validate image
    try:
        pil = Image.open(save_path)
        pil = ImageOps.exif_transpose(pil)
    except Exception as e:
        try: os.remove(save_path)
        except: pass
        return jsonify({'error':'invalid image','detail':str(e)}), 400

    # EXIF
    exif = extract_exif(pil)
    exif_found = len(exif) > 0

    # prepare cv2
    cv2_img = pil_to_cv2(pil)
    if cv2_img is None or cv2_img.size == 0:
        return jsonify({'error':'failed to decode image'}), 400

    # run heuristic
    h = heuristic.analyze(pil_img=pil, cv2_img=cv2_img)
    h_score = float(h.get('score', 0.5))

    # run ML if available
    ml_prob = None
    if ml_detector.session:
        ml_prob = ml_detector.predict(pil)

    # ensemble
    if ml_prob is not None:
        final_raw = (ALPHA * h_score) + (BETA * ml_prob)
    else:
        final_raw = h_score

    final_score = float(round(final_raw * 100.0, 3))

    # reasons
    reasons = []
    if not exif_found:
        reasons.append("No EXIF metadata — suspicious.")
    else:
        reasons.append("EXIF present (may be forged).")
    reasons.append(f"Heuristic score: {h_score:.3f}")
    if ml_prob is not None:
        reasons.append(f"ML model prob(real): {ml_prob:.3f} (model loaded)")
    else:
        reasons.append("ML model: not loaded; using heuristics only.")

    # DB log
    try:
        row = Upload(filename=save_name, ip=request.remote_addr or 'unknown', realness_score=final_score, reasons="\n".join(reasons))
        db.session.add(row); db.session.commit()
    except Exception:
        db.session.rollback()

    # label
    if final_score >= 70:
        label = "Likely REAL"
    elif final_score >= 40:
        label = "Unsure / Possibly Real"
    else:
        label = "Likely AI / Synthetic"

    image_url = url_for('routes.uploaded_file', filename=save_name)

    out = {
        'realness_score': final_score,
        'label': label,
        'reasons': reasons,
        'image_url': image_url,
        'heuristic': h,
        'ml_prob': ml_prob,
        'exif_found': exif_found
    }

    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.accept_mimetypes.best == 'application/json'
    if is_ajax:
        return jsonify(out)
    return jsonify(out)