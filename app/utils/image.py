# app/utils/image.py
import os
from PIL import Image, ExifTags, ImageOps
import piexif
import numpy as np

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def allowed_file_ext(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXT

def save_upload_file(storage_file, upload_dir, filename):
    path = os.path.join(upload_dir, filename)
    storage_file.save(path)
    return path

def extract_exif(pil_img):
    try:
        raw_exif = pil_img.info.get('exif', b'')
        if raw_exif:
            ex = piexif.load(raw_exif)
            out = {}
            for ifd in ex:
                for k,v in ex[ifd].items():
                    try:
                        if isinstance(v, bytes):
                            v = v.decode(errors='ignore')
                    except Exception:
                        pass
                    out[f"{ifd}:{k}"] = v
            return out
    except Exception:
        pass
    # fallback
    try:
        raw = getattr(pil_img, "_getexif", lambda: {})() or {}
        readable = {}
        for k,v in raw.items():
            name = ExifTags.TAGS.get(k, k)
            readable[name] = v
        return readable
    except Exception:
        return {}

def pil_to_cv2(pil_img):
    arr = np.array(pil_img.convert('RGB'))
    return arr[:, :, ::-1].copy()  # RGB->BGR

def jpeg_quality_hint(pil_img):
    try:
        qtables = getattr(pil_img, 'quantization', None)
        if not qtables:
            return None, None
        vals = []
        for k,v in qtables.items():
            vals.extend(v)
        import numpy as _np
        mean_q = float(_np.mean(vals)) if vals else None
        if mean_q is None:
            return None, None
        score = 1.0 - (mean_q / 200.0)
        return max(0.0, min(1.0, score)), mean_q
    except Exception:
        return None, None