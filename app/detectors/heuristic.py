# app/detectors/heuristic.py
import math
import numpy as np
import cv2
from scipy import fftpack
from scipy.stats import kurtosis

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def sensor_noise_score(cv2_gray):
    img = cv2_gray.astype(np.float32) / 255.0
    blurred = cv2.GaussianBlur(img, (5,5), 0)
    residual = img - blurred
    std = float(np.std(residual))
    score = (std - 0.002) / (0.04 - 0.002)
    return clamp01(score), std

def smoothness_score(cv2_gray):
    img = cv2_gray.astype(np.float32) / 255.0
    mean = cv2.blur(img, (3,3))
    mean_sq = cv2.blur(img*img, (3,3))
    var = mean_sq - mean*mean
    mean_var = float(np.mean(var))
    score = (mean_var - 1e-6) / (0.005 - 1e-6)
    return clamp01(score), mean_var

def frequency_kurtosis_score(cv2_gray):
    img = cv2_gray.astype(np.float32)
    h, w = img.shape
    target = 512
    scale = max(1, int(max(h,w)/target))
    if scale > 1:
        img_small = cv2.resize(img, (w//scale, h//scale), interpolation=cv2.INTER_AREA)
    else:
        img_small = img
    f = fftpack.fft2(img_small)
    fshift = fftpack.fftshift(f)
    mag = np.abs(fshift).flatten()
    log_mag = np.log1p(mag)
    k = float(kurtosis(log_mag, fisher=False, nan_policy='omit') or 0.0)
    if math.isnan(k):
        k = 0.0
    if k <= 2:
        score = 0.25
    elif k <= 10:
        score = 0.95
    elif k <= 14:
        score = 0.7
    else:
        score = 0.25
    return clamp01(score), k

def artifact_penalty(cv2_bgr):
    img = cv2_bgr.astype(np.float32) / 255.0
    gray = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)/255.0
    h, w = gray.shape
    sims = []
    num_checks = 24
    for _ in range(num_checks):
        y = np.random.randint(0, max(1, h-32))
        x = np.random.randint(0, max(1, w-32))
        y2 = np.random.randint(0, max(1, h-32))
        x2 = np.random.randint(0, max(1, w-32))
        p1 = gray[y:y+32, x:x+32]
        p2 = gray[y2:y2+32, x2:x2+32]
        num = np.sum((p1-p1.mean())*(p2-p2.mean()))
        den = np.sqrt(np.sum((p1-p1.mean())**2)*np.sum((p2-p2.mean())**2)+1e-9)
        sims.append(num/(den+1e-9))
    sims = np.array(sims)
    high_sim_frac = float(np.mean(sims > 0.82))
    flipped = np.fliplr(gray)
    symmetry = float(np.mean(np.abs(gray - flipped)))
    penalty = 0.0
    if high_sim_frac > 0.2:
        penalty += min(1.0, high_sim_frac * 2.2)
    if symmetry < 0.02:
        penalty += 0.5
    penalty = float(np.clip(penalty, 0.0, 1.0))
    return penalty, high_sim_frac, symmetry

def edge_density_score(cv2_gray):
    g = cv2_gray.astype(np.uint8)
    v = np.median(g)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(g, lower, upper)
    density = float(np.sum(edges>0) / (edges.size + 1e-9))
    score = (density - 0.001) / (0.04 - 0.001)
    return clamp01(score), density

def color_hist_kurtosis_score(cv2_bgr):
    chans = cv2.split(cv2_bgr)
    ks = []
    for c in chans:
        hist = cv2.calcHist([c], [0], None, [256], [0,256]).flatten()
        hist = hist / (hist.sum()+1e-9)
        log_hist = np.log1p(hist)
        try:
            k = float(kurtosis(log_hist, fisher=False, nan_policy='omit'))
        except Exception:
            k = 0.0
        ks.append(k if not math.isnan(k) else 0.0)
    mean_k = float(np.mean(ks))
    if mean_k <= 2:
        score = 0.3
    elif mean_k <= 10:
        score = 0.9
    elif mean_k <= 14:
        score = 0.6
    else:
        score = 0.25
    return clamp01(score), mean_k

def entropy_score(cv2_gray):
    hist = cv2.calcHist([cv2_gray.astype(np.uint8)], [0], None, [256], [0,256]).flatten()
    p = hist / (hist.sum()+1e-9)
    p = p[p>0]
    ent = float(-np.sum(p * np.log2(p)))
    score = (ent - 3.0) / (7.0 - 3.0)
    return clamp01(score), ent

def compute_final_realness(signals):
    weights = {
        'exif': 0.12,
        'sensor': 0.25,
        'smooth': 0.18,
        'freq': 0.15,
        'edge': 0.10,
        'color': 0.08,
        'entropy': 0.07,
    }
    total_w = 0.0
    acc = 0.0
    for k,w in weights.items():
        if k in signals and signals[k] is not None:
            acc += signals[k] * w
            total_w += w
    base = acc / total_w if total_w>0 else 0.5
    penalty = signals.get('artifact_penalty', 0.0)
    raw = base * (1.0 - 0.6 * penalty)
    return float(np.clip(raw, 0.0, 1.0))

class HeuristicDetector:
    def analyze(self, pil_img, cv2_img=None):
        """
        Input: PIL image. Optionally supply cv2_bgr array.
        Returns: dict with signals + final score (0..1)
        """
        if cv2_img is None:
            cv2_img = pil_to_cv2(pil_img) if 'pil_to_cv2' in globals() else None
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        s_sensor, sensor_std = sensor_noise_score(gray)
        s_smooth, mean_var = smoothness_score(gray)
        s_freq, kurt = frequency_kurtosis_score(gray)
        penalty, high_sim_frac, symmetry = artifact_penalty(cv2_img)
        s_edge, edge_density = edge_density_score(gray)
        s_color, color_k = color_hist_kurtosis_score(cv2_img)
        s_entropy, ent = entropy_score(gray)
        signals = {
            'sensor': s_sensor,
            'smooth': s_smooth,
            'freq': s_freq,
            'artifact_penalty': penalty,
            'edge': s_edge,
            'color': s_color,
            'entropy': s_entropy
        }
        score = compute_final_realness(signals)
        return {
            'signals': signals,
            'score': score,
            'sensor_std': sensor_std,
            'smoothness_meanvar': mean_var,
            'frequency_kurtosis': kurt,
            'artifact_penalty': penalty,
            'high_sim_fraction': high_sim_frac,
            'symmetry': symmetry,
            'edge_density': edge_density,
            'color_hist_kurtosis': color_k,
            'entropy': ent
        }