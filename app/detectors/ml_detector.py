# app/detectors/ml_detector.py
import os
import numpy as np

class MLDetector:
    """
    Optional ONNX model wrapper.
    The ONNX model should accept input shape (1,3,H,W) float32 and output either:
      - a single sigmoid/logit scalar (probability 'realness'), or
      - a vector where first element = prob(real)
    You must supply ML_MODEL_PATH in config (or env).
    """
    def __init__(self, model_path=None):
        self.model_path = model_path or os.environ.get("ML_MODEL_PATH", "")
        self.session = None
        if self.model_path and os.path.exists(self.model_path):
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                # get input name/shape
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
            except Exception as e:
                self.session = None
                print("MLDetector: failed to load model:", e)

    def preprocess(self, pil_img, size=224):
        # resize, convert to RGB, normalize (ImageNet-like), return NCHW float32
        from PIL import Image
        img = pil_img.convert('RGB').resize((size,size), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        # normalize mean/std (imagenet)
        mean = np.array([0.485,0.456,0.406], dtype=np.float32)
        std  = np.array([0.229,0.224,0.225], dtype=np.float32)
        arr = (arr - mean) / std
        # HWC -> CHW
        arr = np.transpose(arr, (2,0,1))
        arr = np.expand_dims(arr, 0).astype(np.float32)
        return arr

    def predict(self, pil_img):
        if not self.session:
            return None
        try:
            x = self.preprocess(pil_img, size=224)
            inp = {self.input_name: x}
            out = self.session.run([self.output_name], inp)
            pred = out[0]
            # support outputs like (1,)->prob or (1,2)->logits
            if getattr(pred, "shape", None) is not None:
                pred = np.asarray(pred).reshape(-1)
                # heuristic: if len==1 => probability
                if pred.size == 1:
                    prob = float(pred[0])
                elif pred.size >= 2:
                    # assume [prob_real, prob_fake] or logits -> softmax
                    ex = np.exp(pred - np.max(pred))
                    probs = ex / ex.sum()
                    prob = float(probs[0])
                else:
                    prob = float(pred[0])
            else:
                prob = float(pred)
            # clamp to 0..1
            prob = max(0.0, min(1.0, prob))
            return prob
        except Exception as e:
            print("MLDetector predict error:", e)
            return None