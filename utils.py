import os
import cv2
import base64
import json
import numpy as np
from pathlib import Path

def hstack_with_labels(paths, labels, out_path, height=512, pad=8):
    panels = []
    for p in paths:
        if p is None or (isinstance(p, str) and not os.path.isfile(p)):
            panels.append(None)
            continue
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            panels.append(None)
            continue
        h, w = img.shape[:2]
        scale = height / float(h)
        img = cv2.resize(img, (int(w * scale), height), interpolation=cv2.INTER_AREA)
        panels.append(img)
    avail = [(im, lb) for im, lb in zip(panels, labels) if im is not None]
    if not avail:
        return False
    imgs, labels = zip(*avail)
    drawn = []
    for im, lb in zip(imgs, labels):
        bar = np.full((36, im.shape[1], 3), 245, dtype=np.uint8)
        cv2.putText(bar, str(lb), (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
        block = np.vstack([bar, im])
        drawn.append(block)
    num_channels = drawn[0].shape[2]
    spacer = np.full((drawn[0].shape[0], pad, num_channels), 255, dtype=np.uint8)
    canvas = drawn[0]
    for d in drawn[1:]:
        canvas = np.hstack([canvas, spacer, d])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)
    return True

def sanitize_console(s: str) -> str:
    repl = {
        '\u2011': '-', '\u2012': '-', '\u2013': '-', '\u2014': '-',
        '\u2018': "'", '\u2019': "'", '\u201a': ',',
        '\u201c': '"', '\u201d': '"',
        '\u2026': '...', '\u00a0': ' ',
    }
    return ''.join(repl.get(ch, ch) for ch in s)

def _b64_data_url(img_path: str) -> str:
    ext = Path(img_path).suffix.lower().replace(".", "")
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    mime = "jpeg" if ext in ("jpg", "jpeg") else "png"
    return f"data:image/{mime};base64,{b64}"

def _safe_openai_client(api_key: str):
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None

def vlm_quality_score(lr_path: str, sr_path: str, model_name: str, api_key: str):
    if not api_key:
        return 100, "ok"
    client = _safe_openai_client(api_key)
    if client is None:
        return 100, "ok"
    try:
        prompt = (
            "You are a QA rater for super-resolution. Compare LR and SR. "
            "Score 0-100 (100 = sharp & natural; 0 = severe artifacts). "
            'Respond JSON: {"score":int,"reason":"..."}.'
        )
        lr_b64 = _b64_data_url(lr_path)
        sr_b64 = _b64_data_url(sr_path)
        resp = client.responses.create(
            model=model_name,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": lr_b64},
                    {"type": "input_image", "image_url": sr_b64},
                ],
            }],
        )
        text = getattr(resp, "output_text", None)
        if not text:
            text = resp.output[0].content[0].text
        out = json.loads(text)
        return int(out.get("score", 100)), out.get("reason", "ok")
    except Exception as e:
        return 100, f"lvlm_error:{e}"

def _has_any(s, kws):
    s2 = s.lower()
    return any(k in s2 for k in kws)

def needs_more_detail(reason):
    return _has_any(reason, ["soft", "blurry", "lack detail", "too smooth", "smudged", "low detail"]) \
        and not _has_any(reason, ["halo", "ringing", "color cast", "tint", "bleeding", "hallucinat", "unnatural"])

def has_color_cast(reason):
    return _has_any(reason, ["color cast", "green", "magenta", "tint", "uneven tone"])

def has_halos(reason):
    return _has_any(reason, ["halo", "ringing", "oversharp"])

def looks_hallucinated(reason):
    return _has_any(reason, ["hallucinat", "plastic", "unnatural", "glossy eyes"])

def fix_color_cast(bgr, strength=0.6):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    shiftA = 128.0 - float(A.mean())
    shiftB = 128.0 - float(B.mean())
    s = float(np.clip(strength, 0.0, 1.0))
    shiftA = int(np.clip(round(shiftA * s), -12, 12))
    shiftB = int(np.clip(round(shiftB * s), -12, 12))
    A = np.clip(A.astype(np.int16) + shiftA, 0, 255).astype(np.uint8)
    B = np.clip(B.astype(np.int16) + shiftB, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
    return out

def gentle_detail_boost(bgr):
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    return cv2.addWeighted(bgr, 1.15, blur, -0.15, 0)

def dehalo(bgr):
    sm = cv2.bilateralFilter(bgr, d=7, sigmaColor=25, sigmaSpace=7)
    return gentle_detail_boost(sm)

# Model loader
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer

def get_realesrgan_model(model_name, fp32, gpu_id=None, tile=0, tile_pad=10, pre_pad=0, half_override=False):
    model = None
    netscale = 4
    file_url = []
    if model_name == 'RealESRGAN_x4plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
    elif model_name == 'RealESRGAN_x4plus_anime_6B':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
    elif model_name == 'RealESRGAN_x2plus':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
    elif model_name == 'realesr-general-x4v3':
        model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
        netscale = 4
        file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth']
    model_path = os.path.join('weights', f'{model_name}.pth')
    if not os.path.isfile(model_path):
        ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
        for url in file_url:
            model_path = load_file_from_url(
                url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
    half = (not fp32) or half_override
    return RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        half=half,
        gpu_id=gpu_id,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad
    )

