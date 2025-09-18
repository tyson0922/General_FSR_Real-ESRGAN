import argparse
import cv2
import glob
import os
import torch
import base64
import json
import numpy as np
import logging
from pathlib import Path
from dotenv import load_dotenv

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

# --------------------------
# .env (OPENAI_API_KEY)
# --------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)


# --------------------------
# Helpers from main_test_bsrgan.py
# --------------------------
def hstack_with_labels(paths, labels, out_path, height=512, pad=8):
    """Horizontally stack images (resized to same height) and draw labels."""
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

    # Get the number of channels from the first image panel
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
            'Respond JSON: {"score":int,"reason":"..."}.'  # noqa: E501
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


# --------------------------
# Real-ESRGAN Model Loader
# --------------------------
# --------------------------
# Real-ESRGAN Model Loader
# --------------------------
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

    # Half precision unless fp32 is explicitly requested; allow --fp16 to force
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



# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Base output folder. If omitted, derives from input.')
    parser.add_argument("--mode", type=str, default="both", choices=["baseline", "lvlm", "both"],
                        help="baseline = pure Real-ESRGAN; lvlm = LVLM-guided; both = run both and make comparisons.")
    parser.add_argument('--model_name', type=str, default='RealESRGAN_x4plus', help='Model name for baseline.')
    parser.add_argument('--lvlm_model_name', type=str, default='RealESRGAN_x4plus', help='Model name for LVLM path.')
    parser.add_argument('-s', '--outscale', type=float, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face for all modes.')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision.')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    parser.add_argument('--fp16', action='store_true', help='Use FP16 (half precision) on CUDA')

    parser.add_argument(
        '--alpha_upsampler',
        type=str,
        default='realesrgan',
        help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument(
        '--ext',
        type=str,
        default='auto',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    parser.add_argument(
        '-g', '--gpu-id', type=int, default=None, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

    # LVLM flags
    parser.add_argument("--vlm_model", type=str, default="gpt-4o")
    parser.add_argument("--score_threshold", type=int, default=80)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--color_fix_strength", type=float, default=0.6, help="0..1 strength for color-cast correction")

    # Comparison flags
    parser.add_argument("--cmp_height", type=int, default=512, help="Comparison image panel height")

    args = parser.parse_args()

    # Set up paths
    L_path = args.input
    is_file = os.path.isfile(L_path)
    base_in = os.path.dirname(L_path) if is_file else L_path

    if args.output:
        base_out = args.output
    else:
        base_out = base_in

    base_baseline = os.path.join(base_out, "kaggle_dataset_baseline_mixed")
    base_lvlm = os.path.join(base_out, "kaggle_dataset_lvlm_mixed")
    base_cmp = os.path.join(base_out, "kaggle_dataset_cmp")

    os.makedirs(base_baseline, exist_ok=True)
    os.makedirs(base_lvlm, exist_ok=True)
    os.makedirs(base_cmp, exist_ok=True)

    # Get image paths
    img_paths = [L_path] if is_file else sorted(glob.glob(os.path.join(L_path, '*')))

    # Load Real-ESRGAN models (respect gpu-id and fp16/fp32 flags)
    upsampler_baseline = get_realesrgan_model(
        args.model_name, args.fp32,
        gpu_id=args.gpu_id,
        tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad,
        half_override=args.fp16
    )
    upsampler_lvlm = get_realesrgan_model(
        args.lvlm_model_name, args.fp32,
        gpu_id=args.gpu_id,
        tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad,
        half_override=args.fp16
    )

    face_enhancer = None
    if args.face_enhance:
        from gfpgan import GFPGANer
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler_baseline)  # Use a dummy upsampler as it's not used in this scenario

    print(f"Processing images from: {base_in}")

    for path in img_paths:
        imgname, extension = os.path.splitext(os.path.basename(path))

        baseline_sr_path = None
        lvlm_sr_path = None

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Skipping {path}: Could not read image.")
            continue

        img_mode = 'RGBA' if (len(img.shape) == 3 and img.shape[2] == 4) else None

        # ================== Pure FSR (Baseline) ==================
        if args.mode in ("baseline", "both"):
            print(f"--> Running Baseline on {imgname}{extension}")
            try:
                if args.face_enhance:
                    _, _, output_baseline = face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    # ⬇️ no tiling kwargs here
                    output_baseline, _ = upsampler_baseline.enhance(img, outscale=args.outscale)
            except RuntimeError as error:
                print(f'Error processing {imgname}: {error}')
                output_baseline = None

            if output_baseline is not None:
                baseline_sr_path = os.path.join(base_baseline, f'{imgname}_RealESRGAN.png')
                cv2.imwrite(baseline_sr_path, output_baseline)

        # ================== FSR + LVLM ==================
        if args.mode in ("lvlm", "both"):
            print(f"--> Running LVLM-Guided on {imgname}{extension}")
            best_candidate_path = None

            try:
                if args.face_enhance:
                    _, _, output_lvlm = face_enhancer.enhance(
                        img, has_aligned=False, only_center_face=False, paste_back=True)
                else:
                    # ⬇️ no tiling kwargs here
                    output_lvlm, _ = upsampler_lvlm.enhance(img, outscale=args.outscale)
            except RuntimeError as error:
                print(f'Error processing {imgname} for LVLM path: {error}')
                output_lvlm = None

            if output_lvlm is not None:
                lvlm_sr_path = os.path.join(base_lvlm, f'{imgname}_RealESRGAN.png')
                cv2.imwrite(lvlm_sr_path, output_lvlm)

                best_score, best_path, best_np = None, None, None
                score, reason = vlm_quality_score(path, lvlm_sr_path, args.vlm_model, OPENAI_API_KEY)
                reason_safe = sanitize_console(reason)
                print(f"LVLM Score: {score} ({reason_safe})")

                best_score, best_path, best_np = score, lvlm_sr_path, output_lvlm

                tried = 0
                while score < args.score_threshold and tried < args.max_retries:
                    tried += 1
                    retry_kind = None
                    out_np_retry = None
                    retry_path = None

                    if has_color_cast(reason):
                        retry_kind = "color_fix"
                        print(f"LVLM retry: attempt {tried} -> color_fix(s={args.color_fix_strength})")
                        out_np_retry = fix_color_cast(best_np, strength=args.color_fix_strength)
                        retry_path = os.path.join(base_lvlm, f'{imgname}_colorfix_retry{tried}.png')
                    elif has_halos(reason):
                        retry_kind = "dehalo"
                        print(f"LVLM retry: attempt {tried} -> dehalo")
                        out_np_retry = dehalo(best_np)
                        retry_path = os.path.join(base_lvlm, f'{imgname}_dehalo_retry{tried}.png')
                    elif looks_hallucinated(reason) or needs_more_detail(reason):
                        retry_kind = "gentle_boost"
                        print(f"LVLM retry: attempt {tried} -> gentle_boost")
                        out_np_retry = gentle_detail_boost(best_np)
                        retry_path = os.path.join(base_lvlm, f'{imgname}_boost_retry{tried}.png')
                    else:
                        print("LVLM retry: no safe policy -> stop")
                        break

                    if out_np_retry is not None:
                        cv2.imwrite(retry_path, out_np_retry)
                        score_alt, reason_alt = vlm_quality_score(path, retry_path, args.vlm_model, OPENAI_API_KEY)
                        print(f"LVLM Score (retry): {score_alt} ({sanitize_console(reason_alt)})")

                        if score_alt > best_score:
                            best_score, best_path, best_np = score_alt, retry_path, out_np_retry

                        score, reason = score_alt, reason_alt

                if best_path and best_path != lvlm_sr_path:
                    best_out = os.path.join(base_lvlm, f"{imgname}_best.png")
                    cv2.imwrite(best_out, best_np)
                    print(f"LVLM best: {best_score} -> {os.path.basename(best_path)}")

                best_candidate_path = os.path.join(base_lvlm, f"{imgname}_best.png")
                if not os.path.isfile(best_candidate_path):
                    best_candidate_path = lvlm_sr_path

        # ================== Comparisons ==================
        if args.mode == "both":
            panels = [path, baseline_sr_path, best_candidate_path]
            labels = ["LR", f"Baseline ({args.model_name})", f"LVLM-Guided ({args.lvlm_model_name})"]

            cmp_path = os.path.join(base_cmp, f"{imgname}_cmp.png")
            hstack_with_labels(panels, labels, cmp_path, height=args.cmp_height)
            print(f"--> Saved comparison to {cmp_path}")

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
