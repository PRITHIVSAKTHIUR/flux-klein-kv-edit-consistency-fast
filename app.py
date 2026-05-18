import os
import io
import gc
import uuid
import json
import base64
import random
import subprocess
from pathlib import Path
from typing import List, Optional

import spaces
import numpy as np
import torch
from PIL import Image

from gradio import Server
from fastapi import Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

HF_TOKEN = os.environ.get("HF_TOKEN")

app = Server()

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "outputs"
EXAMPLES_DIR = BASE_DIR / "examples"

STATIC_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

ADAPTER = {
    "title": "Klein-Consistency",
    "adapter_name": "klein-consistency",
    "repo": "dx8152/Flux2-Klein-9B-Consistency",
    "weights": "Klein-consistency.safetensors",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    DEVICE_LABEL = torch.cuda.get_device_name(torch.cuda.current_device()).lower()
else:
    DEVICE_LABEL = str(DEVICE).lower()

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__ =", torch.__version__)
print("Using device:", DEVICE)


def apply_patch():
    import diffusers

    site_packages = os.path.dirname(diffusers.__file__)
    patch_file = os.path.join(os.path.dirname(__file__), "flux2_klein_kv.patch")
    if os.path.exists(patch_file):
        result = subprocess.run(
            ["patch", "-p2", "--forward", "--batch"],
            cwd=os.path.dirname(site_packages),
            stdin=open(patch_file),
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("Patch applied successfully")
        else:
            print(f"Patch output: {result.stdout}\n{result.stderr}")


apply_patch()

from diffusers.pipelines.flux2.pipeline_flux2_klein_kv import Flux2KleinKVPipeline

print("Loading FLUX.2 Klein 9B KV model...")
pipe = Flux2KleinKVPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9b-kv",
    torch_dtype=dtype,
    token=HF_TOKEN,
).to(DEVICE)
print("Base KV Model loaded successfully.")

print(f"Loading adapter: {ADAPTER['title']}")
pipe.load_lora_weights(
    ADAPTER["repo"],
    weight_name=ADAPTER["weights"],
    adapter_name=ADAPTER["adapter_name"],
)
pipe.set_adapters([ADAPTER["adapter_name"]], adapter_weights=[1.0])
print(f"Adapter loaded successfully: {ADAPTER['adapter_name']}")


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def save_image(img: Image.Image, prefix: str = "output") -> str:
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    path = OUTPUT_DIR / filename
    img.save(path, format="PNG")
    return filename


def update_dimensions_on_upload(image):
    if image is None:
        return 1024, 1024

    try:
        if isinstance(image, list) and len(image) > 0:
            first = image[0]
        else:
            first = image

        if isinstance(first, (tuple, list)):
            path_or_img = first[0]
        else:
            path_or_img = first

        if isinstance(path_or_img, str):
            img = Image.open(path_or_img).convert("RGB")
        elif isinstance(path_or_img, Image.Image):
            img = path_or_img.convert("RGB")
        else:
            img = Image.open(path_or_img.name).convert("RGB")

        original_width, original_height = img.size

        if original_width > original_height:
            new_width = 1024
            aspect_ratio = original_height / original_width
            new_height = int(new_width * aspect_ratio)
        else:
            new_height = 1024
            aspect_ratio = original_width / original_height
            new_width = int(new_height * aspect_ratio)

        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8

        new_width = max(256, min(1024, new_width))
        new_height = max(256, min(1024, new_height))

        return new_width, new_height
    except Exception:
        return 1024, 1024


def process_gallery_images(images):
    if not images:
        return []

    pil_images = []
    for item in images:
        try:
            if isinstance(item, (tuple, list)):
                path_or_img = item[0]
            else:
                path_or_img = item

            if isinstance(path_or_img, str):
                pil_images.append(Image.open(path_or_img).convert("RGB"))
            elif isinstance(path_or_img, Image.Image):
                pil_images.append(path_or_img.convert("RGB"))
            else:
                pil_images.append(Image.open(path_or_img.name).convert("RGB"))
        except Exception as e:
            print(f"Skipping invalid image item: {e}")
            continue

    return pil_images


@spaces.GPU(size="xlarge")
def infer(
    images,
    prompt,
    seed,
    randomize_seed,
    width,
    height,
    steps,
):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not prompt or not str(prompt).strip():
        raise ValueError("Please enter a prompt.")

    if isinstance(seed, str):
        seed = int(seed)
    if isinstance(randomize_seed, str):
        randomize_seed = randomize_seed.lower() == "true"
    if isinstance(width, str):
        width = int(width)
    if isinstance(height, str):
        height = int(height)
    if isinstance(steps, str):
        steps = int(steps)

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    pil_images = process_gallery_images(images) if images else []

    if pil_images:
        width, height = update_dimensions_on_upload(pil_images[0])
        image_input = [
            img.resize((width, height), Image.LANCZOS).convert("RGB")
            for img in pil_images
        ]
    else:
        image_input = None
        width = max(256, min(MAX_IMAGE_SIZE, (int(width) // 8) * 8))
        height = max(256, min(MAX_IMAGE_SIZE, (int(height) // 8) * 8))

    try:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        pipe_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "generator": generator,
        }

        if image_input is not None:
            pipe_kwargs["image"] = image_input

        result_image = pipe(**pipe_kwargs).images[0]
        return result_image, seed

    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_example_items():
    example_prompts = {
        "1.jpg": "Change the weather to stormy.",
        "2.jpg": "Transform the scene into a snowy winter day while preserving the original subject identity, framing, and composition.",
        "3.jpg": "Relight the image with soft golden sunset lighting while keeping all structures and subject details consistent.",
        "4.jpg": "Make the texture high-resolution.",
    }

    items = []
    if EXAMPLES_DIR.exists():
        for name in sorted(os.listdir(EXAMPLES_DIR)):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                items.append(
                    {
                        "file": name,
                        "url": f"/example-file/{name}",
                        "prompt": example_prompts.get(name, "Edit this image while preserving composition."),
                    }
                )
    return items


@app.api(name="hello")
def hello(name: str) -> str:
    return f"Hello, {name}!"


@app.get("/example-file/{filename}")
async def example_file(filename: str):
    path = EXAMPLES_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "Example not found"}, status_code=404)
    return FileResponse(path)


@app.get("/download/{filename}")
async def download_file(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, filename=filename, media_type="image/png")


@app.post("/api/edit")
async def edit_image(
    prompt: str = Form(...),
    seed: str = Form("0"),
    randomize_seed: str = Form("true"),
    width: str = Form("1024"),
    height: str = Form("1024"),
    steps: str = Form("4"),
    images: Optional[List[UploadFile]] = File(None),
):
    temp_paths = []
    try:
        image_paths = []

        if images:
            for upload in images:
                suffix = Path(upload.filename).suffix or ".png"
                temp_name = f"upload_{uuid.uuid4().hex}{suffix}"
                temp_path = OUTPUT_DIR / temp_name
                content = await upload.read()
                with open(temp_path, "wb") as f:
                    f.write(content)
                temp_paths.append(str(temp_path))
                image_paths.append(str(temp_path))

        result_image, used_seed = infer(
            images=image_paths,
            prompt=prompt,
            seed=seed,
            randomize_seed=randomize_seed,
            width=width,
            height=height,
            steps=steps,
        )

        output_filename = save_image(result_image, prefix="kv_edit")
        return JSONResponse(
            {
                "success": True,
                "seed": used_seed,
                "image_url": f"/download/{output_filename}",
                "download_url": f"/download/{output_filename}",
                "image_base64": image_to_base64(result_image),
                "device": DEVICE_LABEL,
            }
        )

    except Exception as e:
        return JSONResponse(
            {"success": False, "error": str(e)},
            status_code=500,
        )
    finally:
        for p in temp_paths:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    examples = get_example_items()
    examples_json = json.dumps(examples)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>KV-Edit-Consistency</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap');

    :root {{
      --bg: #0b0b10;
      --panel: #111218;
      --panel-2: #151621;
      --panel-3: #1b1d2a;
      --border: #242638;
      --muted: #9ca3af;
      --text: #f5f7fb;
      --text-dim: #c5cad3;
      --purple: #7c3aed;
      --purple-hover: #6d28d9;
      --purple-soft: rgba(124,58,237,0.14);
      --green: #22c55e;
      --green-soft: rgba(34,197,94,0.14);
      --red: #ef4444;
      --red-soft: rgba(239,68,68,0.14);
      --yellow: #f59e0b;
      --input-bg: #0f1017;
      --same-height: 760px;
    }}

    * {{
      box-sizing: border-box;
      border-radius: 0 !important;
    }}

    html, body {{
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      min-height: 100%;
    }}

    body {{
      overflow-x: hidden;
    }}

    .app-shell {{
      min-height: 100vh;
      background:
        linear-gradient(to bottom, rgba(124,58,237,0.08), transparent 160px),
        var(--bg);
    }}

    .topbar {{
      height: 56px;
      border-bottom: 1px solid var(--border);
      background: #0a0b11;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0 24px;
      color: #d7cdfc;
      font-size: 14px;
      font-weight: 600;
      letter-spacing: 0.02em;
    }}

    .container {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px;
    }}

    .hero {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 24px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
    }}

    .hero-left {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}

    .eyebrow {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 500;
    }}

    .title-row {{
      display: flex;
      align-items: center;
      gap: 14px;
      flex-wrap: wrap;
    }}

    .title {{
      font-size: 44px;
      line-height: 1;
      font-weight: 800;
      margin: 0;
      letter-spacing: -0.03em;
    }}

    .hero-tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .tag {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      height: 34px;
      padding: 0 12px;
      border: 1px solid var(--border);
      font-size: 13px;
      font-weight: 700;
      letter-spacing: 0.01em;
    }}

    .tag svg {{
      width: 15px;
      height: 15px;
      flex-shrink: 0;
    }}

    .tag-purple {{
      color: #d8ccff;
      background: var(--purple-soft);
      border-color: rgba(124,58,237,0.35);
    }}

    .tag-green {{
      color: #bbf7d0;
      background: var(--green-soft);
      border-color: rgba(34,197,94,0.35);
    }}

    .tag-red {{
      color: #fecaca;
      background: var(--red-soft);
      border-color: rgba(239,68,68,0.35);
    }}

    .hero-actions {{
      display: flex;
      gap: 10px;
      flex-shrink: 0;
    }}

    .ghost-btn {{
      height: 40px;
      padding: 0 14px;
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      font-family: 'Outfit', sans-serif;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }}

    .ghost-btn:hover {{
      background: var(--panel-2);
    }}

    .layout {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      align-items: stretch;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      min-height: var(--same-height);
      height: var(--same-height);
      display: flex;
      flex-direction: column;
      overflow: hidden;
    }}

    .panel-header {{
      height: 62px;
      min-height: 62px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      background: #101119;
    }}

    .panel-title {{
      font-size: 22px;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin: 0;
    }}

    .panel-header-right {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }}

    .status-pill {{
      padding: 5px 8px;
      background: var(--panel-3);
      border: 1px solid var(--border);
      color: var(--muted);
      font-size: 12px;
      line-height: 1;
      transition: all 0.2s ease;
    }}

    .status-pill.active {{
      background: rgba(245,158,11,0.12);
      border-color: rgba(245,158,11,0.35);
      color: #fbbf24;
    }}

    .status-pill.idle {{
      background: var(--panel-3);
      border: 1px solid var(--border);
      color: var(--muted);
    }}

    .panel-body {{
      flex: 1;
      min-height: 0;
      padding: 18px;
      overflow: auto;
    }}

    .form-stack {{
      display: flex;
      flex-direction: column;
      gap: 18px;
      height: 100%;
    }}

    .form-group {{
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .label {{
      font-size: 14px;
      font-weight: 600;
      color: var(--muted);
      letter-spacing: 0.02em;
      text-transform: none;
    }}

    .hint {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
      margin-top: -4px;
    }}

    textarea,
    input,
    button,
    select {{
      font-family: 'Outfit', sans-serif;
    }}

    .input,
    .textarea {{
      width: 100%;
      background: var(--input-bg);
      border: 1px solid var(--border);
      color: var(--text);
      outline: none;
      padding: 14px 14px;
      font-size: 15px;
    }}

    .input:focus,
    .textarea:focus {{
      border-color: #3a3d56;
      background: #11131b;
    }}

    .textarea {{
      min-height: 200px;
      resize: vertical;
      line-height: 1.55;
    }}

    .upload-wrap {{
      background: var(--input-bg);
      border: 1px dashed #32354b;
      min-height: 220px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      padding: 14px;
      cursor: pointer;
    }}

    .upload-wrap.dragover {{
      border-color: var(--purple);
      background: rgba(124,58,237,0.08);
    }}

    .upload-wrap input[type="file"] {{
      display: none;
    }}

    .upload-placeholder {{
      min-height: 190px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
      background: transparent;
      border: none;
      color: var(--text-dim);
      cursor: pointer;
      padding: 16px;
      text-align: center;
    }}

    .upload-icon {{
      width: 56px;
      height: 56px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #d8ccff;
    }}

    .preview-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(112px, 1fr));
      gap: 12px;
    }}

    .thumb {{
      position: relative;
      aspect-ratio: 1/1;
      overflow: hidden;
      border: 1px solid var(--border);
      background: #0b0c12;
    }}

    .thumb img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }}

    .thumb-remove {{
      position: absolute;
      top: 6px;
      right: 6px;
      width: 26px;
      height: 26px;
      border: 1px solid var(--border);
      background: rgba(11,11,16,0.88);
      color: white;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 16px;
      line-height: 1;
    }}

    .advanced {{
      border: 1px solid var(--border);
      background: #0f1017;
    }}

    .advanced-toggle {{
      width: 100%;
      height: 48px;
      border: none;
      border-bottom: 1px solid var(--border);
      background: transparent;
      color: var(--text);
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 14px;
      cursor: pointer;
      font-size: 14px;
      font-weight: 600;
    }}

    .advanced-toggle:hover {{
      background: #121420;
    }}

    .advanced-body {{
      display: none;
      padding: 14px;
    }}

    .advanced-body.open {{
      display: block;
    }}

    .advanced-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}

    .checkbox-row {{
      margin-top: 14px;
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--text-dim);
      font-size: 14px;
      font-weight: 500;
    }}

    .checkbox-row input {{
      width: 16px;
      height: 16px;
      accent-color: var(--purple);
    }}

    .actions {{
      margin-top: auto;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding-top: 8px;
    }}

    .btn {{
      height: 48px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      cursor: pointer;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0.01em;
    }}

    .btn:hover {{
      background: #1a1d29;
    }}

    .btn-primary {{
      background: var(--purple);
      border-color: var(--purple);
      color: white;
    }}

    .btn-primary:hover {{
      background: var(--purple-hover);
      border-color: var(--purple-hover);
    }}

    .result-shell {{
      height: 100%;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }}

    .result-stage {{
      position: relative;
      flex: 1;
      min-height: 0;
      border: 1px solid var(--border);
      background: #0d0e14;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}

    .result-stage img {{
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      display: none;
      position: relative;
      z-index: 1;
    }}

    .result-empty {{
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 14px;
      color: var(--text-dim);
      text-align: center;
      padding: 24px;
      position: relative;
      z-index: 1;
      transition: filter 0.25s ease, opacity 0.25s ease;
    }}

    .result-empty-box {{
      width: 72px;
      height: 72px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      display: flex;
      align-items: center;
      justify-content: center;
      color: #d8ccff;
    }}

    .download-fab {{
      position: absolute;
      top: 12px;
      right: 12px;
      width: 42px;
      height: 42px;
      border: 1px solid var(--border);
      background: rgba(17,18,24,0.92);
      color: white;
      display: none;
      align-items: center;
      justify-content: center;
      text-decoration: none;
      z-index: 4;
    }}

    .result-stage.has-image:hover .download-fab {{
      display: flex;
    }}

    .loader {{
      position: absolute;
      inset: 0;
      display: none;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 14px;
      background: rgba(7,8,12,0.34);
      backdrop-filter: blur(7px);
      -webkit-backdrop-filter: blur(7px);
      z-index: 3;
      pointer-events: none;
    }}

    .circle-loader {{
      width: 58px;
      height: 58px;
      border-radius: 50% !important;
      border: 4px solid rgba(255,255,255,0.14);
      border-top-color: #ffffff;
      border-right-color: #c4b5fd;
      animation: spin 0.9s linear infinite;
      box-shadow: 0 0 20px rgba(124,58,237,0.18);
    }}

    .loader span {{
      font-size: 14px;
      font-weight: 600;
      color: #ffffff;
      letter-spacing: 0.02em;
      text-shadow: 0 1px 2px rgba(0,0,0,0.35);
    }}

    .result-stage.processing .result-empty,
    .result-stage.processing img {{
      filter: blur(2px);
    }}

    .result-meta {{
      display: flex;
      align-items: stretch;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }}

    .meta-card {{
      border: 1px solid var(--border);
      background: var(--panel-2);
      padding: 12px 14px;
      min-width: 180px;
      flex: 1 1 240px;
    }}

    .meta-label {{
      font-size: 12px;
      color: var(--muted);
      text-transform: lowercase;
      letter-spacing: 0.04em;
      margin-bottom: 6px;
    }}

    .meta-value {{
      font-size: 14px;
      font-weight: 700;
      color: var(--text);
      word-break: break-word;
      line-height: 1.45;
      text-transform: lowercase;
    }}

    .examples-panel {{
      margin-top: 24px;
      background: var(--panel);
      border: 1px solid var(--border);
      overflow: hidden;
    }}

    .examples-header {{
      height: 58px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      padding: 0 18px;
      font-size: 20px;
      font-weight: 700;
      background: #101119;
    }}

    .examples-body {{
      padding: 18px;
    }}

    .examples-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}

    .example-card {{
      background: #0f1017;
      border: 1px solid var(--border);
      cursor: pointer;
      overflow: hidden;
    }}

    .example-card:hover {{
      border-color: #3a3d56;
      background: #121420;
    }}

    .example-card img {{
      width: 100%;
      aspect-ratio: 1/1;
      object-fit: cover;
      display: block;
      border-bottom: 1px solid var(--border);
    }}

    .example-body {{
      padding: 12px;
    }}

    .example-body p {{
      margin: 0;
      color: var(--text-dim);
      font-size: 13px;
      line-height: 1.5;
      font-weight: 500;
    }}

    .toast-wrap {{
      position: fixed;
      top: 18px;
      right: 18px;
      z-index: 9999;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}

    .toast {{
      min-width: 260px;
      max-width: 360px;
      background: #141623;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 12px 14px;
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }}

    .toast button {{
      border: none;
      background: transparent;
      color: var(--text);
      font-size: 18px;
      cursor: pointer;
      padding: 0;
      line-height: 1;
    }}

    @keyframes spin {{
      from {{ transform: rotate(0deg); }}
      to {{ transform: rotate(360deg); }}
    }}

    @media (max-width: 1200px) {{
      :root {{
        --same-height: 720px;
      }}
      .examples-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}

    @media (max-width: 980px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .panel {{
        min-height: auto;
        height: auto;
      }}
      .panel-body {{
        overflow: visible;
      }}
      .result-stage {{
        min-height: 460px;
      }}
      .hero {{
        flex-direction: column;
      }}
    }}

    @media (max-width: 640px) {{
      .container {{
        padding: 16px;
      }}
      .title {{
        font-size: 32px;
      }}
      .advanced-grid,
      .actions {{
        grid-template-columns: 1fr;
      }}
      .examples-grid {{
        grid-template-columns: 1fr;
      }}
      .panel-header {{
        padding: 0 14px;
      }}
      .panel-body {{
        padding: 14px;
      }}
      .hero-tags {{
        gap: 8px;
      }}
      .tag {{
        width: 100%;
        justify-content: center;
      }}
      .textarea {{
        min-height: 220px;
      }}
    }}
  </style>
</head>
<body>
  <div class="toast-wrap" id="toastWrap"></div>

  <div class="app-shell">
    <div class="topbar">4-Step Fast Inference KV Image Editing Playground</div>

    <div class="container">
      <section class="hero">
        <div class="hero-left">
          <div class="eyebrow">black-forest-labs / flux.2-klein-9b-kv / edit</div>
          <div class="title-row">
            <h1 class="title">KV-Edit-Consistency</h1>
          </div>

          <div class="hero-tags">
            <div class="tag tag-purple">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 3v18"></path>
                <path d="M3 12h18"></path>
              </svg>
              <span>Inference</span>
            </div>

            <div class="tag tag-green">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="5" width="18" height="14"></rect>
                <path d="M8 13l2.5-2.5L13 13"></path>
                <path d="M13 13l2-2 3 3"></path>
              </svg>
              <span>image-to-image</span>
            </div>

            <div class="tag tag-red">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z"></path>
              </svg>
              <span>fast-edit</span>
            </div>
          </div>
        </div>

        <div class="hero-actions">
          <button class="ghost-btn" type="button" onclick="document.getElementById('examplesSection').scrollIntoView({{behavior:'smooth'}})">Examples</button>
        </div>
      </section>

      <section class="layout">
        <div class="panel">
          <div class="panel-header">
            <h2 class="panel-title">Input</h2>
            <div class="panel-header-right">
              <span class="status-pill">Form</span>
            </div>
          </div>
          <div class="panel-body">
            <div class="form-stack">
              <div class="form-group">
                <div class="label">Images</div>
                <div class="upload-wrap" id="uploadZone">
                  <input id="fileInput" type="file" accept="image/*" multiple />
                  <button class="upload-placeholder" id="uploadPlaceholder" type="button">
                    <div class="upload-icon">
                      <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="1.8">
                        <path d="M12 4v10"></path>
                        <path d="M8.5 7.5 12 4l3.5 3.5"></path>
                        <path d="M4 16.5h16"></path>
                        <path d="M6 20h12"></path>
                      </svg>
                    </div>
                    <div>
                      <div style="font-weight:700; color:var(--text); margin-bottom:4px;">Upload one or more images</div>
                      <div style="font-size:13px; color:var(--muted);">Drag and drop or click to browse</div>
                    </div>
                  </button>
                  <div class="preview-grid" id="previewGrid" style="display:none;"></div>
                </div>
                <div class="hint">The first uploaded image is used to auto-fit width and height while preserving aspect ratio.</div>
              </div>

              <div class="form-group">
                <label class="label" for="prompt">Edit Prompt</label>
                <textarea id="prompt" class="textarea" placeholder="Describe the edit you want to apply..."></textarea>
              </div>

              <div class="advanced">
                <button class="advanced-toggle" id="advancedToggle" type="button">
                  <span>Advanced Settings</span>
                  <span id="advancedIcon" style="font-size:22px; font-weight:700; line-height:1;">+</span>
                </button>
                <div class="advanced-body" id="advancedBody">
                  <div class="advanced-grid">
                    <div class="form-group">
                      <label class="label" for="seed">seed</label>
                      <input id="seed" class="input" type="number" min="0" max="{MAX_SEED}" value="0" />
                    </div>
                    <div class="form-group">
                      <label class="label" for="steps">steps</label>
                      <input id="steps" class="input" type="number" min="1" max="20" value="4" />
                    </div>
                    <div class="form-group">
                      <label class="label" for="width">width</label>
                      <input id="width" class="input" type="number" min="256" max="{MAX_IMAGE_SIZE}" step="8" value="1024" />
                    </div>
                    <div class="form-group">
                      <label class="label" for="height">height</label>
                      <input id="height" class="input" type="number" min="256" max="{MAX_IMAGE_SIZE}" step="8" value="1024" />
                    </div>
                  </div>
                  <div class="checkbox-row">
                    <input id="randomizeSeed" type="checkbox" checked />
                    <label for="randomizeSeed">Randomize seed</label>
                  </div>
                </div>
              </div>

              <div class="actions">
                <button class="btn btn-primary" id="runBtn" type="button">Edit Image</button>
                <button class="btn" id="clearBtn" type="button">Clear</button>
              </div>
            </div>
          </div>
        </div>

        <div class="panel">
          <div class="panel-header">
            <h2 class="panel-title">Result</h2>
            <div class="panel-header-right">
              <span class="status-pill idle" id="resultStatus">Idle</span>
            </div>
          </div>
          <div class="panel-body">
            <div class="result-shell">
              <div class="result-stage" id="resultStage">
                <div class="result-empty" id="outputEmpty">
                  <div class="result-empty-box">
                    <svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.8">
                      <rect x="4" y="5" width="16" height="11"></rect>
                      <path d="M8 12l2.5-2.5L13 12"></path>
                      <path d="M13 12l2-2 2 2"></path>
                      <path d="M12 16v4"></path>
                    </svg>
                  </div>
                  <div>
                    <div style="font-size:17px; font-weight:700; color:var(--text); margin-bottom:4px;">No output yet</div>
                    <div style="font-size:14px; color:var(--muted);">Your edited image will appear here</div>
                  </div>
                </div>

                <img id="outputImage" alt="Generated output" />
                <a id="downloadLink" class="download-fab" download title="Download image">
                  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.4">
                    <path d="M12 4v10"></path>
                    <path d="m7.5 10.5 4.5 4.5 4.5-4.5"></path>
                    <path d="M5 20h14"></path>
                  </svg>
                </a>

                <div class="loader" id="loaderOverlay">
                  <div class="circle-loader"></div>
                  <span>Processing image</span>
                </div>
              </div>

              <div class="result-meta">
                <div class="meta-card">
                  <div class="meta-label">seed</div>
                  <div class="meta-value" id="usedSeed">-</div>
                </div>
                <div class="meta-card">
                  <div class="meta-label">device name</div>
                  <div class="meta-value" id="deviceValue">{DEVICE_LABEL}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section class="examples-panel" id="examplesSection">
        <div class="examples-header">Examples</div>
        <div class="examples-body">
          <div class="examples-grid" id="examplesGrid"></div>
        </div>
      </section>
    </div>
  </div>

  <script>
    const examples = {examples_json};

    const state = {{
      files: [],
      advancedOpen: false
    }};

    const uploadZone = document.getElementById("uploadZone");
    const fileInput = document.getElementById("fileInput");
    const uploadPlaceholder = document.getElementById("uploadPlaceholder");
    const previewGrid = document.getElementById("previewGrid");

    const promptEl = document.getElementById("prompt");
    const seedEl = document.getElementById("seed");
    const stepsEl = document.getElementById("steps");
    const widthEl = document.getElementById("width");
    const heightEl = document.getElementById("height");
    const randomizeSeedEl = document.getElementById("randomizeSeed");

    const advancedToggle = document.getElementById("advancedToggle");
    const advancedBody = document.getElementById("advancedBody");
    const advancedIcon = document.getElementById("advancedIcon");

    const runBtn = document.getElementById("runBtn");
    const clearBtn = document.getElementById("clearBtn");

    const resultStage = document.getElementById("resultStage");
    const resultStatus = document.getElementById("resultStatus");
    const outputImage = document.getElementById("outputImage");
    const outputEmpty = document.getElementById("outputEmpty");
    const loaderOverlay = document.getElementById("loaderOverlay");
    const usedSeed = document.getElementById("usedSeed");
    const downloadLink = document.getElementById("downloadLink");
    const deviceValue = document.getElementById("deviceValue");

    const examplesGrid = document.getElementById("examplesGrid");
    const toastWrap = document.getElementById("toastWrap");

    function showToast(message) {{
      const toast = document.createElement("div");
      toast.className = "toast";

      const text = document.createElement("div");
      text.textContent = message;

      const btn = document.createElement("button");
      btn.type = "button";
      btn.innerHTML = "&times;";
      btn.addEventListener("click", () => toast.remove());

      toast.appendChild(text);
      toast.appendChild(btn);
      toastWrap.appendChild(toast);

      setTimeout(() => {{
        toast.remove();
      }}, 4200);
    }}

    function setResultStatus(isActive) {{
      resultStatus.textContent = isActive ? "Active" : "Idle";
      resultStatus.classList.remove("active", "idle");
      resultStatus.classList.add(isActive ? "active" : "idle");
    }}

    function setLoading(loading) {{
      loaderOverlay.style.display = loading ? "flex" : "none";
      resultStage.classList.toggle("processing", loading);
      runBtn.disabled = loading;
      clearBtn.disabled = loading;
      runBtn.style.opacity = loading ? "0.8" : "1";
      clearBtn.style.opacity = loading ? "0.8" : "1";
      runBtn.style.cursor = loading ? "not-allowed" : "pointer";
      clearBtn.style.cursor = loading ? "not-allowed" : "pointer";
      setResultStatus(loading);
    }}

    function setAdvanced(open) {{
      state.advancedOpen = open;
      advancedBody.classList.toggle("open", open);
      advancedIcon.textContent = open ? "−" : "+";
    }}

    advancedToggle.addEventListener("click", () => {{
      setAdvanced(!state.advancedOpen);
    }});

    function createThumb(file, index) {{
      const wrapper = document.createElement("div");
      wrapper.className = "thumb";

      const img = document.createElement("img");
      img.src = URL.createObjectURL(file);
      img.alt = file.name || `upload-${{index}}`;

      const removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "thumb-remove";
      removeBtn.innerHTML = "&times;";
      removeBtn.title = "Remove image";
      removeBtn.addEventListener("click", (e) => {{
        e.stopPropagation();
        state.files.splice(index, 1);
        renderPreviews();
      }});

      wrapper.appendChild(img);
      wrapper.appendChild(removeBtn);
      return wrapper;
    }}

    function renderPreviews() {{
      previewGrid.innerHTML = "";

      if (!state.files.length) {{
        uploadPlaceholder.style.display = "flex";
        previewGrid.style.display = "none";
        return;
      }}

      uploadPlaceholder.style.display = "none";
      previewGrid.style.display = "grid";

      state.files.forEach((file, index) => {{
        previewGrid.appendChild(createThumb(file, index));
      }});
    }}

    function addFiles(fileList) {{
      const valid = Array.from(fileList).filter((file) => file.type.startsWith("image/"));
      if (!valid.length) {{
        showToast("Please upload valid image files.");
        return;
      }}
      state.files = [...state.files, ...valid];
      renderPreviews();
    }}

    uploadPlaceholder.addEventListener("click", () => fileInput.click());

    uploadZone.addEventListener("click", (e) => {{
      if (e.target === uploadZone) fileInput.click();
    }});

    fileInput.addEventListener("change", (e) => {{
      addFiles(e.target.files);
      fileInput.value = "";
    }});

    uploadZone.addEventListener("dragover", (e) => {{
      e.preventDefault();
      uploadZone.classList.add("dragover");
    }});

    uploadZone.addEventListener("dragleave", () => {{
      uploadZone.classList.remove("dragover");
    }});

    uploadZone.addEventListener("drop", (e) => {{
      e.preventDefault();
      uploadZone.classList.remove("dragover");
      if (e.dataTransfer.files?.length) {{
        addFiles(e.dataTransfer.files);
      }}
    }});

    function clearAll() {{
      state.files = [];
      renderPreviews();
      promptEl.value = "";
      seedEl.value = "0";
      stepsEl.value = "4";
      widthEl.value = "1024";
      heightEl.value = "1024";
      randomizeSeedEl.checked = true;

      outputImage.style.display = "none";
      outputImage.removeAttribute("src");
      outputEmpty.style.display = "flex";
      usedSeed.textContent = "-";
      deviceValue.textContent = "{DEVICE_LABEL}";
      downloadLink.style.display = "none";
      downloadLink.removeAttribute("href");
      resultStage.classList.remove("has-image", "processing");

      setResultStatus(false);
      setAdvanced(false);
      setLoading(false);
    }}

    clearBtn.addEventListener("click", clearAll);

    async function fileFromUrl(url, filename = "example.jpg") {{
      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to fetch example image.");
      const blob = await res.blob();
      return new File([blob], filename, {{ type: blob.type || "image/jpeg" }});
    }}

    function renderExamples() {{
      examplesGrid.innerHTML = "";

      examples.forEach((item) => {{
        const card = document.createElement("div");
        card.className = "example-card";

        const img = document.createElement("img");
        img.src = item.url;
        img.alt = item.file;

        const body = document.createElement("div");
        body.className = "example-body";

        const text = document.createElement("p");
        text.textContent = item.prompt;

        body.appendChild(text);
        card.appendChild(img);
        card.appendChild(body);

        card.addEventListener("click", async () => {{
          try {{
            const file = await fileFromUrl(item.url, item.file);
            state.files = [file];
            renderPreviews();
            promptEl.value = item.prompt;
            showToast("Example loaded.");
          }} catch (err) {{
            showToast(err.message || "Failed to load example.");
          }}
        }});

        examplesGrid.appendChild(card);
      }});
    }}

    async function submitEdit() {{
      try {{
        const prompt = promptEl.value.trim();

        if (!prompt) {{
          showToast("Please enter a prompt.");
          return;
        }}

        const formData = new FormData();
        formData.append("prompt", prompt);
        formData.append("seed", seedEl.value || "0");
        formData.append("randomize_seed", String(randomizeSeedEl.checked));
        formData.append("width", widthEl.value || "1024");
        formData.append("height", heightEl.value || "1024");
        formData.append("steps", stepsEl.value || "4");

        state.files.forEach((file) => formData.append("images", file));

        setLoading(true);

        const res = await fetch("/api/edit", {{
          method: "POST",
          body: formData
        }});

        const data = await res.json();

        if (!res.ok || !data.success) {{
          throw new Error(data.error || "Processing failed.");
        }}

        outputImage.onload = () => {{
          resultStage.classList.add("has-image");
        }};

        outputImage.src = data.image_url + "?t=" + Date.now();
        outputImage.style.display = "block";
        outputEmpty.style.display = "none";
        usedSeed.textContent = String(data.seed).toLowerCase();
        deviceValue.textContent = (data.device || "{DEVICE_LABEL}").toLowerCase();
        downloadLink.href = data.download_url;
      }} catch (err) {{
        showToast(err.message || "An unexpected error occurred.");
      }} finally {{
        setLoading(false);
      }}
    }}

    runBtn.addEventListener("click", submitEdit);

    setAdvanced(false);
    setResultStatus(false);
    renderExamples();
    renderPreviews();
  </script>
</body>
</html>
"""


app.launch()
