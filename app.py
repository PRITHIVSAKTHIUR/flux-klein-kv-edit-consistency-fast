import os
import gc
import random
import subprocess
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple

import spaces
import gradio as gr

from typing import Iterable

# --------------------------- theme ---------------------------

from gradio.themes import Soft
from gradio.themes.utils import colors, fonts, sizes

colors.orange_red = colors.Color(
    name="orange_red", c50="#FFF0E5", c100="#FFE0CC", c200="#FFC299", c300="#FFA366",
    c400="#FF8533", c500="#FF4500", c600="#E63E00", c700="#CC3700", c800="#B33000",
    c900="#992900", c950="#802200",
)

class OrangeRedTheme(Soft):
    def __init__(
        self, *, primary_hue: colors.Color | str = colors.gray,
        secondary_hue: colors.Color | str = colors.orange_red,
        neutral_hue: colors.Color | str = colors.slate, text_size: sizes.Size | str = sizes.text_lg,
        font: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Outfit"), "Arial", "sans-serif",
        ),
        font_mono: fonts.Font | str | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue, secondary_hue=secondary_hue, neutral_hue=neutral_hue,
            text_size=text_size, font=font, font_mono=font_mono,
        )
        super().set(
            background_fill_primary="*primary_50",
            background_fill_primary_dark="*primary_900",
            body_background_fill="linear-gradient(135deg, *primary_200, *primary_100)",
            body_background_fill_dark="linear-gradient(135deg, *primary_900, *primary_800)",
            button_primary_text_color="white",
            button_primary_text_color_hover="white",
            button_primary_background_fill="linear-gradient(90deg, *secondary_500, *secondary_600)",
            button_primary_background_fill_hover="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_dark="linear-gradient(90deg, *secondary_600, *secondary_700)",
            button_primary_background_fill_hover_dark="linear-gradient(90deg, *secondary_500, *secondary_600)",
            slider_color="*secondary_500",
            slider_color_dark="*secondary_600",
            block_title_text_weight="600", block_border_width="3px",
            block_shadow="*shadow_drop_lg", button_primary_shadow="*shadow_drop_lg",
            button_large_padding="11px", color_accent_soft="*primary_100",
            block_label_background_fill="*primary_200",
        )

orange_red_theme = OrangeRedTheme()

# --------------------------- theme ---------------------------

MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1024

dtype = torch.bfloat16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

ADAPTER = {
    "title": "Klein-Consistency",
    "adapter_name": "klein-consistency",
    "repo": "dx8152/Flux2-Klein-9B-Consistency",
    "weights": "Klein-consistency.safetensors",
}


# --- Patch (required for the KV pipeline class) ---
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

# --- Model Loading ---
print("Loading FLUX.2 Klein 9B KV model...")
pipe = Flux2KleinKVPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9b-kv",
    torch_dtype=dtype,
).to(device)
print("Base KV model loaded successfully.")

print(f"Loading adapter: {ADAPTER['title']}")
pipe.load_lora_weights(
    ADAPTER["repo"],
    weight_name=ADAPTER["weights"],
    adapter_name=ADAPTER["adapter_name"],
)
pipe.set_adapters([ADAPTER["adapter_name"]], adapter_weights=[1.0])
print(f"Adapter loaded successfully: {ADAPTER['adapter_name']}")


# --- Utility Functions ---
def calc_dimensions(pil_img: Image.Image) -> Tuple[int, int]:
    """Calculates dimensions preserving aspect ratio, snapped to multiples of 8."""
    iw, ih = pil_img.size
    aspect = iw / ih

    if aspect >= 1:
        new_width = 1024
        new_height = int(round(1024 / aspect))
    else:
        new_height = 1024
        new_width = int(round(1024 * aspect))

    new_width = max(256, min(1024, round(new_width / 8) * 8))
    new_height = max(256, min(1024, round(new_height / 8) * 8))
    return new_width, new_height


def parse_and_resize_images(gallery_items: List, target_width: int, target_height: int) -> List[Image.Image]:
    """Extracts images from a Gradio Gallery and resizes them."""
    if not gallery_items:
        return None

    resized = []
    for item in gallery_items:
        try:
            # Gradio Gallery returns a list of tuples: (filepath, label)
            filepath = item[0] if isinstance(item, (tuple, list)) else item
            img = Image.open(filepath).convert("RGB")
            resized.append(img.resize((target_width, target_height), Image.LANCZOS))
        except Exception as e:
            print(f"Skipping invalid image: {e}")

    return resized if resized else None


# --- Inference Function ---
@spaces.GPU()
def edit_image(
    gallery_inputs,
    prompt: str,
    seed: int,
    randomize_seed: bool,
    width: int,
    height: int,
    steps: int,
    progress=gr.Progress(track_tqdm=True),
):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    current_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)

    image_list = None
    if gallery_inputs and len(gallery_inputs) > 0:
        try:
            # Use the first image to calculate reference dimensions
            first_item = gallery_inputs[0]
            first_filepath = first_item[0] if isinstance(first_item, (tuple, list)) else first_item
            first_pil = Image.open(first_filepath).convert("RGB")

            calc_w, calc_h = calc_dimensions(first_pil)
            image_list = parse_and_resize_images(gallery_inputs, calc_w, calc_h)

            # Override manual width/height to match the input aspect ratio
            width, height = calc_w, calc_h
        except Exception as e:
            print(f"Error processing gallery uploads: {e}")

    final_width = max(256, min(MAX_IMAGE_SIZE, round(int(width) / 8) * 8))
    final_height = max(256, min(MAX_IMAGE_SIZE, round(int(height) / 8) * 8))

    kwargs = dict(
        prompt=prompt,
        height=final_height,
        width=final_width,
        num_inference_steps=int(steps),
    )
    if image_list is not None:
        kwargs["image"] = image_list

    generator = torch.Generator(device="cpu").manual_seed(current_seed)
    result = pipe(**kwargs, generator=generator).images[0]

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, current_seed


with gr.Blocks() as demo:
    gr.Markdown("# **Flux.2 Klein 9B — KV Consistency Edit [Fast]**")
    gr.Markdown(
        "Upload one or more images and enter a prompt to perform fast, consistency-preserving "
        "edits using the `Klein-Consistency` LoRA adapter on the 9B KV model. "
        "Leave the gallery empty to generate from text only. [GitHub ↗](https://github.com/PRITHIVSAKTHIUR/flux-klein-kv-edit-consistency-fast)"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gallery_input = gr.Gallery(
                label="Input Images (Optional)",
                type="filepath",
                height=300,
                allow_preview=True,
                elem_id="gallery_input",
            )
            prompt_input = gr.Textbox(
                label="Edit Prompt",
                placeholder="Describe the edit you want to apply...",
                lines=3,
            )

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    width_slider = gr.Slider(minimum=256, maximum=1024, step=8, value=1024, label="Width")
                    height_slider = gr.Slider(minimum=256, maximum=1024, step=8, value=1024, label="Height")

                steps_slider = gr.Slider(minimum=1, maximum=20, step=1, value=4, label="Inference Steps")

                seed_input = gr.Slider(minimum=0, maximum=MAX_SEED, step=1, value=0, label="Seed")
                randomize_seed_checkbox = gr.Checkbox(label="Randomize Seed", value=True)

            generate_button = gr.Button("Edit Image", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Output", type="pil", interactive=False, height=390, format="png")

    # Wire up the button
    generate_button.click(
        fn=edit_image,
        inputs=[
            gallery_input,
            prompt_input,
            seed_input,
            randomize_seed_checkbox,
            width_slider,
            height_slider,
            steps_slider,
        ],
        outputs=[output_image, seed_input],
    )

    # Examples
    gr.Examples(
        examples=[
            [["examples/1.jpg"], "Change the weather to stormy."],
            [
                ["examples/2.jpg"],
                "Transform the scene into a snowy winter day while preserving the original subject identity, framing, and composition.",
            ],
            [
                ["examples/3.jpg"],
                "Relight the image with soft golden sunset lighting while keeping all structures and subject details consistent.",
            ],
            [["examples/4.jpg"], "Make the texture high-resolution."],
        ],
        inputs=[gallery_input, prompt_input],
        outputs=[output_image, seed_input],
        fn=edit_image,
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.queue().launch(theme=orange_red_theme, ssr_mode=False, mcp_server=True, show_error=True)
