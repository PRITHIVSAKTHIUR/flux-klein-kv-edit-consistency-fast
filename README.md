# **[Flux.2 Klein 9B — KV Consistency Edit [Fast]](https://huggingface.co/spaces/prithivMLmods/flux-klein-kv-edit-consistency-fast)**

Flux.2 Klein 9B — KV Consistency Edit [Fast] is an optimized, image-to-image editing suite designed around the advanced `black-forest-labs/FLUX.2-klein-9b-kv` base model. By incorporating a dedicated pipeline patch (`flux2_klein_kv.patch`) alongside the `dx8152/Flux2-Klein-9B-Consistency` LoRA adapter, this suite yields identity-preserving, context-consistent adjustments over input images in a fraction of standard rendering windows.

Operating entirely on custom CUDA setups, the suite offers text-guided image manipulation—such as seasonal swaps, complex structural relighting, and high-fidelity texture enhancement—with zero reliance on external APIs.

<img width="1748" height="1561" alt="screencapture-huggingface-co-spaces-prithivMLmods-flux-klein-kv-edit-consistency-fast-2026-07-07-10_27_45" src="https://github.com/user-attachments/assets/f59aa706-28f1-4b35-b08e-3c0028530b30" />

### **Key Features**

* **KV Attention Modification:** Integrates a core structural patch directly over local `diffusers` modules via subprocess initialization to enable Key-Value consistency mechanisms.
* **Klein-Consistency LoRA Engine:** Leverages the `Flux2-Klein-9B-Consistency` adapter at a unified weight scale ($1.0$), ensuring structural traits and composition stay fixed during inference.
* **Intelligent Resolution Adaptation:** Parses active Gradio Gallery inputs and downscales or upscales dimensions to match native training boundaries, snapping layouts to multiples of 8.
* **Unified ZeroGPU Workflow:** Features memory cleanup utilities (`gc.collect()` and `torch.cuda.empty_cache()`) coupled with the `@spaces.GPU` context executor to provide multi-step editing without encountering out-of-memory overheads.

### **Repository Structure**

```text
├── examples/
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── 3.jpg
│   └── 4.jpg
├── app.py
├── flux2_klein_kv.patch
├── LICENSE.txt
├── pre-requirements.txt
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock

```

### **Installation and Requirements**

To run Flux.2 Klein 9B — KV Consistency Edit locally, ensure you possess an appropriate Python configuration compiled with heavy weight execution backends. A modern CUDA-enabled GPU is required.

This repository specifically relies on **PyTorch 2.11.0 and CUDA 13.0** (`--extra-index-url https://download.pytorch.org/whl/cu130`).

#### **Running with `uv` (Recommended)**

`uv` is an ultra-fast Python package and project manager written in Rust, ensuring rapid virtual environment synchronization and reproducible execution.

**Step 1 — Install `uv`**

* **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`
* **Windows:** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`

**Step 2 — Clone the repository**

```bash
git clone https://github.com/PRITHIVSAKTHIUR/flux-klein-kv-edit-consistency-fast.git
cd flux-klein-kv-edit-consistency-fast

```

**Step 3 — Initialize the project and install dependencies**
This will automatically parse the `uv.lock` and `requirements.txt` to fetch the correct PyTorch 2.11.0 + cu130 wheels.

```bash
uv sync

```

**Step 4 — Run the script**

```bash
uv run app.py

```

#### **Standard PIP Installation**

**1. Install Pre-requirements**
Ensure your local system package manager is upgraded:

```bash
pip install pip>=26.0.0

```

**2. Install Core Dependencies**
Install the primary deep learning stack, diffusion utilities, and ecosystem structures. Place these in a `requirements.txt` file and execute `pip install -r requirements.txt`.

```text
--extra-index-url https://download.pytorch.org/whl/cu130

git+https://github.com/huggingface/transformers.git@v4.57.6
huggingface-hub
gradio==6.16.0
torch==2.11.0
opencv-python
sentencepiece
torchvision
torchaudio
accelerate
omegaconf
termcolor
diffusers
kernels
imageio
hf_xet
spaces
pyyaml
pillow
numpy
peft
ftfy
av
```

### **Usage**

After setting up your environment and ensuring your dependencies are installed, launch the application by executing the primary module script:

```bash
python app.py

```

The script will trigger a safety lookup for your environment, parse `flux2_klein_kv.patch`, and hot-patch your local site-packages `diffusers` library. It will then download and cache the 9B model weights along with the consistency LoRA layers. Once initialized, a local web server interface will be exposed (typically at `http://127.0.0.1:7860/`).

1. **Input Asset:** Upload one or more reference images to the Gradio input gallery. Leaving the panel completely clear switches the underlying generation path back into text-only mode.
2. **Define Modification:** Enter your specific adjustments in the text box (e.g., *"Transform the scene into a snowy winter day"*).
3. **Advanced Tweaks:** Expand the Advanced Settings menu to scale inference steps, lock baseline seeds, or manually enforce structural dimension ratios.
4. **Compile:** Click **Edit Image** to launch the CUDA workspace worker thread and review the consistency-matched result.

### **License and Source**

* **License:** [Apache License 2.0](https://github.com/PRITHIVSAKTHIUR/flux-klein-kv-edit-consistency-fast/blob/main/LICENSE.txt)
* **GitHub Repository:** [https://github.com/PRITHIVSAKTHIUR/flux-klein-kv-edit-consistency-fast.git](https://github.com/PRITHIVSAKTHIUR/flux-klein-kv-edit-consistency-fast.git)
