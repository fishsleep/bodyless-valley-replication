# Replication Guide

This guide walks through reproducing the "Bodyless Valley" study from scratch.

---

## Prerequisites

- **GPU**: AMD RDNA4 (R9700) 32GB VRAM or NVIDIA 24GB+ VRAM
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **Python**: 3.10+
- **R**: 4.3+ with lme4, lmerTest, emmeans, performance
- **Disk**: ~2 GB free space

---

## Step 1: Environment Setup

```bash
# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate ometita-replication

# Option B: pip (if conda not available)
pip install -r requirements.txt
```

**ROCm for AMD GPUs:**
```bash
# Ensure ROCm is installed
export HSA_OVERRIDE_GFX_VERSION=12.0.1  # RDNA4 (gfx1201)
# Verify
rocminfo | grep -i gfx
```

---

## Step 2: Download Models

### Base Checkpoint (SD 1.5)
```bash
# Option 1: Hugging Face Hub (recommended)
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir models/sd15

# Option 2: Single-file for ComfyUI
huggingface-cli download Comfy-Org/stable-diffusion-v1-5-archive --local-dir models/sd15
```

### LoRA Files
```bash
# Update models/LORA_*_URL.txt with your HF repos first!
huggingface-cli download fishsleep/BodyHorror-LoRA-r128 --local-dir data/loras
huggingface-cli download fishsleep/GlitchingBodyPart-LoRA-r8 --local-dir data/loras
```

### Measurement Models
```bash
# DINO (facebook/dino-vits16) - auto-downloaded by timm
# CLIP (openai/clip-vit-large-patch14) - auto-downloaded by open_clip
# Gemma 2 27B (google/gemma-2-27b-it) - auto-downloaded by transformers
```

---

## Step 3: Generate Images

```bash
# Full 180-image run (30 seeds × 2 LoRAs × 3 weights)
python scripts/generation/generate_all.py

# Or quick test (5 images per condition)
python scripts/generation/generate_all.py --subset 2
```

**Expected output:** 180 PNG files in `data/raw_images/` organized by LoRA/weight.

---

## Step 4: Compute Scores

```bash
# DINO + CLIP-I structural/semantic similarity
python scripts/measurement/compute_dino_clip.py

# Gemma VLM probe (requires 24GB+ VRAM for 4-bit)
python scripts/measurement/gemma_probe.py

# Quick test (5 images per condition)
python scripts/measurement/compute_dino_clip.py --subset 2
python scripts/measurement/gemma_probe.py --subset 2
```

**Output:** `data/scores/full_run_scores.csv`, `data/scores/gemma_full_180.csv`

---

## Step 5: Statistical Analysis

```bash
# Mixed-effects models + Holm-Bonferroni correction
Rscript scripts/analysis/analyze_mixed_effects.R

# VLM analysis
python scripts/analysis/gemma_analyze.py
```

**Output:** `data/scores/analysis_results.rds`, `data/scores/gemma_summary.json`

---

## Step 6: Build Figures

```bash
python scripts/analysis/build_figures.py
```

**Output:** 4 JPG figures in `manuscript/figures/`

---

## Step 6: Build Manuscript

```bash
# Build DOCX from markdown
python scripts/docx/build_docx.py

# Convert to IJHI-compliant .doc (Office 97-2003)
libreoffice --headless --convert-to doc:MS\ Word\ 97 manuscript/manuscript_final_v2.docx
```

**Output:** `manuscript/manuscript_final_v2.doc` (IJHI submission format)

---

## Step 7: Compliance Check

```bash
python scripts/analysis/compliance_check.py
```

---

## Expected Outputs

| File | Location | Description |
|------|----------|-------------|
| Manuscript (IJHI) | `manuscript/manuscript_final_v2.doc` | Submission-ready .doc |
| Figures (4) | `manuscript/figures/*.jpg` | Publication-ready JPGs |
| Scores | `data/scores/full_run_scores.csv` | DINO/CLIP-I for 180 images |
| VLM results | `data/scores/gemma_full_180.csv` | Bodyless probe verdicts |
| Analysis | `data/scores/analysis_results.rds` | R mixed-effects results |
| Summary | `data/scores/gemma_summary.json` | VLM summary statistics |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Use `--subset` for testing; enable 4-bit quantization |
| ROCm not found | Install ROCm 6.0+; set `HSA_OVERRIDE_GFX_VERSION=12.0.1` |
| Gemma OOM | Use 4-bit quantization (`load_in_4bit=True`) |
| R packages missing | Install via conda: `conda install -c conda-forge r-lme4 r-lmerTest r-emmeans` |
| LoRA not found | Update `models/LORA_*_URL.txt` and re-download |

---

## Expected Results (from paper)

| Metric | BodyHorror | GlitchingBodyPart |
|--------|------------|-------------------|
| DINO (w=1.2) | ~0.70 | ~0.70 |
| CLIP-I (w=1.2) | ~0.81 | ~0.81 |
| False-INTACT (w=1.2) | 63.3% | 80.0% |
| False-ABNORMAL (w=0.0) | ~10% | ~10% |

---

## Citation

```bibtex
@article{anonymous2026bodyless,
  title={The Bodyless Valley: Seven Fingers, a Threshold, and the Trace of the Model's Own Hand},
  author={{Author, Anonymous}},
  journal={International Journal on Humanistic Ideology},
  year={2026}
  note={Special Issue: Bodyless Hallucination}
}
```