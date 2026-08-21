# Bodyless Valley — Replication Package

**Paper**: "The Bodyless Valley: Seven Fingers, a Threshold, and the Trace of the Model's Own Hand"  
**Venue**: International Journal on Humanistic Ideology, Special Issue "Bodyless Hallucination"  
**Author**: Anonymized (withheld for peer review)  
**DOI**: [pending]

---

## Quick Start

```bash
# 1. Clone repo
git clone https://github.com/fishsleep/bodyless-valley-replication.git
cd bodyless-valley-replication

# 2. Install environment (conda recommended)
conda env create -f environment.yml
conda activate ometita-replication

# 3. Download LoRAs from Hugging Face
# (Update models/LORA_*_URL.txt with your HF repos first)
huggingface-cli download fishsleep/BodyHorror-LoRA-r128 --local-dir data/loras
huggingface-cli download fishsleep/GlitchingBodyPart-LoRA-r8 --local-dir data/loras

# 4. Regenerate scores from images
python scripts/measurement/compute_dino_clip.py
python scripts/measurement/gemma_probe.py

# 5. Run statistical analysis
Rscript scripts/analysis/analyze_mixed_effects.R

# 6. Build manuscript (optional)
python scripts/docx/build_docx.py
```

---

## Exact Models Used (Hugging Face Links)

| Component | HF Repository | Purpose |
|-----------|---------------|---------|
| **Base SD 1.5** | [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5) | Generation backbone |
| **DINO (ViT-S/16)** | [`facebook/dino-vits16`](https://huggingface.co/facebook/dino-vits16) | Structural similarity |
| **CLIP-I (ViT-L/14)** | [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) | Semantic fidelity |
| **Gemma VLM** | [`google/gemma-2-27b-it`](https://huggingface.co/google/gemma-2-27b-it) | Bodyless probe |
| **BodyHorror LoRA** | [`fishsleep/BodyHorror-LoRA-r128`](https://huggingface.co/fishsleep/BodyHorror-LoRA-r128) | Rank 128, photoreal |
| **GlitchingBodyPart LoRA** | [`fishsleep/GlitchingBodyPart-LoRA-r8`](https://huggingface.co/fishsleep/GlitchingBodyPart-LoRA-r8) | Rank 8, anime |

> **Update the LoRA URLs** in `models/LORA_*_URL.txt` with your actual Hugging Face repos.

---

## Experimental Design

| Parameter | Value |
|-----------|-------|
| Base model | `runwayml/stable-diffusion-v1-5` (v1-5-pruned-emaonly-fp16) |
| LoRAs | BodyHorror (rank 128, photoreal) + GlitchingBodyPart (rank 8, anime) |
| Weights | 0.0 / 0.6 / 1.2 (strength_model; strength_clip=0) |
| Seeds | 30 (1001–1030), pinned |
| Prompts | 30 (hand close-ups, full-body figures, portraits) |
| Total images | 180 (30 × 2 LoRAs × 3 weights) |
| Sampler | dpmpp_2m / karras, 25 steps, cfg 7.5, 512×512 |

---

## Measurement Stack

```python
# DINO: facebook/dino-vits16
# CLIP-I: openai/clip-vit-large-patch14  
# Gemma: google/gemma-2-27b-it (forced-choice INTACT/ABNORMAL)
```

---

## Statistical Analysis

```r
# R 4.3+ with lme4, lmerTest, emmeans
# Model: drift ~ weight + weight^2 + content_type * LoRA + (1|image)
# Holm-Bonferroni correction across 12 primary tests
```

---

## License

- **Code/Scripts**: MIT License
- **Paper/Figures/Data**: CC-BY-4.0 (attribution required)
- **LoRA Models**: CC-BY-4.0 (hosted on Hugging Face)

---

## Citation

```bibtex
@article{anonymous2026bodyless,
  title={The Bodyless Valley: Seven Fingers, a Threshold, and the Trace of the Model's Own Hand},
  author={{Author, Anonymous}},
  journal={International Journal on Humanistic Ideology},
  year={2026},
  note={Special Issue: Bodyless Hallucination}
}
```