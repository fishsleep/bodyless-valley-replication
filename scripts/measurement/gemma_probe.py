#!/usr/bin/env python3
"""
Gemma 2 27B VLM probe: forced-choice INTACT/ABNORMAL classification
for the 180 generated images.

Usage:
    python gemma_probe.py                    # Process all 180 images
    python gemma_probe.py --subset 5         # Quick test
"""

import argparse
import os
import torch
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor
import json


def load_gemma(device):
    """Load Gemma 2 27B with quantization for 24GB VRAM."""
    model_id = "google/gemma-2-27b-it"

    print("Loading Gemma 2 27B...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,  # 4-bit quantization for 24GB VRAM
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def build_prompt():
    """Create forced-choice prompt for anatomical integrity."""
    return """<start_of_turn>user
Look at this image and determine if the human body/anatomy appears INTACT (anatomically normal, correct number of fingers, normal proportions) or ABNORMAL (extra/missing fingers, fused limbs, distorted anatomy, melted features).

Answer with exactly one word: INTACT or ABNORMAL

Image: <image><end_of_turn>
<start_of_turn>model
"""


def find_image(data_dir, lora, weight, seed):
    """Find image path."""
    lora_dir = "bodyhorror" if lora == "bodyhorror" else "glitchingbodypart"
    weight_dir = f"{lora_dir}_{weight}"
    fname = f"{lora_dir}_{weight}_seed{seed}.png"
    return Path(data_dir) / "raw_images" / weight_dir / fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output', default='data/scores/gemma_full_180.csv')
    parser.add_argument('--subset', type=int, help='Process only N images per LoRA/weight')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("WARNING: Running on CPU will be very slow for 27B model")

    model, processor = load_gemma(device)
    prompt = build_prompt()

    lorals = ['bodyhorror', 'glitchingbodypart']
    weights = ['w000', 'w060', 'w120']
    seeds = list(range(1001, 1031))

    results = []

    for lora in lorals:
        for weight in weights:
            seed_list = seeds[:args.subset] if args.subset else seeds

            for seed in tqdm(seed_list, desc=f"{lora} {weight}"):
                lora_dir = "bodyhorror" if lora == "bodyhorror" else "glitchingbodypart"
                weight_dir = f"{lora_dir}_{weight}"
                fname = f"{lora_dir}_{weight}_seed{seed}.png"
                img_path = Path(args.data_dir) / "raw_images" / weight_dir / fname

                if not img_path.exists():
                    print(f"Missing: {img_path}")
                    continue

                image = Image.open(img_path).convert('RGB')

                # Build input
                inputs = processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                ).to(model.device)

                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )

                # Decode
                response = processor.decode(outputs[0], skip_special_tokens=True)
                response = response.split("model")[-1].strip().upper()

                verdict = "INTACT" if "INTACT" in response else "ABNORMAL" if "ABNORMAL" in response else "UNCLEAR"

                results.append({
                    'lora': lora,
                    'weight': weight,
                    'seed': seed,
                    'verdict': verdict,
                    'raw_response': response,
                })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == '__main__':
    main()