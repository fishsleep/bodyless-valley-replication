#!/usr/bin/env python3
"""
Compute DINO (ViT-S/16) structural similarity and CLIP-I (ViT-L/14) semantic fidelity
between degraded images and their same-seed baselines.

Usage:
    python compute_dino_clip.py                    # Process all 180 images
    python compute_dino_clip.py --subset 5         # Quick test on 5 images
    python compute_dino_clip.py --output scores.csv
"""

import argparse
import os
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm
import timm
import open_clip


def load_models(device):
    """Load DINO and CLIP models."""
    print("Loading DINO (ViT-S/16)...")
    dino = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
    dino.eval().to(device)
    dino_transform = timm.data.create_transform(
        input_size=224,
        interpolation='bicubic',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    print("Loading CLIP (ViT-L/14)...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai', device=device
    )
    clip_model.eval()

    return dino, dino_transform, clip_model, clip_preprocess


@torch.no_grad()
def compute_dino_similarity(model, transform, img1, img2, device):
    """Compute DINO structural similarity between two images."""
    # Resize to 224x224 for ViT-S/16
    t1 = transform(img1.resize((224, 224))).unsqueeze(0).to(device)
    t2 = transform(img2.resize((224, 224))).unsqueeze(0).to(device)

    feat1 = model(t1)
    feat2 = model(t2)

    # Normalize
    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1)

    # Cosine similarity
    sim = (feat1 * feat2).sum(dim=-1).item()
    return sim


@torch.no_grad()
def compute_clip_similarity(model, preprocess, img1, img2, device):
    """Compute CLIP-I semantic fidelity between two images."""
    t1 = preprocess(img1).unsqueeze(0).to(device)
    t2 = preprocess(img2).unsqueeze(0).to(device)

    feat1 = model.encode_image(t1)
    feat2 = model.encode_image(t2)

    feat1 = F.normalize(feat1, dim=-1)
    feat2 = F.normalize(feat2, dim=-1)

    sim = (feat1 * feat2).sum(dim=-1).item()
    return sim


def find_image_pair(base_dir, lora, weight, seed):
    """Find degraded and baseline image paths."""
    lora_dir = "bodyhorror" if lora == "bodyhorror" else "glitchingbodypart"
    weight_dir = f"{lora_dir}_{weight}"

    # Degraded image
    degraded_name = f"{lora_dir}_{weight}_seed{seed}.png"
    degraded_path = Path(base_dir) / "raw_images" / weight_dir / degraded_name

    # Baseline (w000)
    baseline_name = f"{lora_dir}_w000_seed{seed}.png"
    baseline_path = Path(base_dir) / "raw_images" / f"{lora_dir}_w000" / baseline_name

    return degraded_path, baseline_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--output', default='data/scores/full_run_scores.csv')
    parser.add_argument('--subset', type=int, help='Process only N images per LoRA/weight')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dino, dino_transform, clip, clip_preprocess = load_models(device)

    # Experimental design
    lorals = ['bodyhorror', 'glitchingbodypart']
    weights = ['w000', 'w060', 'w120']
    seeds = list(range(1001, 1031))

    results = []

    for lora in lorals:
        for weight in weights:
            if weight == 'w000':
                continue  # Skip baseline (similarity = 1.0 by definition)

            seed_list = seeds[:args.subset] if args.subset else seeds

            for seed in tqdm(seed_list, desc=f"{lora} {weight}"):
                degraded_path, baseline_path = find_image_pair(args.data_dir, lora, weight, seed)

                if not degraded_path.exists() or not baseline_path.exists():
                    print(f"Missing: {degraded_path} or {baseline_path}")
                    continue

                img_degraded = Image.open(degraded_path).convert('RGB')
                img_baseline = Image.open(baseline_path).convert('RGB')

                dino_sim = compute_dino_similarity(dino, dino_transform, img_degraded, img_baseline, device)
                clip_sim = compute_clip_similarity(clip, clip_preprocess, img_degraded, img_baseline, device)

                results.append({
                    'lora': lora,
                    'weight': weight,
                    'seed': seed,
                    'dino_similarity': dino_sim,
                    'clip_similarity': clip_sim,
                })

    # Add baseline rows (similarity = 1.0)
    for lora in lorals:
        for seed in seeds:
            results.append({
                'lora': lora,
                'weight': 'w000',
                'seed': seed,
                'dino_similarity': 1.0,
                'clip_similarity': 1.0,
            })

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == '__main__':
    main()