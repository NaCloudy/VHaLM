# VHaLM: From Visual Moments to Haiku

**Visual-Haiku Language Model for Multi-Image Poetic Generation**

> UW-Madison STAT 453 (Deep Learning), Fall 2025 — Group 3

Given three semantically related images, VHaLM generates a structurally valid **5-7-5 English haiku** that captures their shared visual essence.

---

## Pipeline

```
[Image Triplet] → Stage 1: Multi-Image Captioner → [Unified Description] → Stage 2: Haiku Generator → [Haiku]
```

### Stage 1 — Multi-Image Captioner

- Encodes each image with a **frozen CLIP ViT-L/14** encoder (768-dim)
- Fuses three image embeddings via **multi-token attention with image position embeddings** and semantic tag augmentation
- Decodes a unified description using **LoRA-adapted T5-base**

### Stage 2 — Haiku Generator (training-free)

- Prompt-based generation via LLM
- **Deterministic syllable checker** enforces 5-7-5 structure
- **LLM-as-judge reranking** selects the best candidate

---

## Dataset

2,099 semantically coherent image triplets constructed from MS COCO (~9,000 images):

1. Extract CLIP ViT-B/32 embeddings for all images
2. Build a *k*-NN graph with a dual constraint (cosine similarity + neighborhood overlap)
3. Mine triangle cliques for high-consistency triplets
4. Synthesize pseudo-reference captions via GPT-4o-mini fusion

---

## Results

### Stage 1 — Multi-Image Captioning

| Model | BLEU-1 | BLEU-4 | METEOR |
|-------|--------|--------|--------|
| T5-small baseline | 0.302 | 0.050 | 0.229 |
| VHaLM Stage 1 (ours) | **0.332** | **0.065** | **0.260** |

### Stage 2 — Haiku Generation

LLM-as-judge scoring (GPT-4o-mini, scale 1–5): **avg. relevance 4.3 / 5**, **100% structure compliance** (≥4/5) on test set.

Example outputs:

```
Colors dance on wind        [group 191 — beach/ocean/sky]
Laughter echoes with the waves
Sky and sea embrace

White ball in the air,      [group 1468 — tennis match]
Racket arcs through summer light,
Joy leaps with each swing.

Waves kiss golden sand      [group 1413 — beach/frisbee]
A blue shirt dances with joy
Frisbee soaring high
```

---

## Repository Structure

```
├── data/
│   ├── image_ids.json                      # Image metadata
│   ├── image_groups_with_captions.json     # 2,099 triplets with captions
│   ├── image_groups_curriculum.json        # Curriculum-stratified triplets
│   ├── merged_captions.json                # LLM-synthesized pseudo-labels
│   └── test/                              # Test set annotations
│
├── src/
│   ├── data_prep/                         # Data pipeline scripts
│   │   ├── download_coco_subset.py
│   │   ├── clip_extract_ordered.py
│   │   ├── group_images_kmeans.py
│   │   ├── data_processing.py
│   │   └── data_processing_augmented.py
│   ├── model1/                            # Stage 1: multi-image captioner
│   │   ├── lib/                           # Model, dataset, eval, logger
│   │   ├── scripts/                       # Train, inference, evaluate
│   │   ├── config.json
│   │   └── requirements.txt
│   └── model2/                            # Stage 2: haiku generator
│       ├── generate.py
│       ├── judge.py
│       ├── structure.py
│       ├── syllables.py
│       ├── prompts.py
│       └── requirements.txt
│
├── experiment/
│   ├── model1/
│   │   ├── v0_multi/                      # Baseline: global attention
│   │   ├── v1_multi/                      # Multi-token fusion
│   │   ├── v1_single/                     # Single-token fusion
│   │   ├── v1_zero/                       # Zero-shot
│   │   └── final/                         # Final test results
│   ├── model2/
│   │   ├── final/                         # Final haiku outputs
│   │   └── test_results/                  # Test set haiku outputs
│   └── metrics_comparison.csv
│
├── models/                                # Model weights (download separately)
└── reports/                               # Course reports (proposal / midway / final)
```

---

## Setup

```bash
# Stage 1
pip install -r src/model1/requirements.txt

# Stage 2
pip install -r src/model2/requirements.txt
```

Model weights for CLIP, T5, and BLIP are downloaded automatically at runtime via HuggingFace.

---

## Related Work

- CLIP: [Radford et al., 2021](https://arxiv.org/abs/2103.00020)
- T5: [Raffel et al., 2020](https://arxiv.org/abs/1910.10683)
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
