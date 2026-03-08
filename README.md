# VHaLM: From Visual Moments to Haiku

> Given three semantically related images, generate a structurally valid **5-7-5 English haiku** that captures their shared visual essence.

---

## Examples

**Three images in. One haiku out.**

| <img src="assets/examples/907_1.jpg" width="210"> | <img src="assets/examples/907_2.jpg" width="210"> | <img src="assets/examples/907_3.jpg" width="210"> |
|:---:|:---:|:---:|

> *Salt spray in the air, / He dances on azure backs, / Ocean's pulse beneath.*

&nbsp;

| <img src="assets/examples/191_1.jpg" width="210"> | <img src="assets/examples/191_2.jpg" width="210"> | <img src="assets/examples/191_3.jpg" width="210"> |
|:---:|:---:|:---:|

> *Colors dance on wind, / Laughter echoes with the waves, / Sky and sea embrace.*

---

## Results

**Stage 1 — Multi-Image Captioning** (test set, n=316)

| Model | BLEU-1 | BLEU-4 | METEOR |
|:------|:------:|:------:|:------:|
| T5-small baseline | 0.302 | 0.050 | 0.229 |
| **VHaLM Stage 1** | **0.332** | **0.065** | **0.260** |

**Stage 2 — Haiku Generation** (GPT-4o-mini judge, scale 1–5)

| Metric | Score |
|:-------|:-----:|
| Relevance | 4.3 / 5 |
| Structure compliance | 5.0 / 5 |
| Overall | 18.4 / 25 |

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  Input: Image Triplet                                           │
│                                                                 │
│  [img₁]  [img₂]  [img₃]                                        │
│     │       │       │                                           │
│     └───────┴───────┘                                           │
│             │                                                   │
│    Stage 1: Multi-Image Captioner                               │
│    CLIP ViT-B/32 → Multi-Token Attention + LoRA T5-base         │
│             │                                                   │
│    "A man skillfully rides a wave on a surfboard in the ocean." │
│             │                                                   │
│    Stage 2: Haiku Generator  (training-free)                    │
│    LLM prompt → Syllable checker → LLM-as-judge reranking       │
│             │                                                   │
│  Output: "Salt spray in the air, / He dances on azure backs,   │
│           / Ocean's pulse beneath."                             │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 1 — Multi-Image Captioner

- Frozen **CLIP ViT-B/32** encodes each image into a 512-dim embedding
- **Multi-token attention** with image position embeddings fuses the three embeddings into 12 visual tokens, preserving which image each token came from
- **LoRA-adapted T5-base** decodes a unified natural language description

### Stage 2 — Haiku Generator

- Prompt-based generation with an LLM conditioned on the Stage 1 description
- **Deterministic syllable checker** enforces exact 5-7-5 structure
- **LLM-as-judge reranking** scores candidates on relevance, imagery, and fluency

---

## Dataset

2,099 semantically coherent image triplets built from MS COCO (~9,000 images):

1. Extract CLIP ViT-B/32 embeddings
2. Build a *k*-NN graph with dual constraint (cosine similarity + neighborhood overlap)
3. Mine triangle cliques for high-consistency triplets
4. Synthesize pseudo-reference captions via GPT-4o-mini fusion

---

## Repository Structure

```
├── data/
│   ├── image_ids.json                      # COCO image metadata
│   ├── image_groups_with_captions.json     # 2,099 triplets with captions
│   ├── image_groups_curriculum.json        # Curriculum-stratified triplets
│   ├── merged_captions.json                # LLM-synthesized pseudo-labels
│   └── test/                              # Test set annotations
│
├── src/
│   ├── data_prep/                         # Dataset construction pipeline
│   ├── model1/                            # Stage 1: multi-image captioner
│   │   ├── lib/                           # Model, dataset, eval, logger
│   │   ├── scripts/                       # Train, inference, evaluate
│   │   └── config.json                    # Training configuration
│   └── model2/                            # Stage 2: haiku generator
│       ├── generate.py
│       ├── judge.py
│       ├── syllables.py
│       └── prompts.py
│
├── experiment/                            # Ablation results and metrics
└── assets/examples/                       # Example image triplets
```

---

## Setup

```bash
# Stage 1
pip install -r src/model1/requirements.txt

# Stage 2
pip install -r src/model2/requirements.txt
```

CLIP, T5, and BLIP weights are downloaded automatically via HuggingFace at runtime.

---

## References

- CLIP — [Radford et al., 2021](https://arxiv.org/abs/2103.00020)
- T5 — [Raffel et al., 2020](https://arxiv.org/abs/1910.10683)
- LoRA — [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
