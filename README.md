
# LMASD-G — Looking at Multiple Attributes for a Single Decision: Gender Recognition at Long Range

**LMASD-G** is a multimodal architecture (CLIP + text→vision cross-attention) for **face-independent gender recognition** at **long distances** (≈10–120 m) and challenging viewpoints (30°, 60°, 90°). The model combines ***soft biometrics*** (beard, moustache, hairstyle, upper, lower, feet, accessories) with a pure visual pathway, fused via cross-attention. We evaluate on the unified **U-DetAGReID** benchmark (DetReIDx + AG-ReID.v2), reporting distance/height-stratified analyses and attention maps for interpretability.

> **Core idea:** “**multiple details** for **a single decision**” — many weak cues, aligned through language–vision fusion, become robust and auditable when the face is not informative.

## Table of Contents

* [Repository Layout](#repository-layout)
* [Key Components](#key-components)
* [Requirements](#requirements)
* [Data & Vocabulary](#data--vocabulary)
* [Datasets & Label Harmonization](#datasets--label-harmonization)
* [Training (example)](#training-example)
* [Evaluation](#evaluation)
* [Checkpoint Inspection](#checkpoint-inspection)
* [Results (summary)](#results-summary)
* [Citation](#citation)
* [License](#license)

---

## Repository Layout

```
.
├── main.py                  # training / orchestration (model + pipeline)
├── test.py                  # evaluation and metrics/plots
├── datasets.py              # Dataset + DataLoader (attributes + CLIP normalization)
├── clip_VITb.py             # CLIP/ViT utility blocks (if applicable)
├── model_factory.py         # (optional) factory for external scripts
├── run_test.sh              # example evaluation runner
├── ver_configuration.py     # checkpoint inspection (LR, dropout, unfrozen layers, etc.)
└── README.md
```

## Key Components

* **Two paths:** (1) visual→gender and (2) text+visual→attributes (+aux gender)
* **SCA (Spatial–Channel Attention):** refines patch tokens in both paths
* **Text→vision attention per attribute:** prompts focus relevant regions
* **Final fusion (CrossAttention):** merges gender logits from both paths
* **Attribute heads:** one head per selected attribute
* **“Unknown” class:** optional for ambiguous cases (Male/Female/Unknown)

## Requirements

* Python 3.8+ (3.8–3.10 recommended)
* PyTorch ≥ 2.0
* `transformers`, `timm` (if used), `scikit-learn`, `pandas`, `numpy`, `matplotlib`

Example (virtualenv):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers timm scikit-learn pandas numpy matplotlib
```

## Data & Vocabulary

* **Attribute vocabulary:** `Expanded_CLIP_Descriptions_CoreAttributes.csv`
  Columns: `Attribute, code, Label, Expanded Description`

* **Seven attributes (SOTA config):**

```python
ATTRS7 = ['beard','moustache','hairstyle','upper','lower','feet','accessories']
```

**Layout example**

```
data/
  train/  val/  test/
  arquivo_descrpts/Expanded_CLIP_Descriptions_CoreAttributes.csv
  *.csv   # annotations with image_name + attributes + gender
```

> Reproduce best checkpoint with the **same attribute order** and **same vocab CSV**.

---

## Datasets & Label Harmonization

We unified the ontologies of **DetReIDx** and **AG-ReID.v2** into a single corpus (**U-DetAGReID**), standardizing class indices per attribute (including **Unknown** where applicable). Below we list **only attributes that required changes**, mapping **original → unified**.

> **Column legend:** **Act_Val** = original value; **Udp_Val** = final unified value.

### Age — mapping (DetReIDx ↔ AG-ReID.v2)

| Age bin | DetReIDx Act_Val | DetReIDx Udp_Val | AG-ReID.v2 Act_Val | AG-ReID.v2 Udp_Val |
| ------- | ---------------: | ---------------: | -----------------: | -----------------: |
| 00–11   |                0 |                — |                  — |                  — |
| 12–17   |                1 |                — |                  — |                  — |
| 18–24   |                2 |            **0** |                  0 |                  0 |
| 25–34   |                3 |            **0** |                  0 |                  0 |
| 35–44   |                4 |            **1** |                  1 |                  1 |
| 45–54   |                5 |            **1** |                  1 |                  1 |
| 55–64   |                6 |            **2** |                  2 |                  2 |
| >65     |                7 |            **2** |                  2 |                  2 |
| Unknown |               −1 |            **3** |                  3 |                  3 |

**Description.** Compact DetReIDx bins to match AG-ReID.v2: (18–34)→0, (35–54)→1, (55+)→2; *Unknown*→3. Bins 0–11/12–17 have no counterpart.

---

### Ethnicity — mapping (DetReIDx ↔ AG-ReID.v2)

| Ethnicity | DetReIDx Act_Val | DetReIDx Udp_Val | AG-ReID.v2 Act_Val | AG-ReID.v2 Udp_Val |
| --------- | ---------------: | ---------------: | -----------------: | -----------------: |
| India     |                0 |            **0** |                  3 |              **0** |
| Black     |                1 |            **1** |                  1 |              **1** |
| Turkish   |                2 |            **2** |                  — |                  — |
| White     |                3 |            **3** |                  0 |              **3** |
| Asian     |                — |                — |                  2 |              **4** |
| Unknown   |                — |                — |                  4 |              **5** |

**Description.** Unified to 6 classes: {0:India, 1:Black, 2:Turkish, 3:White, 4:Asian, 5:Unknown}. DetReIDx lacks *Asian/Unknown*; AG-ReID.v2 lacks *Turkish*.

---

### Glasses — mapping (DetReIDx ↔ AG-ReID.v2)

| Class          | DetReIDx Act_Val | DetReIDx Udp_Val | AG-ReID.v2 Act_Val | AG-ReID.v2 Udp_Val |
| -------------- | ---------------: | ---------------: | -----------------: | -----------------: |
| Normal_glasses |                0 |                0 |                  0 |                  0 |
| Sunglasses     |                1 |                1 |                  1 |                  1 |
| Unknown        |               −1 |            **2** |                  3 |              **2** |
| No             |                — |                — |                  2 |                  2 |

**Description.** *Normal_glasses* and *Sunglasses* preserved. *Unknown* normalized to 2. DetReIDx has no explicit *No* class.

---

### Upper (torso) — mapping (DetReIDx ↔ AG-ReID.v2)

| Class    | DetReIDx Act_Val | DetReIDx Udp_Val | AG-ReID.v2 Act_Val | AG-ReID.v2 Udp_Val |
| -------- | ---------------: | ---------------: | -----------------: | -----------------: |
| T-shirt  |                0 |                0 |                  0 |                  0 |
| Blouse   |                1 |                1 |                  1 |                  1 |
| Sweater  |                2 |                2 |                  2 |                  2 |
| Coat     |                3 |                3 |                  3 |                  3 |
| Bikini   |                — |                — |                  4 |                  4 |
| Naked    |                — |                — |                  5 |                  5 |
| Dress    |                4 |            **6** |                  6 |                  6 |
| Uniform  |                5 |            **7** |                  7 |                  7 |
| Shirt    |                6 |            **8** |                  8 |                  8 |
| Suit     |                7 |            **9** |                  9 |                  9 |
| Hoodie   |                8 |           **10** |                 10 |                 10 |
| Cardigan |                — |           **11** |                 11 |                 11 |
| Unknown  |               −1 |           **12** |                 12 |                 12 |

**Description.** Re-indexed to match AG-ReID.v2; added unified *Cardigan=11*. *Unknown=12*.

---

### Lower (legs) — mapping (DetReIDx ↔ AG-ReID.v2)

| Class   | DetReIDx Act_Val | DetReIDx Udp_Val | AG-ReID.v2 Act_Val | AG-ReID.v2 Udp_Val |
| ------- | ---------------: | ---------------: | -----------------: | -----------------: |
| Jeans   |                0 |                0 |                  0 |                  0 |
| Leggins |                1 |                1 |                  1 |                  1 |
| Pants   |                2 |                2 |                  2 |                  2 |
| Shorts  |                3 |                3 |                  3 |                  3 |
| Skirt   |                4 |                4 |                  4 |                  4 |
| Bikini  |                — |            **5** |                  5 |                  5 |
| Dress   |                5 |            **6** |                  6 |                  6 |
| Uniform |                6 |            **7** |                  7 |                  7 |
| Suit    |                7 |            **8** |                  8 |                  8 |
| Unknown |                9 |                9 |                  9 |                  9 |

**Description.** Added *Bikini* and aligned *Dress/Uniform/Suit*. *Unknown=9*.

---

### Accessories / Bag — mapping (DetReIDx ↔ AG-ReID.v2)

| Class       | DetReIDx Act_Val | DetReIDx Udp_Val | AG-ReID.v2 Act_Val | AG-ReID.v2 Udp_Val |
| ----------- | ---------------: | ---------------: | -----------------: | -----------------: |
| Bag         |                0 |                0 |                  0 |                  0 |
| Backpack    |                1 |                1 |                  1 |                  1 |
| Handbag     |                — |                — |                  2 |                  2 |
| Rolling_bag |                2 |            **3** |                  3 |                  3 |
| Umbrella    |                3 |            **4** |                  4 |                  4 |
| Sportif_bag |                4 |            **5** |                  5 |                  5 |
| Market_bag  |                5 |            **6** |                  6 |                  6 |
| Nothing     |                6 |            **7** |                  7 |                  7 |
| Unknown     |                8 |                8 |                  8 |                  8 |

**Description.** Re-indexed several classes to match AG-ReID.v2; *Handbag* absent in DetReIDx.

---

### Prompt Vocabulary (summary)

* Text→vision queries come from a CSV with expanded per-attribute, per-class descriptions (e.g., *feet*: “a person wearing high heels”).
* For each sample, active descriptions are concatenated into a short sentence and tokenized by **CLIP**’s text encoder.
* Missing/undefined labels use neutral prompts (e.g., “the attribute is unclear”) or are omitted to avoid bias.

---

### Splits & Preprocessing

* Keep official splits; ensure **zero identity leakage** across train/val/test.
* Images resized to **224×224** and normalized with CLIP stats; training uses random horizontal flip; validation uses resize+normalization only.

**U-DetAGReID split counts:**

| Split     |    Male |  Female |       Total |
| --------- | ------: | ------: | ----------: |
| Train     | 177,173 | 131,419 |     308,592 |
| Val       | 110,555 |  84,128 |     194,683 |
| Test      |  67,591 |  67,361 |     134,952 |
| **Total** | 355,319 | 282,908 | **638,227** |

> Global total = 308,592 + 194,683 + 134,952 = **638,227**.

---

## Training (example)

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --seed 42 \
  --dataset-root ./data \
  --train-csv ./data/train_final.csv \
  --val-csv   ./data/val_final.csv \
  --vocab-csv ./data/arquivo_descrpts/Expanded_CLIP_Descriptions_CoreAttributes.csv \
  --selected-attrs beard moustache hairstyle upper lower feet accessories \
  --clip-model-id openai/clip-vit-base-patch16 \
  --input-size 336 \
  --ft-visual-last 4 \
  --ft-text-last 0 \
  --use-sca-path1 1 --use-sca-path2 1 \
  --lr-clip 1e-6 --lr-heads 1e-4 \
  --dropout 0.4 \
  --epochs 32 \
  --outdir ./.checkpoints/detagreid_2
```

## Evaluation

```bash
/bin/bash run_test.sh
# or
python test.py \
  --checkpoint /path/to/checkpoints/epoch_1.pth.tar \
  --vocab_csv ./data/arquivo_descrpts/Expanded_CLIP_Descriptions_CoreAttributes.csv \
  --split val \
  --out_dir ./eval_outputs/lmasdg_val \
  --weights-only
```

**Metrics:** Acc, mA (balanced accuracy), F1 (macro), Precision (macro), Recall (weighted), AUC (Female vs. Rest).
**Outputs:** cumulative-like curves by distance/height bins; optional correct/incorrect galleries.

## Checkpoint Inspection

```bash
python ver_configuration.py \
  --checkpoint /path/to/epoch_1.pth.tar \
  --list-ckpt-keys \
  --weights-only

python ver_configuration.py \
  --checkpoint /path/to/epoch_1.pth.tar \
  --model-factory model_factory:build_model \
  --factory-kwargs '{
    "vocab_csv_path":"./data/arquivo_descrpts/Expanded_CLIP_Descriptions_CoreAttributes.csv",
    "selected_attrs":["beard","moustache","hairstyle","upper","lower","feet","accessories"],
    "clip_model_id":"openai/clip-vit-base-patch16",
    "input_size":336,
    "ft_visual_last":4,
    "ft_text_last":0,
    "use_sca_path1":true,
    "use_sca_path2":true,
    "sca_reduction":16,
    "heads_from_fused_attr":true,
    "num_heads":8
  }' \
  --weights-only \
  --export-csv ./model_params.csv
```

Expected: optimizer groups (if saved), attribute head sizes, dropout `p`, #unfrozen CLIP layers, and a `model_params.csv` inventory.

## Results (summary)

* **ViT-B/16.v2 (5 attrs):** mA **75.72%**, F1-macro **75.77%**, AUC **84.37%**
* **ViT-B/16.v4 (7 attrs):** mA **85.61%**, F1-macro **85.60%**, AUC **94.11%**
* Stable up to ~80 m in oblique views (30°/60°); sharper degradation at 90° above 80–100 m.
* Attention maps: **upper/lower/hairstyle** (short/medium); **feet/accessories** (long); **beard/moustache** (short).

## Citation

```bibtex
@article{lmasdg2025,
  title   = {Looking at Multiple Attributes for a Single Decision: Gender Recognition at Extreme and Long Distances},
  author  = {Mbongo Nzakiese,  Kailash A. Hambardea, Hugo Proença},
  journal = {arXiv},
  year    = {2025},
  archivePrefix={arXiv},
  url={...}, 
}
```
