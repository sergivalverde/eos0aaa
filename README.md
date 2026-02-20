# Bioactivity Similarity Index (BSI)

**Ersilia Model ID:** `eos0aaa`

| | |
|---|---|
| **Task** | Pairwise similarity prediction |
| **Input** | Compound pair (two SMILES) |
| **Output** | Score between 0 (dissimilar) and 1 (similar) |
| **Framework** | PyTorch MLP on ECFP4 fingerprints |
| **License** | MIT |

## Description

Predicts whether two compounds share similar bioactivity profiles based on their binding behavior to protein targets. The model uses ECFP4 fingerprints (256-bit Morgan radius 2) summed for compound pairs, fed through a deep neural network (512-256-128-64, ReLU, dropout 0.3) trained on ChEMBL-derived active-active (similar) and active-decoy (dissimilar) pairs with Tanimoto filtering.

The BSI-Large model is a general-purpose variant trained across all protein families, providing broad applicability at the cost of lower absolute scores compared to group-specific models.

## Interpretation

Higher scores indicate that the two compounds are more likely to share bioactivity profiles (i.e., bind similar protein targets). The model was trained with a Tanimoto threshold of 0.40 to avoid trivially similar pairs. Typical output range for BSI-Large is 0.05-0.30; scores above 0.20 suggest meaningful bioactivity overlap.

## Source

- **Publication:** [Schottlender et al., Front. Bioinform. 2025](https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2025.1695353/full)
- **Source Code:** https://github.com/gschottlender/bioactivity-similarity-index

## Deep Validation

| Check | Status | Details |
|-------|--------|---------|
| Distribution (378 pairs) | PASS | 100% valid, mean=0.13, std=0.10, CV=0.74 |
| Sanity (same-target vs random) | PASS | Active pairs 0.18 vs random 0.05 (3.29x) |
| Paper reproduction (Figure 6 compounds) | WARNING | BSI-Large gives lower absolute scores than group-specific models (expected) |
| Wrapper validation (vs source repo) | PASS | 378 pairs match within 5.4e-7 |

**Overall: PASS (9/10, 1 warning)**

### Highlights

- **Distribution analysis:** 378 compound pairs from the paper's test set. Scores range 0.02-0.53 with mean 0.13 and CV 0.74, showing good discrimination.

- **Sanity check:** Compound pairs sharing the same ChEMBL protein target score 3.29x higher than random cross-target pairs, confirming the model captures bioactivity similarity.

- **Paper reproduction:** The paper's Figure 6 uses group-specific BSI models (trained per protein family) which achieve scores 0.77-0.92. BSI-Large, being a general model, gives lower absolute scores but preserves directional correctness — same-target pairs consistently score higher than cross-target pairs.

- **Wrapper validation:** The eos-template wrapper reproduces the original `evaluate_bsi_pairs.py` output exactly (max difference 5.4e-7 across 378 pairs).

See [`test_report.json`](test_report.json) for the full validation report.

## Usage

```python
# Input CSV format (pairwise — two SMILES columns)
# smiles_1,smiles_2
# CCOC(=O)C1=CC=CC=C1,C1=CC=C(C=C1)N
# COC1=CC=CC=C1O,CC(C)OC

python model/framework/code/main.py input.csv output.csv
```

## Dependencies

```yaml
python: "3.10"
commands:
    - ["pip", "numpy", "1.26.4"]
    - ["pip", "pandas", "2.2.3"]
    - ["pip", "torch", "2.5.1"]
    - ["pip", "rdkit", "2022.9.5"]
    - ["pip", "scikit-learn", "1.4.2"]
```
