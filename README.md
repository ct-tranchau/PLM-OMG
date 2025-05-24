# PLM-OMG: Protein Language Model-Based Ortholog Detection for Cross-Species Cell Type Mapping

This repository contains the code, data references, and results associated with the paper:

> **Title:** PLM-OMG: Protein Language Model-Based Ortholog Detection for Cross-Species Cell Type Mapping  
> **Authors:** Tran N. Chau, Song Li  
> **DOI/Preprint:** [...]

---

## Overview

**PLM-OMG** benchmarks five deep learning models for orthogroup classification and cross-species cell type mapping in plants:

- **ESM2**: Encoder-only transformer for structural and functional modeling  
- **ProGen2**: Decoder-only transformer for functional protein generation  
- **ProteinBERT**: Bidirectional transformer pre-trained with GO annotations  
- **ProtGPT2**: Decoder-only transformer designed for de novo protein design  
- **LSTM**: RNN baseline trained from scratch for protein sequence classification

We evaluate these models on three datasets:
- A curated multi-species dataset from 15 diverse plant species  
- Monocot orthogroups from PLAZA 5.0 (https://bioinformatics.psb.ugent.be/plaza/versions/plaza_v5_monocots) 
- Dicot orthogroups from PLAZA 5.0 (https://bioinformatics.psb.ugent.be/plaza.dev/_dev_instances/feedback/)

---

## Key Results

| Model        | Accuracy | Precision |  Recall | F1 Score |
|--------------|----------|-----------|---------|----------|
| ProtGPT2     | 0.95     | 0.95      | 0.95    | 0.94     |
| ESM2         | 0.86     | 0.88      | 0.88    | 0.86     |
| ProGen2      | 0.85     | 0.89      | 0.85    | 0.85     |
| ProteinBERT  | 0.36     | 0.41      | 0.32    | 0.30     |
| LSTM         | 0.78     | 0.82      | 0.80    | 0.79     |

---

## Installation

```bash
git clone https://github.com/ct-tranchau/PLM-OMG.git
cd PLM-OMG
pip install -r requirements.txt
```
