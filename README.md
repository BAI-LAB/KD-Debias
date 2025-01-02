# Invariant Debiasing Learning for Recommendation via Biased Imputation

This repository contains the code for the paper "Invariant Debiasing Learning for Recommendation via Biased Imputation," which has been accepted by Information Processing and Management(IPM).

## Environment Requirements

- Python Version: 3.8.0

## Installation Steps

### STEP 1: Prepare Datasets

1. **Download Datasets**:
   - *Yahoo!R3*: https://webscope.sandbox.yahoo.com/catalog.php?datatype=r&did=3
   
   - *COAT*: https://www.cs.cornell.edu/schnabts/mnar/
   
   - *MIND*: https://paperswithcode.com/dataset/mind

2. **Configure Datasets Paths**:
   - Configure local datasets paths in `global_config.py`.

### STEP 2: Run the Models

1. **Configure local paths for results saving in `global_config.py`**
2. **Run `main_xxx.py` to train models on datastes**

## Citation

If you use this project in your research, please cite the following paper:

```
@article{BAI2025104028,
title = {Invariant debiasing learning for recommendation via biased imputation},
journal = {Information Processing & Management},
volume = {62},
number = {3},
pages = {104028},
year = {2025},
issn = {0306-4573},
doi = {https://doi.org/10.1016/j.ipm.2024.104028},
url = {https://www.sciencedirect.com/science/article/pii/S030645732400387X},
author = {Ting Bai and Weijie Chen and Cheng Yang and Chuan Shi},
} 
```
