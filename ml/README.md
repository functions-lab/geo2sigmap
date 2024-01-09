# Geo2SigMap: High-Fidelity RF Signal Mapping Using Geographic Databases
[![arXiv](https://img.shields.io/badge/arXiv-2312.14303-green?color=FF8000?color=009922)](https://arxiv.org/abs/2312.14303)
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-yellow.svg)](https://github.com/dvlab-research/LongLoRA/blob/main/LICENSE)


Welcome to the Geo2SigMap. Here are the documents for our cascade U-Net architecture. 

## TABLE OF CONTENTS
1. [Overview](#overview)
2. [Installation](#installation)
3. [Example Usage](#example-usage)
4. [License](#license)

## Overview

* **We develop a novel cascaded U-Net architecture that achieves significantly improved signal strength (SS) map prediction accuracy compared to existing baseline methods based on channel models and ML.**

![CascadeUNet](https://github.com/functions-lab/geo2sigmap/assets/24806755/f7daa234-5f01-4bf5-895a-f5b10410a806)
   
## Installation

1. Install [CUDA](https://developer.nvidia.com/cuda-downloads)
2. Install [Pytorch](https://pytorch.org/get-started/locally/)
3. Install dependencies
```console
pip install -r requirements.txt
```



## Example Usage

### Download Data

Our dataset is available on the [Google Drive](https://drive.google.com/drive/folders/1x3lM8a2jTl197D0C10eFCBiUgEtFmZwq?usp=sharing).

Unzip the "Oct05_1034_a747d4.zip" file and place it under the "data/synthetic" folder.

For detailed dataset structure or generating your dataset, please refer to the "gen_data" folder.

### Train the First U-Net

```console
python ml/train_RT.py --learning-rate 1e-3 --pathloss_multi_modality --median-filter-size 3 --loss-alpha 0 \
--building-height-map-dir=data/synthetic/Oct05_1034_a747d4/Bl_building_npy \
--ground-truth-dir=data/synthetic/Oct05_1034_a747d4/Oct10_1523_RXcross_TXiso-cross_SampleNum7e6_cmres4/
```

### Train the Second U-Net

```console
python ml/train_transfomer_learning.py --learning-rate 1e-3  --median-filter-size 1 --loss-alpha 0 \
--sparse-point 200 --building-height-map-dir=data/synthetic/Oct05_1034_a747d4/Bl_building_npy \
--ground-truth-dir=data/synthetic/Oct05_1034_a747d4/Oct12_2003_RXcross_TXtr38901-cross_SampleNum7e6_cmres4/ \
--transfer-learning-map-dir=data/synthetic/Oct05_1034_a747d4/prediction_result_121
```

### Prediction


### Pre-trained model

A pre-trained model is available for the above public dataset. Check the [Google Drive](https://drive.google.com/drive/folders/1x3lM8a2jTl197D0C10eFCBiUgEtFmZwq?usp=sharing).



### Visualization

Track your training progress in real-time with [wandb](https://wandb.ai/site). It records loss curves, validation data, weights, gradients, and predicted results. 

Upon starting, a console link leads to your dashboard. Connect your existing W&B account via WANDB_API_KEY or run anonymously. For anonymous users, the logs are automatically deleted after 7 days.



    




## License

Distributed under the APACHE LICENSE, VERSION 2.0
