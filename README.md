# URWKV: Unified RWKV Model with Multi-state Perspective for Low-light Image Restoration

📢 This paper has been accepted to CVPR 2025! 🎉

[Main Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xu_URWKV_Unified_RWKV_Model_with_Multi-state_Perspective_for_Low-light_Image_CVPR_2025_paper.pdf) | [Supplementary Materials](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Xu_URWKV_Unified_RWKV_CVPR_2025_supplemental.pdf) | [Poster](https://pan.baidu.com/s/18Z84hr2_HlXGzy1XXcZMIw?pwd=56u9)

## Weights and Results

All released resources are collected in the following Baidu Pan archives:

* **Results:** [Download](https://pan.baidu.com/s/1EiuCvuj_Ycw0YEDpzhFLJg?pwd=kn23) (extraction code: `kn23`). In addition, results of the comparison method [BiFormer](https://github.com/FZU-N/BiFormer) are available on [Baidu Pan](https://pan.baidu.com/s/12g91-HIxCdq36DV-Pt8wvQ?pwd=ssfh) (extraction code: `ssfh`).

* **Pre-trained weights:** [Download](https://pan.baidu.com/s/1UuKmG6WcaCWdwkj3_jsPPg?pwd=5ady) (extraction code: `5ady`)

* **Datasets:** [Download](https://pan.baidu.com/s/1R0L4QEXw0uOyWyVp1x6Zig?pwd=2x5i) (extraction code: `2x5i`)

All metrics below are reported in the sRGB space using `measure_pair.py`, without GT Mean-related post-processing.

| Metric      | LOL-v1 | LOL-v2-real | LOL-v2-sync |  SID  |  SMID | SDSD-indoor | SDSD-outdoor | MIT-5K | LOL-blur |
| :---------- | :----: | :---------: | :---------: | :---: | :---: | :---------: | :----------: | :----: | :------: |
| PSNR (dB) ↑ |  24.27 |    23.11    |    26.36    | 23.11 | 29.44 |    31.24    |     29.99    |  26.08 |   27.27  |
| SSIM ↑      |  0.856 |    0.874    |    0.944    | 0.673 | 0.826 |    0.911    |     0.887    |  0.936 |   0.890  |

> ⚠️ **Missing weights:** The pretrained weights for **SMID** and **MIT-Adobe FiveK** were accidentally overwritten and are currently unavailable. You can retrain these models using the provided configurations. We will update this section if the weights are recovered or retrained.

## Abstract

&#x20;Existing low-light image enhancement (LLIE) and joint LLIE and deblurring (LLIE-deblur) models have made strides in addressing predefined degradations, yet they are often constrained by  dynamically coupled degradations. To address these challenges, we introduce a Unified Receptance Weighted Key Value (URWKV) model with multi-state perspective, enabling flexible and effective degradation restoration for low-light images. Specifically, we customize the core URWKV block to perceive and analyze complex degradations by leveraging multiple intra- and inter-stage states. First, inspired by the pupil mechanism in the human visual system, we propose Luminance-adaptive Normalization (LAN) that adjusts normalization parameters based on rich inter-stage states, allowing for adaptive, scene-aware luminance modulation. Second, we aggregate multiple intra-stage states through exponential moving average approach, effectively capturing subtle variations while mitigating information loss inherent in the single-state mechanism. To reduce the degradation effects commonly associated with conventional skip connections, we propose the State-aware Selective Fusion (SSF) module, which dynamically aligns and integrates multi-state features across encoder stages, selectively fusing contextual information. In comparison to state-of-the-art models, our URWKV model achieves superior performance on various benchmarks,  while requiring significantly fewer parameters and computational resources.

## Overview

![](README_md_files/6cf966f0-5190-11f0-847b-8bd8db6e5334.jpeg?v=1&type=image)

## Main Results

Consistent with [BiFormer](https://github.com/FZU-N/BiFormer), results are measured using `measure_pair.py`. It should be noted that all metrics in our method are computed in the sRGB space, and no GT Mean-related techniques are applied.

![](README_md_files/9e45a430-5190-11f0-847b-8bd8db6e5334.jpeg?v=1&type=image)

![](README_md_files/e4f9c500-5190-11f0-847b-8bd8db6e5334.jpeg?v=1&type=image)

To ensure fairness, if a comparison method does not provide pretrained weights, we retrain it using the recommended settings provided by the authors. Otherwise, we use the officially released pretrained weights for evaluation. All results are evaluated using a unified script, `measure_pair.py`. In this paper, the following methods were retrained: SNR-Net, FourLLIE, UHDFour, LLFormer, Retinexformer, BiFormer, RetinexMamba, LEDNet, PDHAT, MIRNet, Restormer, and MambaIR.

## Environment Setup

### 1. Create and Activate a Conda Environment

```markup
conda create --name URWKV python=3.9
conda activate URWKV 
```

### 2. Install PyTorch and Dependencies

Install PyTorch with CUDA 11.3 support:

```markup
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
```

Install cuDNN:

```markup
conda install cudnn
```

### 3. Install Python Dependencies

```markup
pip install pyyaml yacs tqdm colorama pandas natsort  
pip install matplotlib tensorboardX 
pip install cython thop prefetch_generator 
pip install opencv-python scikit-image  
pip install timm einops mmcls 
pip install pytorch_msssim IQA_pytorch pyiqa lpips
pip install numpy==1.26.4
```

### 4. Install MMCV

```markup
pip install -U openmim
mim install mmcv==1.7.1
```

> ⚠️ Note: If you encounter an error related to Ninja while compiling C++ extensions (e.g., Ninja is required to load C++ extensions), install Ninja with:  `sudo apt-get install ninja-build`

### 🔧 Training Notes

* **Training stability:** Instability observed in some training runs may be related to the CUDA environment or version compatibility. The cause has not yet been confirmed, and this issue remains under investigation.

* **Hyperparameter tuning:** Since we haven't done extensive hyperparameter tuning, you are encouraged to explore better configurations to potentially improve performance.

## Citation

If you find this work useful for your research, please cite:

```markup
@inproceedings{xu2025urwkv,
  title={URWKV: Unified RWKV Model with Multi-state Perspective for Low-light Image Restoration},
  author={Xu, Rui and Niu, Yuzhen and Li, Yuezhou and Xu, Huangbiao and Liu, Wenxi and Chen, Yuzhong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={21267--21276},
  year={2025}
}
```

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/FZU-N/URWKV/blob/main/LICENSE) file.

## Acknowledgement

URWKV is built with reference to the code of the following projects: [RWKV](https://github.com/BlinkDL/RWKV-LM), [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV), and [BiFormer](https://github.com/FZU-N/BiFormer). Thanks for their awesome work!
