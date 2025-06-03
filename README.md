# URWKV: Unified RWKV Model with Multi-state Perspective for Low-light Image Restoration

📢 This paper has been accepted to CVPR 2025! 🎉

[main paper](https://pan.baidu.com/s/1CQz1pBSBw-jjY5EK9iVzWg?pwd=xun8) | [supplementary materials](https://pan.baidu.com/s/193YTr_mGHbU0oDIimk8fiQ?pwd=gbj7) | code | results

## Abstract

&#x20;Existing low-light image enhancement (LLIE) and joint LLIE and deblurring (LLIE-deblur) models have made strides in addressing predefined degradations, yet they are often constrained by  dynamically coupled degradations. To address these challenges, we introduce a Unified Receptance Weighted Key Value (URWKV) model with multi-state perspective, enabling flexible and effective degradation restoration for low-light images. Specifically, we customize the core URWKV block to perceive and analyze complex degradations by leveraging multiple intra- and inter-stage states. First, inspired by the pupil mechanism in the human visual system, we propose Luminance-adaptive Normalization (LAN) that adjusts normalization parameters based on rich inter-stage states, allowing for adaptive, scene-aware luminance modulation. Second, we aggregate multiple intra-stage states through exponential moving average approach, effectively capturing subtle variations while mitigating information loss inherent in the single-state mechanism. To reduce the degradation effects commonly associated with conventional skip connections, we propose the State-aware Selective Fusion (SSF) module, which dynamically aligns and integrates multi-state features across encoder stages, selectively fusing contextual information. In comparison to state-of-the-art models, our URWKV model achieves superior performance on various benchmarks,  while requiring significantly fewer parameters and computational resources.

## Code Availability

The code will be released after the official publication of the paper. Stay tuned!
