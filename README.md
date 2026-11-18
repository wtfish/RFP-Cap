# RFP-Cap: Residual Feature Projection with Spatial Attention for Image Captioning

<p align="center">
  <strong>Accepted at the 2027 IEEE Symposium Series on Computational Intelligence (IEEE SSCI 2027)</strong><br>
  <em>RFP-Cap: Residual Feature Projection with Spatial Attention for Image Captioning</em><br><br>
  Christian Arthur · Isanka Dissanayake · Raymond Halim · Farshid Hajati · Raymond Chiong
</p>
<p align="center">
  <a href="https://attend.ieee.org/ssci-2027/"><img src="https://img.shields.io/badge/IEEE%20SSCI%202027-Accepted-brightgreen" alt="Accepted at IEEE SSCI 2027"></a>
  <img src="https://img.shields.io/badge/Method-Residual%20Projection%20%7C%20Spatial%20Attention-purple" alt="Method">
  <img src="https://img.shields.io/badge/Task-Assistive%20Image%20Captioning-green" alt="Task">
  <img src="https://img.shields.io/badge/Validation-VizWiz-blue" alt="Validation">
</p>

> **Publication status:** Accepted at [IEEE SSCI 2027](https://attend.ieee.org/ssci-2027/), Gold Coast, Australia, 14–17 February 2027. Proceedings details and a DOI will be added when available.

---
## Overview

This work presents **RFP-Cap**, an image-captioning architecture built around a residual feature-projection module placed before spatial attention. The method is designed to improve how high-dimensional visual features are transferred from the encoder to the language decoder.

The architecture combines:
- a spatial ResNet-50 encoder;
- a dual-path residual feature-projection module;
- additive spatial attention;
- an LSTMCell decoder;
- autoregressive caption generation;
- beam-search inference.

Rather than passing the original ResNet region features directly into the attention-decoder stage, RFP-Cap first adapts them into a compact representation while preserving complementary encoder information through a learned residual path.

---
## Key Contributions
1. A **dual-path residual feature-projection module** that adapts high-dimensional spatial visual features before caption generation.
2. A **spatial attention mechanism** that dynamically selects caption-relevant image regions at each decoding step.
3. A comparative evaluation across multiple convolutional, recurrent, attention-based, and transformer-based captioning architectures on VizWiz.
4. An analysis of caption quality, qualitative failure modes, and spatial attention behaviour under challenging assistive-use imagery.

---
## RFP-Cap Architecture

RFP-Cap preserves the spatial structure of the ResNet-50 feature map rather than collapsing the image into a single global vector.

The encoder produces:

$$
F \in \mathbb{R}^{B \times 2048 \times 7 \times 7}
$$

which is reshaped into 49 spatial region descriptors:

$$
v_i \in \mathbb{R}^{2048}, \qquad i=1,\ldots,49
$$

Each region is then processed through two parallel feature-adaptation paths.

### Nonlinear Projection Path

$$
p_i =
\mathrm{Dropout}
\left(
\mathrm{ReLU}
\left(
\mathrm{LN}(W_p v_i + b_p)
\right)
\right)
$$

### Residual Path

$$
r_i = W_s v_i
$$

### Residual Fusion

$$
z_i = p_i + r_i
$$

The resulting adapted representation is:

$$
z_i \in \mathbb{R}^{512}
$$

and the spatial tensor passed to the attention-decoder stage has size:

$$
B \times 49 \times 512
$$

---
## Spatial Attention and Decoding

RFP-Cap applies additive attention over the adapted region features.

The alignment score for region $i$ at decoding step $t$ is:

$$
e_{t,i}
=
w_a^T
\tanh
\left(
W_z z_i + W_h h_{t-1} + b_a
\right)
$$

The normalized attention weights are:

$$
\alpha_{t,i}
=
\frac{\exp(e_{t,i})}
{\sum_j \exp(e_{t,j})}
$$

and the context vector is:

$$
c_t
=
\sum_i \alpha_{t,i} z_i
$$

The context vector is combined with the previous-token embedding and passed to an LSTMCell decoder. The updated hidden state is projected to vocabulary logits and normalized using softmax.

During optimization, the ground-truth token from the previous decoding step is provided as input. During inference, captions are generated using beam search with width five.

---
## Computational Motivation

RFP-Cap reduces the dimensionality of each ResNet spatial region from:

$$
d_v = 2048
$$

to:

$$
d_p = 512
$$

so that:

$$
d_p = \frac{d_v}{4}
$$

The residual projection is computed once for the spatial regions, while the compact representation is reused throughout caption generation.

For $N$ spatial regions, the projection has complexity:

$$
O(Nd_vd_p)
$$

whereas the repeated feature-dependent computation in the attention-decoder interface changes from:

$$
O(TNd_v) + O(THd_v)
$$

to:

$$
O(TNd_p) + O(THd_p)
$$

where $T$ is the generated sequence length and $H$ is the decoder hidden dimension.

The proposed design therefore moves repeated visual processing from the original 2,048-dimensional encoder space into a compact 512-dimensional representation.

---
## Experimental Setting

Experiments are conducted on **VizWiz-Captions**, which contains images captured by blind or low-vision users under real assistive-use conditions.

The dataset includes challenging visual characteristics such as:

- blur and poor illumination;
- partial or unconventional framing;
- occlusion;
- product packaging and visible text;
- rare or unfamiliar objects;
- visually ambiguous scenes.

The evaluated RFP-Cap configuration uses:

| Component | Setting |
|---|---|
| Encoder | ImageNet-pretrained ResNet-50 |
| Fine-tuning | ResNet layer 4 trainable |
| Spatial representation | 7 × 7 grid |
| Encoder region dimension | 2,048 |
| Projected region dimension | 512 |
| Main projection | Linear → LayerNorm → ReLU → Dropout |
| Residual branch | Bias-free Linear 2048 → 512 |
| Attention | Bahdanau additive attention |
| Decoder | Single LSTMCell |
| Hidden dimension | 512 |
| Optimizer | Adam |
| Decoder learning rate | $4 \times 10^{-4}$ |
| Encoder learning rate | $1 \times 10^{-4}$ |
| Regularization | Label smoothing 0.1; gradient clipping 5.0 |
| Training | 15 epochs |
| Inference | Beam search, width 5 |

---
## Performance

RFP-Cap is evaluated against representative encoder-decoder configurations using BLEU-1 through BLEU-4, METEOR, and the CIDEr approximation retained from the original experiment.

| Model | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | CIDEr* |
|---|---:|---:|---:|---:|---:|---:|
| EffNet-LSTM | 0.6140 | 0.4200 | 0.2840 | 0.1910 | — | — |
| ViT-LSTM | 0.6193 | 0.4173 | 0.2762 | 0.1856 | — | — |
| ViT-Attn-LSTM | **0.6284** | **0.4342** | 0.2923 | 0.1962 | — | — |
| ResNet-LSTM | 0.6195 | 0.4309 | 0.2903 | 0.1944 | 0.3473 | 0.0302 |
| **RFP-Cap** | 0.6239 | 0.4334 | **0.2934** | **0.1971** | **0.3552** | 0.0280 |

\* The reported CIDEr value follows the approximation used in the original experiment and should not be interpreted as standard COCO CIDEr.

RFP-Cap achieves the strongest **BLEU-3**, **BLEU-4**, and available **METEOR** results among the evaluated architectures. The comparison with ResNet-LSTM is particularly informative because both models already preserve spatial ResNet features and use attention-based recurrent decoding. The additional gain therefore indicates that feature adaptation before attention provides a measurable benefit beyond the spatial encoding and attention mechanisms already present.

---
## Qualitative Findings

Qualitative evaluation shows that RFP-Cap performs well when the dominant object or interaction is visually salient.

Representative behaviours include:

- category recognition under substantial blur;
- person-object interaction recognition;
- correct localization of visually dominant products;
- spatial attention concentrated on foreground regions.

The remaining errors are concentrated around:
- text-heavy packaging;
- fine-grained product identity;
- rare or out-of-vocabulary concepts;
- ambiguous local visual structures;
- broader contextual interpretation.

Attention visualizations indicate that the decoder can focus on the correct foreground region even when the generated caption remains less specific than the reference description. This suggests that localization and fine-grained lexical identification remain distinct challenges in assistive image captioning.

---
## Main Findings
- RFP-Cap achieves the highest BLEU-3 and BLEU-4 scores among the evaluated architectures.
- RFP-Cap records the highest available METEOR score.
- The proposed module improves BLEU-1 through BLEU-4 and METEOR over the closest ResNet-LSTM configuration.
- The strongest higher-order captioning performance is achieved without replacing ResNet-50 with a larger transformer-based visual backbone.
- The results support the encoder-attention interface as an important design point for image-captioning architectures.
- Residual feature adaptation improves the representation used by spatial attention while reducing the visual dimensionality processed downstream.
- Fine-grained product recognition, embedded text, lexical coverage, and ambiguous scene interpretation remain open challenges.

---
## Authors

- Christian Arthur
- Isanka Dissanayake
- Raymond Halim
- Farshid Hajati
- Raymond Chiong

---
## Code Availability

This repository accompanies the paper **accepted at IEEE SSCI 2027**. At present, the public repository contains the methodological overview, experimental configuration, and reported results. The implementation, training scripts, evaluation code, and pretrained checkpoints have **not yet been uploaded** to this repository. Availability information will be updated when these materials are released.

---
## Citation

If you use this work, please cite the accepted paper as follows. The final IEEE Xplore citation, DOI, and page numbers will be added after publication.

```bibtex
@inproceedings{arthur2027rfpcap,
  title     = {RFP-Cap: Residual Feature Projection with Spatial Attention for Image Captioning},
  author    = {Arthur, Christian and Dissanayake, Isanka and Halim, Raymond and Hajati, Farshid and Chiong, Raymond},
  booktitle = {2027 IEEE Symposium Series on Computational Intelligence (SSCI)},
  year      = {2027},
  note      = {Accepted for presentation}
}
```
