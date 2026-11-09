This repository contains the code and data for the paper "Modality-Aware Fine-Grained Feature Fusion Architecture for Lung Cancer Subtype Classification using PET/CT Imaging". The code will be made public once the article is accepted！
<img width="693" height="468" alt="image" src="https://github.com/user-attachments/assets/02ebe507-674d-4231-9342-14b1fa48ffe9" />

We propose MAFF-Net for lung cancer subtype classification based on multimodal PET/CT images. The task involves distinguishing between highly similar lesion types under conditions of limited lesion volume and high background interference. To address these challenges, MAFF-Net is designed to extract fine-grained complementary representations from different imaging modalities and integrate them in a robust, confidence-adaptive manner.
![Uploading image.png…]()
# Summary
The main contributions are summarized as follows:
(1) We propose MAFF-Net, a novel network for lung cancer subtype classification based on multimodal PET/CT images.
(2) We develop the Res-PEA module to enhance lesion localization, and introduce the DDCB and class-consistency loss in the CT branch to boost fine-grained representation learning.
(3) We design the STLM for robust PET feature representation and employ prototype-supervised learning in the PET-CT fusion branch, achieving superior feature discrimination and generalization.
(4) We present a dynamic cascading voting mechanism to reliably integrate multi-branch predictions, significantly improving the overall classification accuracy.
# Dataset
The recruited dataset:
# Requirements
The experiments were conducted using Python version 3.8 (Python Software Foundation, Wilmington, DE, USA), PyTorch 2.0.0, with Compute Unified Device Architecture (CUDA) version 11.8, and an NVIDIA GeForce RTX 4090 GPU, running on Ubuntu 20.04 (Canonical, London, UK).
