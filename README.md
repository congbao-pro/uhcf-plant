<div align="center"> <h1>A Unified Hybrid Classification Framework with Organ-Aware V-MoE for Fine-Grained Plant Identification</h1> </div>

## 📚 Overview

Accurate fine-grained plant identification remains a critical challenge in computer vision due to severe inter-species morphological convergence and complex structural variations. To address these inherent limitations, this study first introduces **BigPlants-100**, a rigorously curated dataset constructed to serve as a high-quality benchmark for fine-grained taxonomy. Based on this foundation, **we propose a novel Organ-Aware Vision Mixture of Experts (Organ-Aware V-MoE)** to explicitly capture localized structural priors. This module is subsequently integrated into a **comprehensive Unified Hybrid Classification Framework**, which effectively fuses it with global semantic representations to significantly enhance the ability to distinguish highly similar botanical classes. Comprehensive evaluations demonstrate the superior discriminative capacity of the proposed architecture. The complete hybrid framework achieves an outstanding **accuracy of 0.896** on the standard test set and maintains a robust **accuracy of 0.838** on a strictly unseen unselected data pool. Notably, the unified approach yields an impressive absolute accuracy improvement over the standalone Organ-Aware V-MoE, achieving a **7.7% increase** on the test set and a **13.1% increase** under unseen data conditions. Furthermore, the model ensures highly stable computational dynamics, recording a rapid average inference time of **21.47 ms** with a minimal standard deviation of **1.78 ms**. These empirical results confirm that the proposed framework provides a highly accurate, generalized, and computationally efficient solution for real-world automated plant taxonomy.

## 🌳 BigPlants-100 Dataset

- The entire raw dataset is available at the following link:
  ```
  https://drive.google.com/drive/folders/1zbczeI8HnfzKhMAybibRq9a40Jcm7bX_?usp=sharing
  ```
- The full dataset is available at the following link:
  ```
  https://drive.google.com/drive/folders/1uEFtoS-XivF030a5BAbM8mD341eqd_I9?usp=sharing
  ```

## 🍃 Leaf Disease Segmentation Dataset

The dataset is available in Kaggle:
  ```
  https://www.kaggle.com/datasets/fakhrealam9537/leaf-disease-segmentation-dataset
  ```