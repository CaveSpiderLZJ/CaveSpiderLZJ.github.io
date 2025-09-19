---
layout: publication
title: "[ECCV'24] PoseAugment: Generative Human Pose Data Augmentation with Physical Plausibility for IMU-based Motion Capture"
subtitle: '<b>Zhuojun Li</b>, Chun Yu*, Chen Liang, Yuanchun Shi'
author: "CaveSpider"
header-style: text
---

![Teaser](/img/2024-11-23/poseaugment_teaser.jpg)

## Paper Abstract

The data scarcity problem is a crucial factor that hampers
the model performance of IMU-based human motion capture.
However, effective data augmentation for IMU-based motion capture is challenging,
since it has to capture the physical relations and constraints of the human body,
while maintaining the data distribution and quality.
We propose PoseAugment, a novel pipeline incorporating VAE-based
pose generation and physical optimization.
Given a pose sequence, the VAE module generates infinite poses with both high fidelity
and diversity, while keeping the data distribution.
The physical module optimizes poses to satisfy physical constraints
with minimal motion restrictions.
High-quality IMU data are then synthesized from the augmented poses
for training motion capture models.
Experiments show that PoseAugment outperforms previous data augmentation
and pose generation methods in terms of motion capture accuracy,
revealing a strong potential of our method to alleviate the data collection
burden for IMU-based motion capture and related tasks driven by human poses.

## External Links
[[Paper]](/paper/PoseAugment.pdf)
[[DOI]](https://doi.org/10.1007/978-3-031-73411-3_4)
