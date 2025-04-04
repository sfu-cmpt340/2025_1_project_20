<!-- # Interactive Generation of DermSynth3D Data
This repository is a template for your CMPT 340 course project.
Replace the title with your project title, and **add a snappy acronym that people remember (mnemonic)**.

Add a 1-2 line summary of your project here.

## Acronym: DermSynth3D

Summary: DermSynth3D synthesizes realistic skin lesion images to address the lack of available data for deep learning in medicine, overcoming privacy and ethical concerns.

## What was the motive behind this topic?
The motive behind DermSynth3D is to overcome the challenges of data scarcity in dermatology for deep learning. It addresses issues such as privacy concerns, limited labeled data, and class imbalances by generating realistic synthetic skin lesion images. This synthetic data helps to augment real datasets, enabling more robust AI models for skin disease detection while adhering to privacy regulations and ethical standards.

[DermSynth3D github link](https://github.com/sfu-mial/DermSynth3D)

## Important Links

| [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EQVnMvkVw5NBqdqjeR0Sy2sBDikpXcyxfIWPbuRUXovYVg?e=af9bcT) | [Slack channel](https://cmpt340spring2025.slack.com/archives/C0877AZ4ASW) | [Project report](https://www.overleaf.com/4416194535yqcgjwkxtbny#ada67e) |
|-----------|---------------|-------------------------|


## Video/demo/GIF
Record a short video (1:40 - 2 minutes maximum) or gif or a simple screen recording or even using PowerPoint with audio or with text, showcasing your work.


## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

4. [Guidance](#guide)


<a name="demo"></a>
## 1. Example demo

A minimal example to showcase your work

```python
from amazing import amazingexample
imgs = amazingexample.demo()
for img in imgs:
    view(img)
```

### What to find where

Explain briefly what files are found where

```bash
repository
├── src                          ## source code of the package itself
├── scripts                      ## scripts, if needed
├── docs                         ## If needed, documentation   
├── README.md                    ## You are here
├── requirements.yml             ## If you use conda
```

<a name="installation"></a>

## 2. Installation

Provide sufficient instructions to reproduce and install your project. 
Provide _exact_ versions, test on CSIL or reference workstations.

```bash
git clone https://github.com/sfu-cmpt340/2025_1_project_20.git
cd https://github.com/sfu-cmpt340/2025_1_project_20.git
conda env create -f requirements.yml
conda activate amazing
```

<a name="repro"></a>
## 3. Reproduction
Demonstrate how your work can be reproduced, e.g. the results in your report.
```bash
mkdir tmp && cd tmp
wget https://yourstorageisourbusiness.com/dataset.zip
unzip dataset.zip
conda activate amazing
python evaluate.py --epochs=10 --data=/in/put/dir
```
Data can be found at ...
Output will be saved in ...

<a name="guide"></a>
## 4. Guidance

- Use [git](https://git-scm.com/book/en/v2)
    - Do NOT use history re-editing (rebase)
    - Commit messages should be informative:
        - No: 'this should fix it', 'bump' commit messages
        - Yes: 'Resolve invalid API call in updating X'
    - Do NOT include IDE folders (.idea), or hidden files. Update your .gitignore where needed.
    - Do NOT use the repository to upload data
- Use [VSCode](https://code.visualstudio.com/) or a similarly powerful IDE
- Use [Copilot for free](https://dev.to/twizelissa/how-to-enable-github-copilot-for-free-as-student-4kal)
- Sign up for [GitHub Education](https://education.github.com/)  -->

# Interactive Generation of DermSynth3D Data

This repository contains the complete pipeline for generating, training, and evaluating synthetic dermatology images using deep learning and statistical methods. Our project, **DermSynth3D**, addresses the challenges of limited dermatology datasets by focusing on class balancing through synthetic image generation.

---

## 🧠 Acronym: DermSynth3D
**Dermatology Synthetic 3D** – A pipeline to synthesize realistic skin lesion images to enhance machine learning performance for underrepresented lesion classes.

---

## 📌 Project Summary
DermSynth3D synthesizes realistic dermatology images for weak classes and evaluates their utility through classification and quality metrics. This project combines synthetic image generation, classical ML training, PCA visualization, and FID-based evaluation for a complete pipeline in dermatological AI.

---

## 💡 Motivation
Medical image datasets often suffer from:
- **Privacy constraints** making real data hard to collect
- **Limited data** for rare skin conditions
- **Severe class imbalance** in disease types

This project uses synthetic generation to:
- Expand weak class samples
- Maintain realistic features
- Improve classifier generalization
- Enable experimentation without data privacy concerns

---

## 🔗 Important Links

| Resource         | Link                                                                 |
|------------------|----------------------------------------------------------------------|
| Timesheet        | [Timesheet](https://1sfu-my.sharepoint.com/:x:/g/personal/hamarneh_sfu_ca/EQVnMvkVw5NBqdqjeR0Sy2sBDikpXcyxfIWPbuRUXovYVg?e=af9bcT) |
| Slack Channel    | [Slack](https://cmpt340spring2025.slack.com/archives/C0877AZ4ASW)    |
| Project Report   | [Overleaf Report](https://www.overleaf.com/4416194535yqcgjwkxtbny#ada67e) |

---

## 📽️ Demo
A walkthrough demo video of the full DermSynth3D pipeline including synthetic generation, training, and evaluation (shared separately with submission).

[Video Demo](https://youtu.be/tvr8eKKy2vw)

---

## 📁 Directory Structure
```
2025_1_PROJECT_20/
├── augmented_dataset/                 # Holds post-processed augmented images
├── augmented_images/                 # Holds image augmentations applied
├── data/blending/data/DermSynth3D/  # Blending-related data (unused in current pipeline)
├── datasets/                         # (Optional) legacy data holder
├── dermDatabaseOfficial/            # Original derm database
│   └── release_v0/images/           # Raw derm image folders by class
├── dermsynth3dimages/               # Older synthesis experiments
├── synthetic_dataset/               # Output of generate_synthetic.py (new synthetic imgs)
├── outputs/                         # Results, logs, eval images, PCA plots
├── src/
│   ├── amazing/                     # Placeholder module (template/example)
│   ├── train_feature/
│   │   ├── generate_synthetic.py       # Generates synthetic images for weak classes
│   │   ├── train_features.py           # Trains classifier on extracted features
│   │   ├── train_combined_features.py # Trains classifier on real+synthetic features
│   │   ├── pca_visualization.py       #  Visualizes PCA clusters
│   ├── augment.py                   # Legacy script for image augmentation
│   ├── check_images.py              # Utility to preview loaded images
│   ├── evaluate_quality.py          #  Orchestrator to compute FID/SSIM
│   ├── evaluate.py                  # Implements FID + SSIM logic
│   ├── feature_extraction.py        #  Extracts ResNet18 features, saves CSV
│   ├── process.py                   # Loads and previews raw images
│   ├── visual_feature.py            # (Optional) legacy visualizer
│   ├── run.py                       #  Runs entire pipeline from generation to evaluation
│   ├── synthetic_features.csv       #  Feature CSV from synthetic images
│   ├── extracted_features_with_labels.csv # Feature CSV from real images
├── requirements.yml                 # Conda env definition
├── LICENSE
├── README.md                        # Current markdown project summary
```

---

## ⚙️ Installation
```bash
git clone https://github.com/sfu-cmpt340/2025_1_project_20.git
cd 2025_1_project_20

# create a python virtual environment
python -m venv .venv

# activate the virtual environment
source .venv/bin/activate

# install the requirements
pip install -r requirements.txt
```
Ensure Python ≥ 3.10 and PyTorch ≥ 2.0.

---

## 🔄 Reproducing the Project
```bash

cd src/

# [1] Generate synthetic images
python train_feature/generate_synthetic.py

# [2] Extract features
python feature_extraction.py

# [3] Train classifier
python train_feature/train_combined_features.py

# [4] Visualize PCA
python train_feature/pca_visualization.py

# [5] Evaluate image quality
python evaluate_quality.py
```

Or run everything at once:
```bash
python run.py
```

---

## 📊 Evaluation Summary
| Experiment        | Accuracy | FID Score |
|------------------|----------|-----------|
| Real Only        | ~76%     | —         |
| Synthetic Only   | ~90%     | 131.19    |
| Combined         | **94.12%**| —         |

> SSIM was skipped due to dimensional mismatch errors. FID was the main quality metric.

---

## 🔬 Insights
- Synthetic images improved weak class performance
- Combined features clustered tightly in PCA space
- FID score ≈120–130 showed high-quality synthesis
- RandomForestClassifier gave solid performance for feature-based data

---

## 🚀 Future Enhancements
- Try GANs or diffusion models for more realistic generation
- Use smart or conditional augmentations
- Extend to other medical imaging modalities
- Build a web-based tool for interactive class augmentation

---

## ✅ Status: Complete
- Fully working synthesis + training pipeline
- High accuracy with synthetic data
- Evaluated with PCA + FID
- One-click automation via `run.py`

---
