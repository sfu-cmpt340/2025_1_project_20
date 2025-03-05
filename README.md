# Interactive Generation of DermSynth3D Data
This repository is a template for your CMPT 340 course project.
Replace the title with your project title, and **add a snappy acronym that people remember (mnemonic)**.

Add a 1-2 line summary of your project here.

## Acronym: DermSynth3D

Summary: DermSynth3D synthesizes realistic skin lesion images to address the lack of available data for deep learning in medicine, overcoming privacy and ethical concerns.

## What was the motive behind this topic?
The motive behind DermSynth3D is to overcome the challenges of data scarcity in dermatology for deep learning. It addresses issues such as privacy concerns, limited labeled data, and class imbalances by generating realistic synthetic skin lesion images. This synthetic data helps to augment real datasets, enabling more robust AI models for skin disease detection while adhering to privacy regulations and ethical standards.

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
git clone [$THISREPO](https://github.com/sfu-cmpt340/2025_1_project_20.git)
cd [$THISREPO](https://github.com/sfu-cmpt340/2025_1_project_20.git)
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
- Sign up for [GitHub Education](https://education.github.com/) 
