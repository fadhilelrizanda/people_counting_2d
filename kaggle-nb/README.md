# Kaggle Notebooks Directory

This directory (`kaggle-nb`) is dedicated to pushing Kaggle kernels. 

## Purpose
It stores the notebooks (`.ipynb`), scripts, and configuration files (such as `kernel-metadata.json`) required to train, fine-tune, or perform heavy inference tasks remotely on Kaggle's GPU infrastructure.

## Structure
To keep tasks organized, **each specific execution** (e.g., dataset preparation, YOLO26 training, inference evaluation) should have its own dedicated subdirectory within `kaggle-nb/`. Each subdirectory will contain its own isolated `kernel-metadata.json` and notebook/script, allowing you to push independent jobs to Kaggle.
