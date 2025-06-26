# Deep Learning Training Pipeline
This project implements a modular deep learning training pipeline, developed as part of a graduate-level course on deep learning. It demonstrates practical skills in model training, logging, evaluation, and code organization using PyTorch.


## 📘 Project Overview
The goal of this project was to build and train a deep learning model using a clean, reusable codebase. Key components include:

- Model definition with PyTorch

- Training loop with dynamic logging

- Evaluation and grading framework

- Command-line interface for ease of use

⚠️ _This repository does not include assignment prompts, datasets, or instructor-provided materials to comply with university academic integrity guidelines._

## 🚀 Features
- Modular code organized into models, logger, and train components

- Custom training loop with integrated logging and visualization

- Evaluation utilities for testing model performance

- Reproducible experiments through saved configurations and scripts

## 🛠️ Technologies
- Python 3

- PyTorch

- NumPy

- Matplotlib (for plotting and logging)

- Custom evaluation scripts

## 📁 Repository Structure
├── homework/  
│   ├── train.py        # Main training script  
│   ├── models.py       # Neural network model(s)  
│   ├── logger.py       # Logging and metrics tracking  
│   └── utils.py        # Helper functions  
├── grader/  
│   ├── grader.py       # Evaluation and scoring tools  
│   └── tests.py        # Automated test cases  
├── requirements.txt    # Environment dependencies    
└── README.md           # This file

## 🔧 How to Run
Install the requirements:

pip install -r requirements.txt

Then run the training script:

python -m homework.train

_Note: Since the dataset is not included, you may need to modify train.py to load your own dataset._

## 🔍 What I Learned
- Structuring deep learning projects for clarity and reusability

- Implementing custom training and evaluation pipelines

- Debugging and testing deep learning models effectively

- Logging and visualizing model performance
