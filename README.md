# Neural-Network-Based Relative Pose Estimation for Autonomous Formation Flying

## Overview

This project develops a neural-network-based system to estimate the relative position and roll orientation of a leader aircraft with respect to a follower aircraft using visual features.

A synthetic dataset is generated to simulate aircraft poses and their corresponding camera-observed structural features. A multilayer perceptron (MLP) classifies these observations into discrete pose classes.

## Problem Statement

Develop a vision-based machine learning model to estimate a leader aircraft's relative 3D position and orientation from noisy or occluded visual features for precise autonomous formation flying.

## Objectives

- Generate a synthetic dataset of aircraft poses and visual feature observations.
- Represent 14 structural features using their pixel coordinates and visibility status.
- Train a neural network to predict the aircraft's relative pose.
- Evaluate model performance using classification accuracy and pose estimation errors.
- Demonstrate predictions by comparing actual and estimated poses.

## Methodology

The project consists of four stages:

1. **Synthetic Dataset Generation:** Simulate aircraft poses and project predefined 3D structural points onto a camera image using a simplified pinhole camera model.
2. **Feature Representation:** Represent 14 features using their horizontal pixel coordinate, vertical pixel coordinate, and visibility status, resulting in 42 input values.
3. **Model Training:** Train an MLP with two hidden layers of 100 neurons each and an output layer representing 4,800 discrete pose classes.
4. **Evaluation:** Measure classification accuracy and position/orientation errors on unseen test data.

The synthetic observations incorporate measurement noise and simulated feature occlusion.

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure

```text

formation-pose-nn/
├── data/
│   ├── dataset.npz
│   └── dataset_sample.csv
├── docs/
├── models/
│   ├── .gitkeep
│   └── pose_net.pt
├── results/
│   ├── accuracy_vs_visibility.png
│   ├── actual_vs_predicted.png
│   ├── demo_frame.png
│   ├── demo_trajectory.png
│   ├── error_histograms.png
│   ├── metrics.json
│   ├── training_curves.png
│   └── training_history.json
├── src/
│   ├── aircraft_model.py
│   ├── config.py
│   ├── demo.py
│   ├── evaluate.py
│   ├── generate_dataset.py
│   ├── model.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup and Installation

**1. Clone the repository**

```bash
git clone <your-repository-url>
cd vision-pose-estimation
```

**2. Create a virtual environment (recommended)**

```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```

Linux/macOS:
```bash
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

## How to Run

Run the following commands from the project root, adjusting paths to match the implemented scripts.

**Step 1: Generate the synthetic dataset**

```bash
python data/generate_dataset.py
```

**Step 2: Train the neural network**

```bash
python src/train.py
```

**Step 3: Evaluate the model**

```bash
python src/evaluate.py
```

**Step 4: Run the demonstration**

```bash
python src/demo.py
```

The training process should save the trained model, and the evaluation script should report the metrics and generate relevant plots.

## Evaluation Metrics

The model is evaluated using:

- Exact pose-classification accuracy
- Top-5 classification accuracy
- Mean absolute position error along the x, y, and z axes
- Roll angle error
- Three-dimensional position error

## Results

In the current reported experiment, the model achieved approximately 48% test accuracy for exact pose-class prediction. The reported mean 3D position error was approximately 6.6 metres.

Results may vary depending on the dataset, training configuration, and random initialization.


## Future Work

- Improve the realism of the synthetic dataset.
- Evaluate the model on real-world aircraft images.

## References

Punnoose, R. *Vision-Based Precision Pose Estimation For Autonomous Formation Flying*. Stanford University.

