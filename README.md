# Autonomous Drone Landing & Risk Detection Pipeline (JAX/Flax NNX)


## Overview
This repository hosts a framework for parsing aerial/satellite raster imagery and identifying complex terrain configurations or micro-hazards critical for drone landing decisions.

### Key Modules

**Data Ingestion Layer**: A static-shape, generator-backed infrastructure converting raw elements into strictly bounded arrays ```(256x256) ```to ensure lightning-fast ahead-of-time (AOT) XLA compilation without re-compilation overheads.

**Model Engine ```(FlaxASPPConv)```**: A functional refactor utilizing the modern object-oriented yet purely functional Flax NNX paradigm, explicitly separating weights (```nnx.Param```) and running state tracking (```nnx.BatchStat```).

**Loss & Metrics Optimization**: Smooth, continuous Cross-Entropy and Dice loss formulations optimized for gradient backpropagation using ```jax.grad```.

## Installation & Setup

### Prerequisites

- Python 3.10+

- JAX & Flax

- Pillow, NumPy

```sh
# Clone the repository
git clone https://github.com/michaelmukiibi/drone-landing-safety.git
cd drone-landing-safety

# Install dependencies
uv sync

# Run
uv run train.py
```
## Evaluation Metrics

o counter heavy spatial class imbalances (e.g., small hazards hiding within a massive safe open field), standard pixel accuracy measurements are bypassed.

Instead, the framework computes **Intersection over Union (IoU)**

$$\text{IoU} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)} + \text{False Negatives (FN)}}$$

This guarantees that a model prioritizing overall pixel count accuracy over explicit spatial micro-hazard boundary detection is caught and properly penalized during evaluation phases.

