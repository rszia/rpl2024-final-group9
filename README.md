# Assessing Phi-3's Ability on Vision-Language Navigation
<em>RPL2024 Final Project Group 9</em>

# README is not done yet!!

## Directories

The `Matterport3DSimulator` directory contains everything related to the MP3D dataset, including R2R benchmark (which is base on MP3D).
However, to run the simulator, one needs to request access to the MP3D dataset by sending an email the owner ; for more details, see [their repo](https://github.com/peteanderson80/Matterport3DSimulator/).
(We have download the RGB skybox images, which is about 23GB.)

## Introduction

The **Room-to-Room (R2R) Benchmark** is designed to evaluate the navigation capabilities of AI agents using natural language instructions within the Matterport3D (MP3D) environment. While the original R2R Benchmark provides detailed instructions to aid navigation, real-world scenarios often present agents with instructions of varying levels of detail. To assess how agents perform under these conditions, we introduce two new benchmarks:

1. **Reduced-Detail Benchmark**: Strips away descriptive details, retaining only essential directional information.
2. **Goal-Only Benchmark**: Retains only the final goal location, omitting all intermediate navigational steps.

These benchmarks are generated using three different Large Language Models (LLMs) of varying sophistication: **4o-mini**, **4o**, and **o1-mini**.

## Dataset Format

Each entry in the R2R dataset is structured as follows:

```json
{
  "distance": float,
  "scan": str,
  "path_id": int,
  "path": [str x num_steps],
  "heading": float,
  "instructions": [str x 3]
}
```

- **distance**: Length of the path in meters.
- **scan**: Matterport scan ID.
- **path_id**: Unique identifier for the path.
- **path**: List of viewpoint IDs from start to goal location.
- **heading**: Agent’s initial heading in radians (elevation is always assumed to be zero).
- **instructions**: Three unique natural language instructions describing how to navigate from the start pose to the goal.

## Prerequisites

Ensure you have the following installed:

- **Python 3.7+**
- **OpenAI API Key**: Required to access the LLMs.

For reference, the environment we used contains
- numpy 1.24.4
- opencv-python 4.10.0.84
- python 3.8.17
- torch 2.4.1
- transformers 4.46.3


## Scripts

### `process.py`

Transforms the original R2R instructions into **Reduced-Detail** and **Goal-Only** instructions using the **4o-mini** and **4o** LLMs.

**Key Features**:
- Uses system prompts for instruction transformation.
- Processes multiple JSON files and outputs transformed data.

**Usage**:
```bash
python process.py
```

### `process_o1.py`

Transforms the original R2R instructions into **Reduced-Detail** and **Goal-Only** instructions using the **o1-mini** LLM.

**Key Features**:
- Integrates all necessary instructions into a single user-level prompt, as o1-mini does not support separate system prompts.
- Processes multiple JSON files and outputs transformed data.

**Usage**:
```bash
python process_o1.py
```

### `extract_100.py`

Extracts the first 100 objects from the original R2R JSON files for testing or sampling purposes.

**Usage**:
```bash
python extract_100.py
```

## Directory Structure

```
R2R-Benchmark-Enhancement/
├── 4o-mini/
│   ├── R2R_test_processed.json
│   ├── R2R_train_processed.json
│   ├── R2R_val_seen_processed.json
│   └── R2R_val_unseen_processed.json
├── 4o/
│   ├── R2R_test_processed.json
│   ├── R2R_train_processed.json
│   ├── R2R_val_seen_processed.json
│   └── R2R_val_unseen_processed.json
├── o1/
│   ├── R2R_test_processed.json
│   ├── R2R_train_processed.json
│   ├── R2R_val_seen_processed.json
│   └── R2R_val_unseen_processed.json
├── process.py
├── process_o1.py
├── extract_100.py
└── .env
```

- **4o-mini/**: Contains processed JSON files using the **4o-mini** LLM.
- **4o/**: Contains processed JSON files using the **4o** LLM.
- **o1/**: Contains processed JSON files using the **o1-mini** LLM.
- **process.py**: Script for processing with 4o-mini and 4o models.
- **process_o1.py**: Script for processing with o1-mini model.
- **extract_100.py**: Script to extract the first 100 objects from JSON files.
- **.env**: Stores environment variables (e.g., OpenAI API key).