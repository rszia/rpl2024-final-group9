# Assessing Phi-3's Ability on Vision-Language Navigation
<em>RPL2024 Final Project Group 9</em>

This repo contains the code we wrote for the project.

## How to run the code

For running the Matterport3D dataset, we only put the `tasks/R2R` folder here. 

You need to clone the repo for [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator/) here, and replace the `tasks/R2R` folder with ours.
Note that in order to run the simulator, one needs to request access to the MP3D dataset by sending an email the owner, and agree to the Terms of Service. Detailed instructions can be found in their repo linked above.
(The entire dataset is about 1TB. If you only download the RGB skybox images, they will take about 23GB.)

After that, to run the agent, you need to go inside the folder `tasks/R2R` and specify the paths for reading path to the data. The files to be modified are `my_agents_eval.py` (which is the replacement to `eval.py`), `env.py`, and `agent.py`. Then run

```
python tasks/R2R/my_agents_eval.py
```

<!-- ## Introduction

The **Room-to-Room (R2R) Benchmark** is designed to evaluate the navigation capabilities of AI agents using natural language instructions within the Matterport3D (MP3D) environment. While the original R2R Benchmark provides detailed instructions to aid navigation, real-world scenarios often present agents with instructions of varying levels of detail. To assess how agents perform under these conditions, we introduce two new benchmarks:

1. **Reduced-Detail Benchmark**: Strips away descriptive details, retaining only essential directional information.
2. **Goal-Only Benchmark**: Retains only the final goal location, omitting all intermediate navigational steps.

These benchmarks are generated using three different Large Language Models (LLMs) of varying sophistication: **4o-mini**, **4o**, and **o1-mini**. -->

<!-- ## Dataset Format

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
- **instructions**: Three unique natural language instructions describing how to navigate from the start pose to the goal. -->

## Environment

Ensure you have the following installed:

- **Python 3.7+**
- **OpenAI API Key**: Required to access the LLMs.

For reference, the environment we used contains
- numpy 1.24.4
- opencv-python 4.10.0.84
- python 3.8.17
- torch 2.4.1
- transformers 4.46.3


## Scripts for generating benchmark

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

<!-- Inside the `tasks/R2R/data_another_bench` is our generated data, which contains
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
├── o1-mini/
│   ├── R2R_test_processed.json
│   ├── R2R_train_processed.json
│   ├── R2R_val_seen_processed.json
│   └── R2R_val_unseen_processed.json
├── process.py
├── process_o1.py
├── extract_100.py
└── .env
``` -->

- **4o-mini/**: Contains processed JSON files using the **4o-mini** LLM.
- **4o/**: Contains processed JSON files using the **4o** LLM.
- **o1-mini/**: Contains processed JSON files using the **o1-mini** LLM.
- **4o-mini-45-ins/**, **4o-45-ins/**, **o1-mini-45-ins/**: The same as above, but only keep the first 45 instructions. Used when inference is taking too long.

Other scripts you may find useful are listed in the `.` folder.

- **process.py**: Script for processing with 4o-mini and 4o models.
- **process_o1.py**: Script for processing with o1-mini model.
- **extract_100.py**: Script to extract the first 100 objects from JSON files.
- **.env**: Stores environment variables (e.g., OpenAI API key).