# 🤖 Kaggle MCP Server Agent Guide

Welcome to the **Kaggle MCP Server** repository. This document is written specifically for AI agents to understand the codebase architecture, multi-user credential profile system, tool mappings, and runtime workflows.

---

## 📖 Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Multi-User Profiles](#2-multi-user-profiles)
3. [Tool Categories & Mappings](#3-tool-categories--mappings)
4. [Key Agent Workflows & Design Patterns](#4-key-agent-workflows--design-patterns)
5. [Local Testing & Debugging](#5-local-testing--debugging)
6. [Dataset Downloader & Visualizer Workflows](#6-dataset-downloader--visualizer-workflows)
7. [Recommended Datasets (Overhead/Top-View)](#7-recommended-datasets-overheadtop-view)

---

## 1. System Architecture

This project implements a Model Context Protocol (MCP) server wrapping the `kaggle` CLI (v2.2.1). It is built using Python's `FastMCP` framework.

When an LLM client invokes an MCP tool, the server handles it as follows:
1. Validates the input arguments.
2. Formats arguments into flags matching the native `kaggle` CLI command.
3. Injects the active profile's token via environment variables.
4. Executes the CLI in a subprocess and returns the output stream as plain text.

### Repository Structure & Key Files
This repository is organized into distinct subdirectories, each containing its own `README.md` file detailing its purpose:
* **`code/`** ([README.md](file:///home/fadhil/program/people_counting_2d/code/README.md)): Contains the core source code for local CPU execution.
* **`kaggle-nb/`** ([README.md](file:///home/fadhil/program/people_counting_2d/kaggle-nb/README.md)): Dedicated to pushing Kaggle kernels and managing training notebooks.
* **`kaggle-mcp/`** ([README.md](file:///home/fadhil/program/people_counting_2d/kaggle-mcp/README.md)): Implements the FastMCP server wrapping the Kaggle CLI.
* **`helper/`** ([README.md](file:///home/fadhil/program/people_counting_2d/helper/README.md)): Contains utility scripts for development workflows, such as downloading Kaggle kernel outputs.

> [!WARNING]
> **Missing Files**: Please note that since the root environment was reset, the main `README.md` file and other legacy `.md` files in the root directory were deleted. Additionally, some files previously referenced in documentation (like `kaggle-mcp/CLAUDE.md`, `kaggle_notebooks/dataset_visualizer/` scripts, or `kaggle-mcp/dummy-kernel/` template files) may be missing and need to be recreated.

---

## 2. Multi-User Profiles

The server supports up to **4 Kaggle users** stored as profiles. This allows developers to seamlessly switch credentials during interactive workflows.

> [!NOTE]
> Profile credentials are persisted in `~/.kaggle/profiles.json` with permissions restricted to the file owner (`0o600`).

### Profile Bootstrapping & Migration
On the first run, the server automatically reads credentials in the following order and creates a profile named `"default"`:
1. `KAGGLE_API_TOKEN` environment variable.
2. `~/.kaggle/access_token` file content.
3. Fallback to the active user's environment settings.

---

## 3. Tool Categories & Mappings

The MCP server exposes various tools mapped directly to the `kaggle` CLI syntax.

### Key Tool Groups
* **Competitions**: `kaggle_competitions_list`, `kaggle_competitions_download`, `kaggle_competitions_submit`, etc.
* **Datasets**: `kaggle_datasets_list`, `kaggle_datasets_download`, `kaggle_datasets_create`, etc.
* **Kernels**: `kaggle_kernels_push`, `kaggle_kernels_logs_tail`, `kaggle_kernels_status`, etc.
* **Models**: `kaggle_models_list`, `kaggle_models_get`, `kaggle_models_instances_versions_download`, etc.

---

## 4. Key Agent Workflows & Design Patterns

### Kernel Logging: Blocking vs. Non-blocking
When running a remote Kaggle kernel, use **`kaggle_kernels_logs_tail(kernel, lines)`** instead of blocking log commands.
* It requests logs without follow (`-f`) and returns immediately.
* **Pattern**: Set up a background timer or poll sequentially (e.g. every 60s) calling `kaggle_kernels_logs_tail` to monitor progress.

---

## 5. Local Testing & Debugging

To run the MCP server manually or debug it:

```bash
# Run server using Python
python kaggle-mcp/server.py

# Run commands to test profile tools inline via python interactive shell
python -c "from server import *; print(kaggle_profile_show())"
```

---

## 6. Dataset Downloader & Visualizer Workflows

For downloading and visualizing training/testing datasets remotely on Kaggle and pulling them locally, the system uses the following components:

### Directory Layout
* **`kaggle-nb/dataset_visualizer/`**:
  * **`download_and_visualize.py`**: Python script that dynamically locates the dataset via `os.walk('/kaggle/input')`, parses the annotations (filtering for `body_valid == 1`), and renders a 1-minute `.mp4` ground-truth video highlighting only the pedestrians' bodies.
  * **`kernel-metadata.json`**: Configures the Kaggle kernel (P100 GPU enabled) and attaches the pre-requisite dataset `almightyj/oxford-town-centre`.
* **`code/`**: Scripts like `create_visualizer_notebook.py`, `local_visualize_town_centre.py`, and `download_visualizer_videos.py`.

### Oxford Town Centre Annotation Mapping
The ground truth file `TownCentre-groundtruth.top` contains **12 columns**. Standard YOLO loaders must include the two validation flags to prevent a 2-column shift:
```python
df = pd.read_csv(annotations_path, header=None, names=[
    'person_id', 'frame_idx', 'head_valid', 'body_valid',
    'head_l', 'head_t', 'head_r', 'head_b',
    'body_l', 'body_t', 'body_r', 'body_b'
])
```
* **Note**: Bounding box coordinates can contain `NaN` values when a pedestrian is partially out of frame.

---

## 7. Primary Dataset & Model

This project relies exclusively on the **Oxford Town Centre** dataset as its primary data source, and utilizes the **YOLO26** model architecture for top-view pedestrian detection and counting.

* **Primary Dataset (Oxford Town Centre)**: Provided as a Kaggle dataset `almightyj/oxford-town-centre` for tracking and pedestrian counting.
* **Primary Model (YOLO26)**: The model is fine-tuned and executed locally via ONNX to achieve optimal performance for overhead detections.

*(Note: Other datasets like SCUT-HEAD or Roboflow Universe datasets are not used in this specific project setup.)*

---

## 8. Development & Execution Rules

To maintain consistency and proper workflow throughout the project, adhere to the following rules:
1. **Execution Environment**: Code must be run in `kaggle-mcp` using a Kaggle Kernel for remote execution.
2. **Primary Kaggle Profile**: All operations using the Kaggle API and MCP server must default to the `fadhilelrizandamicr` profile.
3. **Commit First Workflow**: Any updated or newly developed code must be pushed to the Git repository first before it is executed or tested in the remote Kaggle kernel.
4. **Hardware Configuration**: All remote Kaggle kernels must be configured to run on the **P100 GPU** (`NvidiaTeslaP100`).
5. **GitHub Operations**: Use the GitHub MCP server for all Git and GitHub interactions (e.g., pushing code, managing branches, or creating pull requests).
