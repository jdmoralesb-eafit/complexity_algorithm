# Holographic Image Processing & Computational Complexity Analysis System

A comprehensive GUI-based system for off-axis hologram reconstruction and computational complexity analysis of image processing algorithms.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Test Data](#test-data)
- [Algorithms](#algorithms)
- [Testing Functions](#testing-functions)
- [Documentation](#documentation)

## Overview

This system provides an automated framework for **experimental evaluation of computational complexity** using a graphical user interface. It measures algorithm performance through hardware-level metrics (CPU instructions, processor cycles, and execution time) using the Linux `perf` utility. The system includes pre-simulated off-axis holograms at various resolutions and implements two reconstruction methods: PyDHM (Vortex-Legendre) and SHPC (Semi-Heuristic Phase Compensation).

### How It Works

1. The program processes datasets organized by input size (e.g., different image resolutions)
2. For each size, it executes the selected algorithm and records performance metrics using `perf`
3. Statistical regression models the relationship between input size and resource consumption
4. The resulting curves provide an **experimental profile of computational complexity**

## Features

- **Dual Operation Modes**:
  - **Analysis Mode**: Execute algorithms and perform complexity analysis with `perf`
  - **Visualization Mode**: Load and visualize previously computed results
- **Hardware-Level Profiling**: Measures CPU instructions, cycles, and execution time
- **Automated Regression**: Generates computational and time complexity curves
- **Multiple Resolutions**: Support for holograms from 128×128 to 4096×4096 pixels
- **Flexible Algorithm Testing**: Load any Python algorithm (.py file) for analysis
- **Real-Time Execution Log**: Monitor progress and errors during processing
- **Test Functions**: Built-in functions for validating complexity measurements (O(N²), O(N³), O(N² log N))
- **Pre-computed Results**: Sample performance data (PyDHM and SHPC) for testing visualization

## Installation

### Prerequisites

- **Operating System**: Linux or Windows with WSL2
- **Administrator Access**: Required for WSL installation on Windows

### For Windows Users: WSL Setup (Required)

The system uses the Linux `perf` utility for hardware-level performance measurements, which requires a Linux kernel. Windows users must install WSL2.

#### Step 1: Install WSL2

Open PowerShell **as Administrator** and run:

```powershell
wsl --install
```

This command will:
- Enable necessary Windows features
- Install Ubuntu (default distribution)
- Set WSL2 as the active version

**Restart your computer** after installation completes.

#### Step 2: Initial WSL Configuration

1. Launch WSL from the Start menu or type `wsl` in PowerShell/Command Prompt
2. On first run, create a username and password for your Linux environment

#### Step 3: Transfer Setup Script to WSL

If `setup.py` is on your Windows desktop (replace `<username>` with your Windows username):

```bash
cp /mnt/c/Users/<username>/Desktop/setup.py ~/
```

Alternatively, navigate to your project location:
```bash
cd /mnt/c/path/to/your/project
```

#### Step 4: Run Automated Setup

The `setup.py` script will automatically:
- Verify WSL environment
- Update system packages
- Install development tools
- Clone Microsoft WSL2 kernel sources
- Compile the `perf` utility with required libraries
- Install `perf` globally to `/usr/local/bin/`

Execute the setup:

```bash
python3 setup.py
```

**Optional parameters**:
- `--skip-update`: Skip system package updates
- `--only-build-perf`: Only recompile `perf` without reinstalling dependencies

The script creates a log file (`perf_setup.log`) in your home directory for troubleshooting.

### Step 5: Install Python Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

### Step 6: Launch the Application

```bash
python main.py
```

The GUI will open, ready for algorithm analysis or result visualization.

### For Linux Users

If you're already on Linux:

1. Ensure `perf` is installed (usually via `linux-tools-common` package)
2. Install Python dependencies: `pip install -r requirements.txt`
3. Run the application: `python main.py`

## Project Structure

```
.
├── main.py                      # Main GUI application
├── requirements.txt             # Python dependencies
├── setup.py                     # WSL configuration script
├── User_manual/                 # Complete user documentation
├── Hologram_stack/              # Simulated off-axis holograms
│   ├── 128x128/                # 10 holograms per resolution
│   ├── 256x256/
│   ├── 512x512/
│   ├── 640x480/
│   ├── 800x600/
│   ├── 1024x768/
│   ├── 1024x1024/
│   ├── 1280x960/
│   ├── 1600x1200/
│   ├── 1920x1440/
│   ├── 2048x2048/
│   ├── 2560x1920/
│   ├── 3840x2880/
│   └── 4096x4096/
├── Gray_images/                 # Noise test images (3 per resolution)
│   └── [same size folders as Hologram_stack]
├── Test_algorithm/              # Reconstruction algorithms
│   ├── PyDHM_methods.py        # PyDHM implementation
│   └── SHPC.py                 # Semi-Heuristic Phase Compensation
└── Test_files/                  # Testing utilities
    ├── test_function.py        # Complexity test functions
    ├── Vortex_Performance.txt  # Pre-computed PyDHM results
    └── SHPC_Performance.txt    # Pre-computed SHPC results
```

## Usage

### Quick Start Guide

1. **Launch the application**:
   ```bash
   python main.py
   ```

2. **Select operation mode**:
   - **Analysis Mode**: Evaluate algorithm performance
   - **Visualization Mode**: View previously computed results

### Analysis Mode Workflow

#### Step 1: Mode Selection
Select **Analysis Mode** from the top section of the GUI.

#### Step 2: Load Algorithm
1. Click **"Browse"** in the Input Section
2. Navigate to and select a Python file (`.py`) containing your algorithm
3. The system will automatically detect all functions in the file

#### Step 3: Select Main Function
From the **"Function:"** dropdown menu, select the main function you want to analyze.

#### Step 4: Configure Function Parameters

**Default Parameter (Input Dataset)**:
- Click the **folder icon** next to the first parameter
- Select a folder containing **subfolders** with images of different sizes
- Example structure:
  ```
  Hologram_stack/
  ├── 128x128/     (contains images)
  ├── 256x256/     (contains images)
  ├── 512x512/     (contains images)
  └── ...
  ```

**User-Defined Parameters**:
- Enter values for each parameter in the text fields
- **Important**: If a parameter expects a **function** as input, check the **"Is it a function?"** checkbox

#### Step 5: Execution Settings
1. **Number of Iterations**: Enter how many times to run the algorithm on each dataset (e.g., `5`)
2. **Output File Name**: Specify output filename with `.txt` extension (e.g., `vortex_analysis.txt`)

#### Step 6: Run Analysis
Click the execution button. The **Execution Log** at the bottom will show:
- Start/end times
- Current function and iteration
- Dataset being processed
- Any warnings or errors

### Visualization Mode Workflow

#### Step 1: Mode Selection
Select **Visualization Mode** from the top section.

#### Step 2: Load Results
Click **"Load Results File"** and select a `.txt` file from a previous analysis.

#### Step 3: Generate Plots
Click **"Generate Plots"**. The system will display two graphs:

1. **Computational Complexity Fit**: Shows the relationship between operations and input size
2. **Time Complexity Fit**: Shows the relationship between execution time and input size

These plots help identify whether the algorithm exhibits linear, quadratic, or higher-order growth.

## Test Data

### Hologram Stack

Pre-simulated off-axis holograms with the following parameters:

- **Pixel pitch (Δx, Δy)**: 3.75 μm
- **Wavelength (λ)**: 0.633 μm (HeNe laser)
- **Quantity**: 10 holograms per resolution
- **Available resolutions**: 14 different sizes (128×128 to 4096×4096)

### Gray Images

Noise-based test images for validating the test functions:

- **Quantity**: 3 images per resolution
- **Same resolutions** as Hologram_stack
- **Purpose**: Testing GUI behavior with `test_function.py`

## Algorithms

### 1. PyDHM (Vortex-Legendre Method)

**File**: `Test_algorithm/PyDHM_methods.py`

**Function to use**: `vortexLegendre`

**Parameters**:
```python
inp = "Hologram_stack"  # or path to custom hologram folder
wavelength = 0.633      # in micrometers
dx = 3.75               # pixel pitch in micrometers
dy = 3.75               # pixel pitch in micrometers
limit = 255/2           # intensity threshold
filter_type = "Circular"
manual_coords = None
spatial_filtering_fn = spatialFilteringCF
```

### 2. SHPC (Semi-Heuristic Phase Compensation)

**File**: `Test_algorithm/SHPC.py`

**Parameters**:
```python
archivo = "Hologram_stack"  # or path to custom hologram folder
dx = 3.75                   # pixel pitch in micrometers
dy = 3.75                   # pixel pitch in micrometers
lamb = 0.633                # wavelength in micrometers
G = 3                       # gain parameter
radio_mascara = 200         # mask radius
paso = 0.2                  # step size
```

## Testing Functions

The `Test_files/test_function.py` module provides complexity validation functions to verify that the GUI correctly identifies computational complexity patterns.

### Available Test Functions

1. **`matrix_addition(matrix)`**
   - **Complexity**: O(N²) for N×N matrices
   - **Operation**: Matrix self-addition with ones matrix
   - **Use case**: Testing quadratic complexity detection

2. **`matrix_multiplication(matrix)`**
   - **Complexity**: O(N³) for N×N matrices  
   - **Operation**: Naive matrix self-multiplication
   - **Use case**: Testing cubic complexity detection

3. **`triple_matrix_multiplication(matrix)`**
   - **Complexity**: O(N³)
   - **Operation**: Triple self-multiplication
   - **Use case**: Testing cubic complexity detection (alternative)

4. **`compute_fft_matrix(matrix)`**
   - **Complexity**: O(N² log N) for 2D FFT
   - **Operation**: 2D Fast Fourier Transform
   - **Use case**: Testing log-linear complexity detection

### How to Test with These Functions

1. Load `test_function.py` in the GUI's Input Section
2. Select one of the test functions from the dropdown
3. For the input parameter, select the **Gray_images** folder (contains noise images at various sizes)
4. Set the number of iterations (e.g., `3`)
5. Specify an output filename (e.g., `matrix_addition_test.txt`)
6. Run the analysis

The system should return complexity classifications matching the theoretical values. For example, testing `matrix_multiplication` should yield an O(N³) complexity curve.

### Using Gray Images

The `Gray_images/` folder contains noise-based test images specifically for validating test functions:
- **3 images per resolution** (vs. 10 in Hologram_stack)
- Same size options as Hologram_stack
- Purpose: Faster testing without full hologram processing

## Pre-computed Performance Data

Two performance profile files are included for **testing the visualization functionality** without running full reconstructions:

### Files

- **`Vortex_Performance.txt`**: Pre-computed PyDHM (Vortex-Legendre) algorithm benchmark results
- **`SHPC_Performance.txt`**: Pre-computed SHPC algorithm benchmark results

### How to Use

1. Launch the application and select **Visualization Mode**
2. Click **"Load Results File"**
3. Select either `Vortex_Performance.txt` or `SHPC_Performance.txt`
4. Click **"Generate Plots"** to view the complexity curves

This allows you to:
- Test the plotting functionality immediately
- Understand the expected output format
- Compare your own results against reference benchmarks
- Verify the visualization workflow before running time-consuming analyses

## Documentation

### User Manual

Complete usage instructions, detailed GUI walkthrough, and troubleshooting information are available in the **User Manual** located in the `User_manual/` folder. The manual covers:

- **Theoretical Background**: Understanding computational complexity (theoretical vs. empirical approaches)
- **Installation Guide**: Detailed WSL setup with troubleshooting steps
- **GUI Sections**: Complete description of all interface components
- **Workflow Examples**: Step-by-step analysis and visualization procedures
- **Parameter Configuration**: Detailed explanations of function parameters and execution settings
- **Execution Monitoring**: Using the real-time execution log
- **Result Interpretation**: Understanding complexity fit plots

### Quick Reference

- **Default hologram parameters**: λ = 0.633 μm, Δx = Δy = 3.75 μm
- **Supported formats**: Python (.py) files with function definitions
- **Output format**: Text files (.txt) containing performance metrics
- **Minimum dataset structure**: At least one subfolder with images (multiple subfolders recommended)

## Support

For issues or questions:

1. **First**: Consult the User Manual in `User_manual/` folder
2. **Verify**: Check that WSL and `perf` are correctly installed (`perf --version`)
3. **Check logs**: Review `perf_setup.log` for installation issues
4. **Monitor**: Use the Execution Log in the GUI to identify runtime errors
5. **Validate**: Ensure hologram parameters match specifications
6. **Test**: Use test functions with Gray_images to verify system functionality

## Technical Specifications

| Parameter | Value |
|-----------|-------|
| Wavelength (λ) | 0.633 μm |
| Pixel Pitch (Δx, Δy) | 3.75 μm |
| Image Sizes | 14 resolutions |
| Hologram Type | Off-axis |
| Reconstruction Methods | 2 (PyDHM, SHPC) |

---

**Note**: This system is designed for experimental computational complexity analysis of holographic reconstruction algorithms. Results may vary based on hardware specifications and system configuration.