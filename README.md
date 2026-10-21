# Computational Complexity Analysis of Holographic Reconstruction Algorithms

A GUI that **measures** how an image-processing algorithm scales with input size (CPU instructions and elapsed time, collected with the Linux `perf` utility) and **classifies** the empirical complexity (O(N), O(N log N), O(N²), …) from those measurements.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Workflow at a Glance](#workflow-at-a-glance)
- [Main GUI, Section by Section](#main-gui-section-by-section)
- [How a Measurement Works](#how-a-measurement-works)
- [How the Classification Works](#how-the-classification-works)
- [Test Data and Included Results](#test-data-and-included-results)
- [Algorithms](#algorithms)
- [Tests](#tests)
- [Credits](#credits)

## Overview

For every size folder in a dataset, the program runs the selected function on each image under `perf`, writes the raw counters to a `.txt` file, and later fits a family of complexity models to those counters. The result is an **experimental complexity profile**: a verdict (or an explicit abstention when the data cannot support one) plus two plots, instructions vs. N and time vs. N, where N is the number of pixels.

Typical use: compare holographic phase-compensation algorithms (SHPC, FRS, ERS, PCA) and validate the method itself against functions of known complexity.

## Installation

`perf` only exists on Linux, so on Windows everything runs through **WSL2**. The GUI runs on Windows and calls into WSL for each measurement.

### Windows (automated)

From a Windows terminal, **as Administrator the first time**:

```powershell
python setup_windows.py
```

It installs WSL2 + Ubuntu if missing (Windows may ask for a **restart**; run the same command again afterwards), initializes the distribution as `root` (no user/password needed), and runs `setup.py` inside WSL to compile and install `perf`. Finally it prints `perf --version`.

### Windows (manual) / Linux

Inside WSL (or any Linux where `perf` is not available as a package):

```bash
python3 setup.py                 # clones the WSL2 kernel sources and builds only tools/perf
python3 setup.py --skip-update   # skip apt update/upgrade
python3 setup.py --only-build-perf
```

On regular Linux, `perf` is usually available through `linux-tools-*` packages and `setup.py` is not needed. Log: `~/perf_setup.log`.

### Python dependencies

Install on **Windows** (where the GUI runs):

```bash
pip install -r requirements.txt
```

Functions under analysis are executed inside WSL, so the libraries they import (numpy, scipy, Pillow, opencv-python, scikit-image …) must also be installed **in WSL**. The GUI checks this before the first run and tells you what is missing.

### Launch

```bash
python Main.py
```

## Project Structure

```
.
├── Main.py                      # GUI: Analysis tab + Visualization tab
├── perf_protocol.py             # Measurement protocol (what is run under perf)
├── complexity_fit.py            # Parser, model fitting, verdict (no GUI code)
├── visualizer_v2.py             # Plot builder + verdict report used by the Visualization tab
├── test_complexity_fit.py       # Unit tests of the fitting module
├── setup_windows.py             # One-command WSL2 + perf setup from Windows
├── setup.py                     # Builds perf inside WSL/Linux
├── requirements.txt
├── Hologram_stack/              # Simulated off-axis holograms (11 sizes x 10)
├── Test_algorithm/              # Algorithms that were measured
│   ├── SHPC.py                  # Semi-Heuristic Phase Compensation
│   ├── phaseCompensation.py     # FRS, ERS, CFS, CNT (from pyDHM)
│   └── karina_methods.py        # PCA (and PVM) phase compensation
├── Test_files/                  # Validation functions and their results
│   ├── test_functions.py        # Functions of known complexity
│   ├── crear_imagenes_de_testeo.py
│   └── Lineal / fftl / quadratic / cubic .txt
└── Algorythm_results_windows/   # Results of the measured algorithms
    └── SHPC / FRS / ERS / PCA .txt
```

## Workflow at a Glance

```
 1. Dataset          Hologram_stack/  (one subfolder per size)
        |
 2. Analysis tab     pick a .py file -> function -> parameters -> repetitions -> Run
        |            (Main.py + perf_protocol.py, runs perf in WSL)
        v
 3. Results .txt     raw perf counters per image, grouped by size folder
        |
 4. Visualization    load one or more .txt -> Generate Plots
        |            (complexity_fit.py + visualizer_v2.py)
        v
 5. Verdict + plots  model, equation, R², confidence — or an explicit abstention
```

Steps 2 and 4 are independent: you can classify results measured earlier (or on another machine) without re-running anything.

## Main GUI, Section by Section

The window has two tabs and a menu (`File`: open Python file `Ctrl+O`, load results `Ctrl+L`, exit; `Analysis`: run `Ctrl+R`, generate plots `Ctrl+G`).

### Analysis tab — *produce measurements*

| Section | What it does |
|---|---|
| **Python File** | Browse to the `.py` file with the algorithm. All its functions are detected and listed. |
| **Function** | The entry point to measure. Selecting it builds the parameter form from its signature (and any defaults). |
| **Function Parameters** | One field per parameter. The **first parameter is the dataset**: use the folder button to pick a directory whose subfolders contain images of different sizes (e.g. `Hologram_stack/`). Main replaces it with each image in turn. For parameters that expect a *function* (e.g. a spatial filter), tick *Is it a function?* so the value is not quoted. `None`, integers and floats are converted automatically. |
| **Repetitions** | How many times each measurement block is repeated per image. More repetitions give tighter medians and better error bars, at the cost of time. |
| **Output file** | Where the `.txt` results are written. The file is written progressively and flushed after each image, so a long run can be inspected (or recovered) while running. |
| **Run Analysis** | Checks WSL dependencies, sorts the size folders from smallest to largest, and measures every image. |
| **Execution Log** | Constructed call, baseline progress, current folder and image, errors. The console also prints `[folder i/n]` and `[image j/m]` progress. |

At the end the file contains a summary (`Total runs`, `Errors found`, `Success rate`).

### Visualization tab — *interpret measurements*

| Control | What it does |
|---|---|
| **Load Results Files** | Load one or several `.txt` files (several files are overlaid for comparison, each with its own color). Unreadable files — wrong format, or files where `perf` never produced counters — are reported instead of silently ignored. |
| **Generate Plots** | Fits the models and draws two stacked plots: *Instructions vs. N* and *Time vs. N* (N = total pixels). Points are medians with error bars; a dashed curve is the winning model with its equation, R² and bootstrap confidence. |
| **Clear Files** | Empties the loaded-files list. |
| **Download Plot** | Saves the current figure (format chosen by extension, e.g. `.png`, `.svg`, `.pdf`). |
| **Log** | Per-file report: chosen model, ranking of all candidate models, discarded points, warnings. |

Warnings (e.g. a sample dominated by powers of two) appear in an orange line under the controls, not on the plots.

**Reading the plots.** Each point is the *median* of the repetitions and the bar is the *MAD* (median absolute deviation). Bars are absolute, so on a wide log scale they are invisible for small sizes where the variation is a tiny fraction of the value. A dotted curve labeled `[sin veredicto]` is the best BIC fit shown for reference only when the classifier abstained.

## How a Measurement Works

Each image is measured in separate `perf` processes (so memory and caches do not leak between blocks), following the **v2 protocol** in `perf_protocol.py`:

| Block | Contents | Repeated |
|---|---|---|
| `L` | libraries + module, no image, no call | once per run |
| `LI` | `L` + loading the image | R times per image |
| `L1` | `LI` + 1 call to the function | R times per image |
| `LK` | `LI` + K calls in a loop (K ≥ 2) | R times per image |

The cost of **one steady-state call** is `(LK − L1) / (K − 1)`: it removes process start-up, library import, image loading and the cold first call. K is chosen adaptively so the loop lasts about one second. These raw blocks stay in the `.txt` so the analysis can be redone later.

The counters collected are `instructions`, `cycles`, `cache-references`, `cache-misses` and elapsed time (user-space only).

## How the Classification Works

`complexity_fit.py` reads the `.txt`, aggregates the repetitions of each size (median and MAD), and fits these candidate models by relative least squares:

`O(1)`, `O(√N)`, `O(N)`, `O(N log N)`, `O(N^1.5)`, `O(N²)`, `O(N³)` and a free-exponent `O(N^p)`.

The winner is chosen by BIC and its stability is checked with a hierarchical bootstrap (the *confidence* shown). The classifier **abstains** instead of guessing when the data does not support a verdict (too few sizes, ambiguous models, a sample biased by cache effects, etc.). Powers of two are flagged because they cause cache-aliasing bias; that is why the dataset also includes non-power-of-two sizes (192, 384, 768, 1536, 3072).

## Test Data and Included Results

### Hologram_stack

Simulated off-axis holograms: λ = 0.633 µm, pixel pitch 3.75 µm, 10 holograms per size, 11 square sizes:
power-of-two family 128, 256, 512, 1024, 2048, 4096 and 5-smooth non-power-of-two family 192, 384, 768, 1536, 3072. `simulation_summary.txt` has the generation parameters.

### Validation functions (`Test_files/test_functions.py`)

Functions of known complexity used to check that the method recovers the right law:

| Function | Expected | Results file |
|---|---|---|
| `process_nm` | O(N) | `Lineal.txt` |
| `process_nm_logn` | O(N log N) | `fftl.txt` |
| `process_nm2` | O(N²) | `quadratic.txt` |
| `process_nm3` | O(N³) | `cubic.txt` |

> `quadratic.txt` and `cubic.txt` were measured on dedicated datasets (`Hologram_Simulations_nm2`, `_nm3`) that are not included in this repository. Their results can still be loaded and plotted; to re-measure them, generate suitable image sets of the same sizes.

### Algorithm results (`Algorythm_results_windows/`)

Measurements of SHPC, FRS, ERS and PCA over `Hologram_stack`, produced on Windows + WSL2. Load them in the Visualization tab to reproduce the plots without re-running anything.

## Algorithms

All are measured through `Main.py`: select the file, then the function.

| Algorithm | File → function | Notes |
|---|---|---|
| SHPC | `Test_algorithm/SHPC.py` → `shpc_reconstruction` | `dx=3.75, dy=3.75, lamb=0.633, G=3, radio_mascara=200, paso=0.2` |
| FRS | `phaseCompensation.py` → `FRS` | `inp, upper, wavelength, dx, dy, s=2, step=10` |
| ERS | `phaseCompensation.py` → `ERS` | `inp, upper, wavelength, dx, dy, s, step` |
| PCA | `karina_methods.py` → `PCA` | `wavelength=0.632, dx=3.75, dy=3.75, limit=16` |

In every case the first parameter (`archivo` / `inp`) is the one Main fills with each image. Parameters in the original runs are recorded in the header of each results file (`Original parameters: …`).

## Tests

```bash
pip install pytest
pytest test_complexity_fit.py
```

Covers the parser, the model fitting and the abstention logic of `complexity_fit.py`.

## Credits

- `Test_algorithm/phaseCompensation.py` is part of **pyDHM** (C. Trujillo, A. Doblas, R. Castañeda; <https://github.com/catrujilla/pyDHM>) and keeps its original header. Check its license before redistributing.
- `Test_algorithm/karina_methods.py` adapts the PCA/PVM phase-compensation algorithms of K. Ortega and R. Restrepo (SOPHIA, EAFIT).
- The `perf` build in `setup.py` uses Microsoft's WSL2 Linux kernel sources.

---

**Note**: results depend on hardware and system configuration; compare algorithms only from measurements taken on the same machine under the same conditions.
