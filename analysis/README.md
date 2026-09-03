# Open4DFlow — Analysis Pipeline

This folder contains the post-processing pipeline for reconstructing and extracting velocity, displacement and strain data from 4D flow MRI acquisitions.

A demo notebook with example data is available: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KA2dQRTU1l7U4Darnd9UKPh657nCDPp5#scrollTo=Q05HA9ElTfg6) → data is fetched automatically from [Zenodo](https://zenodo.org/records/20085845).

## Included Pipeline Scripts

- **`reconstruct_and_process.py`** – Converts raw ISMRMRD data to BART format, performs ESPIRiT coil sensitivity estimation and compressed sensing reconstruction per venc and temporal phase (with optional joint reconstruction and GPU acceleration), computes phase-difference velocities, and saves the result in [ORMIR-MIDS](https://github.com/ORMIR-MIDS/ORMIR_MIDS) NIfTI format.
  > **Note:** If converting from Siemens XA files, use this fork of the converter: [fsantini/siemens_to_ismrmrd](https://github.com/fsantini/siemens_to_ismrmrd)

Usage: 
```
reconstruct_and_process.py [-h] [--output OUTPUT] [--no-gpu] [--cc CC] [--joint] [--keep-bart] mrd json_config

Reconstruct and process 4D flow MRI data.

positional arguments:
  mrd              Path to the .mrd input file.
  json_config      Path to the sidecar JSON config file.

options:
  -h, --help       show this help message and exit
  --output OUTPUT  Output directory (defaults to same directory as the MRD file).
  --no-gpu         Disable GPU acceleration.
  --cc CC          Number of compressed coils for the processing. Default: no compression
  --joint          Perform joint reconstruction of phases and velocities
  --keep-bart      Keep bart files
```
 
- **`calc_strain.py`** – Computes 3D displacement fields from velocity data and calculates the full strain tensor via Savitzky-Golay spatial derivatives, outputting principal strain eigenvalues and eigenvectors in [ORMIR-MIDS](https://github.com/ORMIR-MIDS/ORMIR_MIDS) NIfTI format.

Usage: 
```
calc_strain.py [-h] [--output-plot OUTPUT_PLOT] [--no-display] input

Calculate 3D strain from ORMIR-MIDS velocity data.

positional arguments:
  input                 Path to *_vel.nii.gz file, or ORMIR-MIDS subject folder (mr-quant/*_vel.nii.gz is used in that case)

options:
  -h, --help            show this help message and exit
  --output-plot OUTPUT_PLOT
                        Save strain plot to this file (.png)
  --no-display          Do not display plots
```
- **`geometry.py`** – Extracts geometry information from ISMRMRD data to compute the affine matrix required for NIfTI export (not intended to be run as standalone).
## Installation

```bash
pip install -r requirements.txt
```

### Installing BART
For BART toolbox installation, follow the instructions at the [BART website](https://mrirecon.github.io/bart/). Make sure that the BART python package is installed and that it is findable in the Python path. For example (when using bash shell, provided that the bart binary is already installed, for example using apt):
```
git clone https://codeberg.org/mrirecon/bart.git
export PYTHONPATH=$PYTHONPATH:"$(pwd)/bart/python"
```
 > **Note:** BART does not support native Windows execution. If you are on Windows, you must run this repository inside **Windows Subsystem for Linux (WSL2)** with NVIDIA drivers configured for WSL if GPU acceleration is needed.

## Usage
For a usage example, refer to the following Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KA2dQRTU1l7U4Darnd9UKPh657nCDPp5#scrollTo=Q05HA9ElTfg6)

## Demo Data
Demo data are available on [Zenodo](https://zenodo.org/records/20085845).

## Notes

These scripts are intended for research purposes.
