# Pulseq-4DFlow

Pulseq-4DFlow provides fully sampled and undersampled 4D flow MRI sequence files created using [PyPulseq](https://github.com/imr-framework/pypulseq). These sequences are designed for experimental use and can be exported as `.seq` files to run directly on compatible MRI scanners.

## Project Overview

This repository contains several Pulseq-based 4D flow sequences for testing and development:

- **4DFlow** – The standard, fully sampled 4D flow sequence.  
- **4DFlow_undersampled** – A pilot sequence that explores scan time reduction through undersampling.
- **gradient_probing** – Asequnce to be run once before any scans to map the physical direction of the gradiens on your specific system.  
- **Undersampling_arteries** – Application of the undersampled sequence for neurovascular application (this version is for a 3T Siemens Vida scanner).  
- **Undersampling_leg** – Version adapted for lower limb (leg) imaging (this version is for a 3T Siemens Vida Fit scanner).  
- **Undersampling_forearm** – Version adapted for upper limb (forearm) imaging (this version is for a 3T Siemens Prisma scanner).

Each sequence produces a `.seq` file that can be executed on the scanner using PyPulseq.


## Notes

These sequences are intended for research and educational purposes. Parameters, timing, and gradients should be carefully verified before in vivo use and potentially adapted to you system.

***
