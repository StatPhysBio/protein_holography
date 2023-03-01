

![image](https://user-images.githubusercontent.com/16233540/197881942-d5de8f34-9aa6-4c83-aba1-53efa27293c5.png)

# Protein holography

## Overview

The protein holography package implements efficient rotationally-equivariant encoding of protein structure and minimal rotationally-equivariant processing of protein microenvironements via H-CNN. 

## Installation

### pyRosetta

This package is dependent on pyrosetta which can be downloaded from [here](https://www.pyrosetta.org/downloads#h.6vttn15ac69d). A license is available at no cost to academics and can be obtained [here](https://www.pyrosetta.org/home/licensing-pyrosetta).

The env.yml file should be edited upon download with the local path to the wheel file to install.

### setup

Once the pyrosetta wheel file has been downloaded and the path has been specified in the env.yml file, one can create the protein holography conda environment by running

```bash
conda env create -f env.yml
```

to install the necessary dependencies.
Then run

```bash
pip install .
```

to install the `protein_holography` package.
If you're going to make edits to the `protein_holography` package, run

```bash
pip install -e .
```

so you can test your changes.

### Testing install

The installation can be tested by running `pytest tests`. Currently only the preprocessing pipeline is tested. Testing will be implemented soon for the network.

## Quick run

A bash script for complete processing of pdb files is located in `scripts`. This script requires simply a csv file with protein pdb IDs and, in addition to intermediate outputs, will produce predicted pseudoenergies and probabilities for all sites in a protein as well as the protein network energy for all chains in the proteins. See `scripts` for an example and more details.

## Detailed overview

### Components

#### pdb preprocessing module

The pdb preprocessing module filters pdbs by criteria such as imaging type (e.g., X-ray crystallography, cryo-EM, etc.), resolution, date of deposition, or any other metadata deposited with the structure.

#### Coordinates
The coordinate module features all preprocessing of pdb files and ultimately results in the holograms that are used in the H-CNN. Specifically, pdb files are processed in three steps. 
##### chemical inference and coordinate extraction via PyRosetta
First, pdb files are read into pyrosetta where hydrogen atom positions are inferred, partial charges are assigned, and solvent-accessible surface area (SASA) is calculated on a per atom basis.
##### neighborhood segmentation
Second, neighborhoods of a fixed radius are extracted from each structure. 
##### holographic projection
Third, each neighborhood is projected into Fourier space via the 3D zernike polynomials. 

#### H-CNN

The hnn class is a fully fourier neural network coded in tensorflow and operates on fully complex inputs.


