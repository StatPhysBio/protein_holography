## Quick run script

Quick HCNN prediction can be run with `process_pdbs.csv`. This script will run preprocessing,
neighborhood extraction, Zernike-spherical harmonic transform, and finally HCNN prediction.

The script requires specification of a few parameters:
- A csv file with one column named pdb. Each entry in the column should be a pdb file name without the extension.
- A binary variable on whether pdbs should be downloaded or not. For custom pdb files or pdbs that exist on
the machine already, this should be specified to be 0.
- An existing directory for writing data.
- The number of workers to use for all processing.


For the exmaple provided, these variables are given as 
```bash
codedir="../protein_holography"
datadir="../scripts"
download_pdbs=1 # 1 to download pdb files. 0 to skip download
parallelism=4
project=quick_run
subproject=example
csv_file=$datadir/$project/pdbs.csv
```
