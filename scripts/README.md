## Quick run script

Quick HCNN prediction can be run with `process_pdbs.csv`. This script will run preprocessing,
neighborhood extraction, Zernike-spherical harmonic transform, and finally HCNN prediction.

The script requires specification of a few parameters:
- A csv file with one column named pdb. Each entry in the column should be a pdb file name without the extension.
- A binary variable on whether pdbs should be downloaded or not. For custom pdb files or pdbs that exist on
the machine already, this should be specified to be 0.
- An existing directory for writing data.
- The number of workers to use for all processing.


For the example provided, these variables are given as 
```bash
codedir="../protein_holography"
datadir="../scripts"
download_pdbs=1 # 1 to download pdb files. 0 to skip download
parallelism=4
project=quick_run
subproject=example
csv_file=$datadir/$project/pdbs.csv
```

Running the example script will create the following output.

### Predictions

Predictions for all sites in the pdbs are stored in `$datadir/$project/energies/${subproject}_pseudoenergies.csv`. 
Furthermore, a summary output of protein network energies (i.e. the sum of the pseudoenergies of the crystal protein sequence)
is saved in `$datadir/$project/energies/${subproject}_pnEs.csv`.

### Zernikegrams

Zernikegrams are saved in `$datadir/$project/zernikegrams/${subproject}_zernikegrams.hdf5`.

### Neighborhoods

Neighborhoods are stored in `$datadir/$project/neighborhoods/${subproject}_neighborhoods.hdf5`.

### Preprocessed proteins

Preprocessed proteins are stored in `$datadir/$project/proteins/${subproject}_proteins.hdf5`.
