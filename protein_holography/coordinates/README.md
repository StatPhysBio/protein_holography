# README file for the coordinate gathering process.


## get_structural_info.py

   Purpose:
	The purpose of this script is to collect the structural info from
	a collection of pdb files. The structural info collected includes
	the coordinates of all atoms in the protein as well as five pieces
	of information at each of these coordinates
	   1. atom_names (e.g. CA, CB, OE, etc.)
	   2. elements (e.g. C, N, O, S, H)
	   3. coords - (x,y,z) coords of each atom as given in the pdb file
	   3. res_ids - ids of the residue each atom belongs to
	   4. SASAs
	   5. charges

Arguments:

	--hdf5_in: hdf5 file that contains a list of pdb ids which
		   this program will collect the structural info of
	--pdb_list: name of the dataset in hdf5_in file that contains
		    the list of pdbs
	--pdb_dir: path to the directory where the pdb files exist
	--hdf5_out: Output hdf5 file to write protein structual info to
     	--parallelism: number of processes to use

## get_neighborhoods.py

Purpose:
	The script will then collect amino acid neighborhoods from the
	previously collected structural info. It will collect neighborhoods
	for all residues in the proteins given in the hdf5_in file. The
	format of the data is the same as in the previous step except the
	coordinates are now spherical centered around the central residue's
	alpha carbon.

Arguments:

	--hdf5_in: hdf5 file that contains a np array of proteins which
		   this program will break into neighborhoods
	--protein_list: name of the dataset in hdf5_in file that contains
		    the np array of proteins (usually the same as the
		    pdb_list above if you're using this pipeline)
	--hdf5_out: Output hdf5 file to write the neighborhoods to
	--num_nhs: The number of neighborhoods for declaration of the array in
		   the hdf5_out file. Can be an upper bound.
	--r_max: Radius of the neighborhoods
	--unique_chains: Bool flag for only gathering neighborhoods for
			 one chain of a set that has identical amino acid
			 sequences
     	--parallelism: number of processes to use

## get_holograms.py

Purpose:
	Finally this program will project local protein structures in
	spherical coordinates into zernikegrams.
	
Arguments:

	--hdf5_in: hdf5 file that contains a np array of neighborhoods which
		   this program will project into fourier space
	--neighborhood_list: name of the dataset in hdf5_in file that contains
		    the np array of neighborhoods (usually the same as the
		    protein_list above if you're using this pipeline)
	--hdf5_out: Output hdf5 file to write the zernikegrams to
	--num_nhs: The number of neighborhoods for declaration of the array in
		   the hdf5_out file. Can be an upper bound.
	--r_max: Radius of the neighborhoods
	--Lmax: Maximum spherical frequency to use
	-k: All the wavenumbers to use in the projection
	    For example, to use all the integers between 0 and 5 inclusive
	    as well as 17 and 100 you can put -k 0 1 2 3 4 5 17 100
	--parallelism: number of processes to use
	
