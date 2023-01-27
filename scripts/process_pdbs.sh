#!/bin/bash

codedir="../protein_holography"
datadir="../scripts"
download_pdbs=1 # 1 to download pdb files. 0 to skip download
parallelism=4
project=quick_run
subproject=example
csv_file=$datadir/$project/pdbs.csv

cd $codedir

echo "Making necessary directories"
download_tag=""
if [ $download_pdbs -eq 1 ];
then
    echo "Making pdb dir and downloading pdbs"
    mkdir $datadir/$project/pdbs
    download_tag="--download --pdb_dir $datadir/$project/pdbs"
    echo -e "Pdbs downloaded\n"
fi

echo "Making data dirs for holographic steps"
mkdir $datadir/$project/proteins
mkdir $datadir/$project/neighborhoods
mkdir $datadir/$project/zernikegrams
mkdir $datadir/$project/energies
echo -e "Directories creation attempted\n"

echo "Making csv file of pdbs"
python $codedir/coordinates/make_pdb_list.py --csv_file $csv_file \
       --hdf5_file $datadir/$project/${subproject}_pdbs.hdf5 \
       --dataset $subproject $download_tag 
echo python $codedir/coordinates/make_pdb_list.py --csv_file $csv_file \
       --hdf5_file $datadir/$project/${subproject}_pdbs.hdf5 \
       --dataset $subproject $download_tag 
echo -e "csv file made\n"

echo "Gathering structural info from pdb files"
python $codedir/coordinates/get_structural_info.py --hdf5_out \
       $datadir/$project/proteins/${subproject}_proteins.hdf5 \
       --pdb_list $subproject --parallelism $parallelism --hdf5_in \
       $datadir/$project/${subproject}_pdbs.hdf5\
       --pdb_dir $datadir/$project/pdbs/
echo -e "Done gathering structural info\n"

echo "Getting number of CAs"
output=$((python $codedir/utils/get_CAs.py \
		 $datadir/${project}/proteins/${subproject}_proteins.hdf5 \
		 $subproject) 2>&1)
echo -e "Found " $(($output)) " CAs in structural info\n"

echo "Gathering neighborhoods"
python $codedir/coordinates/get_neighborhoods.py --hdf5_out \
       $datadir/$project/neighborhoods/${subproject}_neighborhoods.hdf5 \
       --hdf5_in $datadir/$project/proteins/${subproject}_proteins.hdf5 \
       --protein_list $subproject --parallelism $parallelism --num_nhs $output --r_max 10. 
echo -e "Done gathering neighborhoods\n"

echo "Gathering zernikegrams"
python $codedir/coordinates/get_zernikegrams.py --hdf5_out \
       $datadir/$project/zernikegrams/${subproject}_zernikegrams.hdf5 \
       --hdf5_in $datadir/$project/neighborhoods/${subproject}_neighborhoods.hdf5 \
       --neighborhood_list $subproject --parallelism $parallelism --num_nhs $output --Lmax 5 \
       -k 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --r_max 10.
echo -e "Done gathering zernikegrams\n"


echo "Making predictions"
python predict/predict.py --input zgram \
       --zgram_file $datadir/$project/zernikegrams/${subproject}_zernikegrams.hdf5 \
       --zgram_dataset $subproject --outfile $datadir/$project/energies/${subproject}_pseudoenergies.csv \
       --pnE_outfile $datadir/$project/energies/${subproject}_pnEs.csv \
       --network_dir $codedir/model_weights/best_network
echo -e "Done making predictions\n"
