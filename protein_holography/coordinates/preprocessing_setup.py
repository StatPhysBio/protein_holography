import h5py

def printname(x):
    print(x)

with h5py.File('/gscratch/spe/mpun/protein_holography/data/preprocess/casp11_training30.hdf5','r') as f:
    f.visit(printname)

#with h5py.File('/gscratch/spe/mpun/protein_holography/data/coordinates/1PGA_SASA.hdf5','r') as f:
#    f.visit(printname)

    
