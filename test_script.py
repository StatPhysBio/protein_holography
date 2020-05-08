import pdb_interface as pi
import os

os.chdir('../training30')
res = pi.sample_amino_acid_from_protein('1MG7','TRP')
print([x for x in res])
shapes = pi.get_res_atomic_coords(res)
print(shapes)

protein_dir = '/home/mpun/scratch/protein_workspace/casp7/validation'
protein_list = pi.get_proteins_from_dir(protein_dir)

instances = 2
d = 2.
r_h = 5.
k = 0.1
L = 2
h,hl = pi.get_amino_acid_shapes_from_protein_list(protein_list,
                                               protein_dir,
                                                  instances,d,r_h,k,L)

print('Successfully terminating')

