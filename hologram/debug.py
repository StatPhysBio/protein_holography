import numpy as np
import Bio.PDB as pdb


parser = pdb.PDBParser()
p = '1GTV'
pdir = '/gscratch/stf/mpun/data/casp11/training30'

struct =  parser.get_structure(p + 'struct',pdir + '/' + p + '.pdb')

all_res = pdb.Selection.unfold_entities(struct,'R')
print(all_res[0].get_resname())
ASNs = [x.get_full_id() for x in all_res if x.get_resname() == 'ASN']
print(ASNs)

# get atomic coords from neighboring atoms

# first find the neighboring atoms
ca_coord = res['CA'].get_coord()
    
model_tag = res.get_full_id()[1]
model = struct[model_tag]
atom_list = pdb.Selection.unfold_entities(model,'A')
ns = pdb.NeighborSearch(atom_list)
neighbor_atoms = ns.search(ca_coord,10.)
print(neighbor_atoms)
print(res)

neighbor_atoms = [x for x in neighbor_atoms if (x.get_name() == 'CA'
                                                and x.get_parent() != res)]
print(neighbor_atoms)

t = [x.get_parent() for x in neighbor_atoms if np.sum(np.abs(x.get_coord() - ca_coord))==0 ] 
if len(t) > 0:
    print('t = ' + str(t))
    print(res)

print(res in neighbor_atoms)
# get atomic coords from neighboring atoms


