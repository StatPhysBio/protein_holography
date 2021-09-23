#
# File to take coordinates from list of neighborhoods
#

import COA_ref_frame as COA_calc
import numpy as np
import protein
from geo import cartesian_to_spherical
import Bio.PDB as pdb

# get spherical coordinates of an atom given an origin
def atomic_spherical_coord(atom, origin, COA=False, axes=None):
        
    # get cartesian coords of current atom in the
    # pdb chosen coord-system
    curr_r = atom.get_coord() - origin
    if COA:
        curr_r = np.einsum('ij,j->i',axes,curr_r)

    # check to make sure the atom we're looking at is not 
    # located at the origin
    if np.sum(np.abs(curr_r)) == 0:
        print(atom.get_parent().get_full_id())
        print('Error: Atom lies at the origin. Coordinates will vause singularity for hologram projections')
        return (None,None,None)
    # convert cartesian coords to spherical
    r_mag,theta,phi = cartesian_to_spherical(curr_r)
    
    return r_mag,theta,phi

# get atomic coordinates of a residue and return them in format of r,t,p each of which
# is 4 x N_e where N_e is the number of times that the element e appears in the residue
def get_coords(atom_list,origin,ch_func,ch_num,ch_dict,COA=False,res=None):
    
    # set up hologram structure
    r = [[] for i in range(ch_num)]
    t = [[] for i in range(ch_num)]
    p = [[] for i in range(ch_num)]
    ch_to_ind = ch_dict
    ch_keys = ch_dict.keys()

    axes=None
    if COA:
        axes = COA_calc.get_COA_axes(res)
    
    for atom in atom_list:
        
        curr_ch = ch_func(atom)

        if curr_ch not in ch_to_ind.keys():
            continue
        ch_ind = ch_to_ind[curr_ch]
        if curr_ch not in ch_keys:
            continue
            
        r_mag,curr_t,curr_p = atomic_spherical_coord(atom,origin,COA,axes)
        if r_mag == None:
            continue
        # append spherical coords of current atom to
        # the overall lists
        r[ch_ind].append(r_mag)
        t[ch_ind].append(curr_t)
        p[ch_ind].append(curr_p)

    return r,t,p

def el_channel(atom):
    return atom.element

EL_CHANNEL_NUM = 4

# get atomic coordinates of a residue and return them in format of r,t,p each of which
# is 4 x N_e where N_e is the number of times that the element e appears in the residue
def get_res_atomic_coords(res,COA=False,ignore_alpha=True):

    ca_coord = res['CA'].get_coord()
    if ignore_alpha:
        atom_list = [x for x in res if x.get_name() != 'CA']
    else:
        atom_list = [x for x in res]
    return get_coords(atom_list,ca_coord,el_channel,EL_CHANNEL_NUM,protein.el_to_ind,COA=COA,res=res)



# returns the spherical coordinates of the atoms of the amino acids that at
# least partially lie within the distance d to the residues alpha Carbon
def get_res_neighbor_atomic_coords(res,d,struct,remove_center=True,COA=False):

    # first find the neighboring atoms
    ca_coord = res['CA'].get_coord()
    
    model_tag = res.get_full_id()[1]
    model = struct[model_tag]
    atom_list = pdb.Selection.unfold_entities(model,'A')
    ns = pdb.NeighborSearch(atom_list)
    neighbor_atoms = ns.search(ca_coord,d)
    if remove_center:
        neighbor_atoms = [x for x in neighbor_atoms if x.get_parent() != res]
    # get atomic coords from neighboring atoms
    return get_coords(neighbor_atoms,ca_coord,el_channel,EL_CHANNEL_NUM,protein.el_to_ind,COA=COA)
