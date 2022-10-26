"""Module for extracting structural info from pyrosetta pose"""

from functools import partial
import sys
from typing import Tuple
sys.path.append(
    '/gscratch/stf/mpun/software/PyRosetta4.Release.python38.linux.release-299')

import h5py
import numpy as np
import pyrosetta
from pyrosetta.toolbox.extract_coords_pose import pose_coords_as_rows
from pyrosetta.rosetta.core.pose import Pose

def get_pose_residue_number(
    pose: Pose, 
    chain: str, 
    resnum: int, 
    icode: str=' '
) -> int:
    """Translate pdb residue id to pyrosetta index"""
    return pose.pdb_info().pdb2pose(chain, resnum, icode)

def get_pdb_residue_info(
    pose : Pose, 
    resnum : int
) -> Tuple[str, int, str]:
    """Translate pyrosetta index to pdb residue id"""
    pi = pose.pdb_info()
    return (pi.chain(resnum), pi.number(resnum), pi.icode(resnum))

def calculate_sasa(
    pose : Pose,
    probe_radius : float=1.4
):
    """Calculate SASA for a pose"""
    # structures for returning of sasa information
    all_atoms = pyrosetta.rosetta.core.id.AtomID_Map_bool_t()
    atom_sasa = pyrosetta.rosetta.core.id.AtomID_Map_double_t()
    rsd_sasa = pyrosetta.rosetta.utility.vector1_double()
    
    pyrosetta.rosetta.core.scoring.calc_per_atom_sasa(
        pose,
        atom_sasa,
        rsd_sasa,
        probe_radius
    )
    
    return atom_sasa

def get_hb_counts(
    hbond_set,
    i
):
    """
    Classifies a pose's h-bonds by main- and side-chain linkages
    
    Parameters
    ----------
    hbond_set : 
        The h-bond object from pyrosetta
    i : int
       
    Returns
    -------
    np.ndarray
        Float array of shape [8] where each entry is the number of
        h-bonds where the central residue and the partner are categorized 
        according to the donor/accceptor role and the backbone (bb) vs. 
        side-chain (sc) location of the bond. Specifically, the array is 
        organized as
            central|partner
            ---------------
            acc-bb  don-bb
            don-bb  acc-bb
            acc-bb  don-sc
            don-bb  acc-sc
            acc-sc  don-bb
            don-sc  acc-bb
            acc-sc  don-sc
            don-sc  acc-sc
    """
    counts = np.zeros(8,dtype=int)
    for hb in hbond_set.residue_hbonds(i):
        ctrl_don = hb.don_res() == i
        if ctrl_don:
            ctrl_side = not hb.don_hatm_is_backbone()
            nb_side = not hb.acc_atm_is_backbone()
        else:
            ctrl_side = not hb.acc_atm_is_backbone()
            nb_side = not hb.don_hatm_is_backbone()
        counts[4*ctrl_side + 2*nb_side + 1*ctrl_don] += 1
    return counts

def get_structural_info(pose : Pose) -> 
Tuple[str,
      Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Extract structural information from pyrosetta pose
    
    Parameters
    ----------
    pose : pyrosetta.rosetta.core.pose.Pose
        The pose created for the protein of interest
      
    Returns
    -------
    nested tuple of (bytes, (np.ndarray, np.ndarray, np.ndarray, np.ndarray,
      np.ndarray,np.ndarray)
        This nested tuple contains the pdb name followed by arrays containing
        the atom names, elements, residue ids, coordinates, SASAs, and charges 
        for each atom in the protein.
    """

    # lists for each type of information to obtain
    atom_names = []
    elements = []
    sasas = []
    coords = []
    charges = []
    res_ids = []
    
    k = 0
    
    # extract secondary structure for use in res ids
    DSSP = pyrosetta.rosetta.protocols.moves.DsspMover()
    DSSP.apply(pose)
      
    # extract physico-chemical information
    atom_sasa = calculate_sasa(pose)
    coords_rows = pose_coords_as_rows(pose)
    
    pi = pose.pdb_info()
    pdb = pi.name().split('.')[0][-4:].encode()
    # get structural info from each residue in the protein
    for i in range(1,pose.size()+1):
        
        # these data will form the residue id
        aa = pose.sequence()[i-1]
        chain = pi.chain(i)
        resnum = str(pi.number(i)).encode()
        icode = pi.icode(i).encode()
        ss = pose.secstruct(i)
        
        ## optional info to include in residue ids if analysis merits it
        ## - hbond info
        ## - chi1 angle
        #hbond_set = pose.get_hbonds()
        #chi1 = b''
        #print(aa)
        #if aa not in ['G','A','Z']:
        #    try:
        #        chi1 = str(pose.chi(1,i)).encode()
        #    except:
        #        print(pdb,aa,chain,resnum)
        #        #print(chi1)
        for j in range(1,len(pose.residue(i).atoms())+1):

            atom_name = pose.residue_type(i).atom_name(j)
            idx = pose.residue(i).atom_index(atom_name)
            atom_id = (pyrosetta.rosetta.core.id.AtomID(idx,i))
            element = pose.residue_type(i).element(j).name
            sasa = atom_sasa.get(atom_id)
            curr_coords = coords_rows[k]
            charge = pose.residue_type(i).atom_charge(j)
            #hb_counts = get_hb_counts(hbond_set,i)
            #if len(atom_name) < 4:
                #print('During gathering prcoess')
                #print(pdb)
                #print(atom_name)
            
            res_id =np.array([
                aa,
                pdb,
                chain,
                resnum,
                icode,
                ss,
                #*hb_counts,
                #chi1
            ],dtype='S5')
            
            atom_names.append(atom_name)
            elements.append(element)
            res_ids.append(res_id)
            coords.append(curr_coords)
            sasas.append(sasa)
            charges.append(charge)
            
            k += 1

    #if len(atom_names[0]) < 4:
    #    print('Pre-array-making')
    #    print(pdb)
    #   print(atom_names)
            
    atom_names = np.array(atom_names,dtype='|S4')
    if len(atom_names[0]) < 4:
        print('Post-array-making')
        print(pdb)
        print(atom_names)
        print(atom_names.dtype)
    elements = np.array(elements,dtype='S1')
    sasas = np.array(sasas)
    coords = np.array(coords)
    charges = np.array(charges)
    res_ids = np.array(res_ids)
    
    return pdb,(atom_names,elements,res_ids,coords,sasas,charges)

# given a matrix, pad it with empty array
def pad(arr,padded_length=100):
    #print('array = ',arr)
    # get dtype of input array
    dt = arr.dtype

    # shape of sub arrays and first dimension (to be padded)
    shape = arr.shape[1:]
    orig_length = arr.shape[0]

    # check that the padding is large enough to accomdate the data
    if padded_length < orig_length:
        print('Error: Padded length of {}'.format(padded_length),
              'is smaller than original length of array {}'.format(orig_length))

    # create padded array
    padded_shape = (padded_length,*shape)
    mat_arr = np.zeros(padded_shape, dtype=dt)
    
    # if type is string fill array with empty strings
    #if np.issubdtype(bytes, dt):
    #    mat_arr.fill(b'')

    # add data to padded array
    mat_arr[:orig_length] = np.array(arr)
    
    return mat_arr

def pad_structural_info(
    ragged_structure,
    padded_length=100
):
    
    
    pad_custom = partial(pad,padded_length=padded_length)
    
    mat_structure = list(map(pad_custom,ragged_structure))

    return mat_structure
