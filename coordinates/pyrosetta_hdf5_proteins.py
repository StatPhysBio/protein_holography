import sys
sys.path.append('/gscratch/stf/mpun/software/PyRosetta4.Release.python38.linux.release-299')
import pyrosetta
import numpy as np
from pyrosetta.toolbox.extract_coords_pose import pose_coords_as_rows
import h5py

def get_pose_residue_number(
    pose,
    chain,
    resnum,
    icode=' ',
):
    return pose.pdb_info().pdb2pose(chain, resnum, icode)

def get_pdb_residue_info(
    pose,
    resnum,
):
    pi = pose.pdb_info()
    return (pi.chain(resnum), pi.number(resnum), pi.icode(resnum))

def calculate_sasa(
    pose,
    probe_radius=1.4
):
    
    
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

def get_structural_info(
    pose
):
    
    # lists for each type of information to obtain
    atom_names = []
    elements = []
    sasas = []
    coords = []
    charges = []
    res_ids = []
    
    k = 0
    
    # extract coords and sasa from pyRosetta
    atom_sasa = calculate_sasa(pose)
    coords_rows = pose_coords_as_rows(pose)

    pi = pose.pdb_info()
    pdb = pi.name().split('.')[0][-4:].encode()
    
    # get structural info from each residue in the protein
    for i in range(1,pose.size()+1):
        for j in range(1,len(pose.residue(i).atoms())+1):
            atom_name = pose.residue_type(i).atom_name(j)
            idx = pose.residue(i).atom_index(atom_name)
            atom_id = (pyrosetta.rosetta.core.id.AtomID(idx,i))
            element = pose.residue_type(i).element(j).name
            aa = pose.sequence()[i-1]
            chain = pi.chain(i)
            resnum = str(pi.number(i)).encode()
            icode = pi.icode(i).encode()
            sasa = atom_sasa.get(atom_id)
            curr_coords = coords_rows[k]
            charge = pose.residue_type(i).atom_charge(j)
            res_id = np.array(
                [aa,
                 pdb,
                 chain,
                 resnum,
                 icode
                ],
                dtype='S5'
            )
                
            
            
            atom_names.append(atom_name)
            elements.append(element)
            res_ids.append(res_id)
            coords.append(curr_coords)
            sasas.append(sasa)
            charges.append(charge)
            
            k += 1

    return pdb,atom_names,elements,res_ids,coords,sasas,charges

def pad_structural_info(
    pdb,
    atom_names,
    elements,
    res_ids,
    coords,
    sasas,
    charges,
    padded_length=100
):
     
    mat_atom_names = np.empty([padded_length], dtype="S4")
    mat_atom_names.fill(b'')
    mat_elements = np.empty([padded_length], dtype="S1")
    mat_elements.fill(b'')
    mat_res_ids = np.empty([padded_length,5], dtype="S5")
    mat_res_ids.fill(b'')
    mat_coords = np.zeros([padded_length,3], dtype=float)
    mat_charges = np.zeros([padded_length], dtype=float)
    mat_sasas = np.zeros([padded_length], dtype=float)
    

    mat_atom_names[:len(atom_names)] = np.array(atom_names,dtype='S4')
    mat_elements[:len(elements)] = np.array(elements,dtype='S1')
    mat_charges[:len(charges)] = np.array(charges)
    mat_coords[:len(coords),:] = np.array(coords)
    mat_res_ids[:len(res_ids)] = np.array(res_ids,dtype='S5')
    mat_sasas[:len(sasas)] = np.array(sasas)

    return pdb,mat_atom_names,mat_elements,mat_res_ids,mat_coords,mat_charges,mat_sasas
