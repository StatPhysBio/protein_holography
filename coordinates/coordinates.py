#
# File to take coordinates from list of neighborhoods
#



import COA_ref_frame as COA_calc
import numpy as np
import protein
from geo import cartesian_to_spherical
import Bio.PDB as pdb
import sys
sys.path.append('/gscratch/stf/mpun/software/freesasa-python')
import freesasa

# six channels for C,N,O,S,HOH,SASA
EL_CHANNEL_NUM = 6


def atomic_spherical_coord(atom, origin, COA=False, axes=None):
    """
    Get spherical coordinates of an atom with respect to an origin

    This function takes an pdb.Atom type and returns the spherical coordinates of 
    the atom with respect to the origin.

    Parameters:
      atom (pdb.Atom): Atom to take coordinates of.
      origin (array): Cartesian coordinates of the origin
      COA (bool): if true use axes to determine xyz axes
      axes (array): xyz vectors of desired coordinate system in the pdb 
         defined reference frame
    
    Returns:
      An array of r, theta, and phi coordinates. 
    """
    
    # get cartesian coords of current atom in the
    # pdb chosen coordinate system
    curr_r = atom.get_coord() - origin
    if COA:
        curr_r = np.einsum('ij,j->i',axes,curr_r)

    # check to make sure the atom we're looking at is not 
    # located at the origin
    if np.sum(np.abs(curr_r)) == 0:
        print(atom.get_parent().get_full_id())
        print('Error: Atom lies at the origin. Coordinates will cause singularity '
              'for hologram projections')
        return (None,None,None)
    # convert cartesian coords to spherical
    r_mag,theta,phi = cartesian_to_spherical(curr_r)
    
    return r_mag,theta,phi

# get atomic coordinates of a residue and return them in format of r,t,p each of which
# is 4 x N_e where N_e is the number of times that the element e appears in the residue
def get_coords(atom_list,origin,ch_func,ch_num,ch_dict,COA=False,res=None):
    """
    Gets spherical coordinates of a list of atoms

    This function takes a list of pdb.Atom types and outputs a tuple of arrays
    of coordinates. The output is of shape 4xN_c where N_c is the number of atoms
    associated with the given channel

    Parameters:
      atom_list (list): list of pdb.Atom types for which to gather the coordinates
      origin (array): coordinates for the coordinate system origin
      ch_func (function): function to determine the channel of a given coordinate
      ch_num (int): total numer of channels
      COA (bool): if true xyz axes are aligned with the COA bonds
      res : unsure of this parameter's use
    Returns:

    """

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
        
        atom_channels = ch_func(atom)
        r_mag,curr_t,curr_p = atomic_spherical_coord(atom,origin,COA,axes)
        if r_mag == None:
            continue

        for curr_ch in atom_channels:
            if curr_ch not in ch_to_ind.keys():
                continue
            ch_ind = ch_to_ind[curr_ch]
            if curr_ch not in ch_keys:
                continue
            

            r[ch_ind].append(r_mag)
            t[ch_ind].append(curr_t)
            p[ch_ind].append(curr_p)

    return r,t,p

def el_channel(atom):
    """
    Get channel associated with a given atom

    Each atom should contribute to the channel associated with its element.
    Furthermore water Oxygens belong to their own channel and all atoms except water
    atoms contribute to the SASA channel.

    Parameters:
      atom (pdb.Atom): Atom to determine the channel type for

    Returns:
      List of channels associated with that coordinate
    """

    channels = []
    #channels.append(atom.element)
    if atom.get_parent().get_resname() == 'HOH':
        channels.append('HOH')
    else:
        channels.append(atom.element)
        channels.append('SASA')

    return channels



# get atomic coordinates of a residue and return them in format of r,t,p each of which
# is 4 x N_e where N_e is the number of times that the element e appears in the residue
def get_res_atomic_coords(res,COA=False,ignore_alpha=True):
    """
    Gets spherical coordinates of atoms associated with a residue

    This function get spherical coordinates of a residue and return them in format of 
    r,t,p each of which is 4 x N_e where N_e is the number of times that the element
    e appears in the residue

    Parameters:
      res (pdb.Residue): residue to get coordinates for
      COA (bool): if true then xyz axes are aligned with COA bonds
      ignore_alpha (bool): if true then alpha Carbon is ignored (generally used if  
        coordinates will be used in fourier projection to avoid singularity at zero)

    Returns:
    
    """

    
    ca_coord = res['CA'].get_coord()
    if ignore_alpha:
        atom_list = [x for x in res if x.get_name() != 'CA']
    else:
        atom_list = [x for x in res]
    return get_coords(atom_list,ca_coord,el_channel,EL_CHANNEL_NUM,protein.el_to_ind,COA=COA,res=res)



def get_res_neighbor_atomic_coords(res,d,struct,remove_center=True,COA=False):
    """
    Gets spherical coordinates of atoms within a neighborhood

    This function returns the spherical coordinates of the atoms of the amino 
    acids that at least partially lie within the distance d to the residues alpha Carbon

    Parameters:
      res (pdb.Residue): central residue
      d (float): radius of the neighborhood
      struct (pdb.Structure): pdb structure that the central residue belongs to
      remove_center (bool): if true central residue coordinates will not be 
            included in returned coordinates
      COA (bool): if true the xyz axes will be given by the COA bonds
    
    Returns:
      coords (list): coordinates in the shape (d,C,N_c) where N_c is the number of points associated
          with channel c
      SASA_weights (list): a list of the SASAs at each of the protein atoms
    """

    # get central coord of the nieghborhood
    ca_coord = res['CA'].get_coord()

    # set up neighborsearch to search the same model
    # within the pdb file
    model_tag = res.get_full_id()[1]
    model = struct[model_tag]
    atom_list = pdb.Selection.unfold_entities(model,'A')
    non_hetero_non_hydrogen_atoms = [x for x in atom_list if
                        x.get_parent().get_full_id()[3][0] == ' '
                        and x.element != 'H'
    ]
    ns = pdb.NeighborSearch(atom_list)

    # perform neighborsearch on atoms within the radius d
    neighbor_atoms = ns.search(ca_coord,d)
    if remove_center:
        # remove atoms associated belonging to the central residue
        neighbor_atoms = [x for x in neighbor_atoms
                          if x.get_parent() != res
                          and x.element != 'H'
        ]
        # remove atoms beonging to same chain--for peptide classification
        #neighbor_atoms = [x for x in neighbor_atoms
        #                  if x.get_full_id()[2] != res['CA'].get_full_id()[2]]
    #print('Number of neihgbor atoms is {}'.format(len(neighbor_atoms)))
    if len(neighbor_atoms) == 0:
        print(res.get_full_id())
    SASA_neighbors = [x for x in neighbor_atoms
                      if x.get_parent().get_full_id()[3][0] == ' '
                      and x.element != 'H']
    SASA_neighbor_residues = [x.get_parent().get_resname() for x in SASA_neighbors]
    #print(SASA_neighbor_residues)
    SASA_neighbor_sns = [non_hetero_non_hydrogen_atoms.index(x) for x in SASA_neighbors]
    
    #calculate SASA
    #print(struct)
    freesasa.setVerbosity(freesasa.silent)
    freesasa_result = freesasa.calcBioPDB(struct,
                                          options = {'hetatm': False,
                                                     'hydrogen': False,
                                                     'join-models': False,
                                                     'skip-unknown': False,
                                                     'halt-at-unknown': False})[0]
    SASA_weights = [freesasa_result.atomArea(x-1) for x in SASA_neighbor_sns]
    SASA_weights = []
    for i in SASA_neighbor_sns:
        try:
            SASA_weights.append(freesasa_result.atomArea(i-1))
        except Exception as e:
            print(e)
            print(i)
            print(np.array(SASA_neighbors)[np.where(np.array(SASA_neighbor_sns) == i)])
            return 1
                                
    # get atomic coords from neighboring atoms
    return get_coords(neighbor_atoms,ca_coord,el_channel,EL_CHANNEL_NUM,protein.ch_to_ind_encoding,COA=COA),SASA_weights
