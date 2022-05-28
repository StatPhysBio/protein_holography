#SphericalCoordCalculation.py
# interface module for accessing pdb files
#

import imp
import os

import Bio.PDB as pdb
import Bio.PDB.mmtf
import numpy as np

import protein_holography.coordinates.COA_ref_frame as COA
from protein_holography.coordinates.geo import cartesian_to_spherical
import protein_holography.hologram.hologram as hgm
import protein_holography.hologram.sample_protein as sample_protein
import protein_holography.utils.protein as protein

imp.reload(sample_protein)

DIMENSIONS = 3
EL_CHANNEL_NUM = len(protein.el_to_ind.keys())
AA_CHANNEL_NUM = len(protein.aa_to_ind.keys())

# check validity of a given residue
def check_CA_in_res(res):
    if 'CA' not in res:
        return False
    else:
        return True

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
        axes = COA.get_COA_axes(res)
    
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


def aa_channel(atom):
    return atom.get_parent().resname


# get atomic coordinates of a residue and return them in format of r,t,p each of which
# is 4 x N_e where N_e is the number of times that the element e appears in the residue
def get_res_atomic_coords(res,COA=False):

    ca_coord = res['CA'].get_coord()
    atom_list = [x for x in res if x.get_name() != 'CA']
    return get_coords(atom_list,ca_coord,el_channel,EL_CHANNEL_NUM,protein.el_to_ind,COA=COA,res=res)


# returns the spherical coordinates of the atoms of the amino acids that at
# least partially lie within the distance d to the residues alpha Carbon
def get_res_neighbor_atomic_coords(res,d,struct):

    # first find the neighboring atoms
    ca_coord = res['CA'].get_coord()
    
    model_tag = res.get_full_id()[1]
    model = struct[model_tag]
    atom_list = pdb.Selection.unfold_entities(model,'A')
    ns = pdb.NeighborSearch(atom_list)
    neighbor_atoms = ns.search(ca_coord,d)
    neighbor_atoms = [x for x in neighbor_atoms if x.get_parent() != res]
    # get atomic coords from neighboring atoms
    return get_coords(neighbor_atoms,ca_coord,el_channel,EL_CHANNEL_NUM,protein.el_to_ind)


# returns the spherical coordinates of the atoms of the amino acids that at
# least partially lie within the distance d to the residues alpha Carbon
def get_res_neighbor_aa_coords(res,d,struct):

    # first find the neighboring atoms
    ca_coord = res['CA'].get_coord()
    
    model_tag = res.get_full_id()[1]
    model = struct[model_tag]
    atom_list = pdb.Selection.unfold_entities(model,'A')
    ns = pdb.NeighborSearch(atom_list)
    neighbor_atoms = ns.search(ca_coord,d)


    neighbor_atoms = [x for x in neighbor_atoms if (x.get_name() == 'CA'
                                                    and x.get_parent() != res)]

    # get atomic coords from neighboring atoms
    return get_coords(neighbor_atoms,ca_coord,
                                    aa_channel,AA_CHANNEL_NUM,protein.aa_to_ind)



# returns the hologram coefficients of instances number of samples of each amino acid
# randomly sampled from the protein list provided
#
# note about structure of this function: we want the data organized in l x N x n_c x m
# In order to do this, we create a dictionary each time we take a sample. Simultaneously
# we create a list of these dictionaries so that our data is in the order N x l x n_c x m
# in order to reverse the first two indices, we stack the entries of each dictionary into
# one np array of length N and index this under teh appropriate l in our ultimate data
# structure
def get_amino_acid_el_shapes_from_protein_list(protein_list,
                                               protein_dir,
                                               instances,
                                               noise_distance,
                                               r_h,
                                               k,
                                               l_max,
                                               center):


    
    # list of amino acids
    amino_acids = list(protein.aa_to_ind.keys())
    # a list of the amino acids we need to take samples of
    aa_to_sample = amino_acids*instances

    num_proteins = len(protein_list)
       
    # indices to keep track of location in the amino acids to sample list
    # as well as the protein list
    protein_ind = 0
    aa_ind = 0

    hologram_coeffs = {}
    hologram_labels = []

    parser = pdb.PDBParser(QUIET=True)

    data_dicts = []
    
    while aa_ind < len(aa_to_sample):
        if(aa_ind%20 == 0):
            print(aa_ind)
        # cool to keep track of whether or not the current amino acid was found
        aa_found = False

        # get the current aa and the current protein to check for this aa
        curr_aa = aa_to_sample[aa_ind]
        curr_protein = protein_list[protein_ind%num_proteins]
        curr_struct = parser.get_structure(curr_protein + 'struct',protein_dir + '/' + curr_protein + '.pdb')
        protein_ind += 1

        # take sample residue from protein. If the given aa doesn't exist in this protein
        # then note the aa is not found and move on to the next protein
        sample_res = sample_protein.sample_amino_acid_from_protein(curr_struct,curr_protein,curr_aa)
        if sample_res == None:
            continue
        aa_found = True

        # get the atomic coords of the residue in spherical form
        # also get the neighboring atomic coords if the noise distance is greater
        # than zero
        if center:
            curr_coords = get_res_atomic_coords(sample_res)
        else:
            curr_coords = [[[] for i in range(EL_CHANNEL_NUM)] for j in range(DIMENSIONS)]
        if noise_distance > 0:
            neighbor_coords = get_res_neighbor_atomic_coords(sample_res,noise_distance,curr_struct)
            for i in range(DIMENSIONS):
                for j in range(EL_CHANNEL_NUM):
                    curr_coords[i][j] = curr_coords[i][j] + neighbor_coords[i][j]

        curr_r,curr_t,curr_p = curr_coords

        # to be used once coefficients  are gathered for all channels for current res
        curr_coeffs = {}
        # list of the dicts for each channel for current res
        channel_dicts = []
        for curr_ch in range(EL_CHANNEL_NUM):
            curr_channel_coeffs = {}
            for l in range(l_max + 1):
                # if the current channel has no signal then append zero holographic signal
                if len(curr_r[curr_ch]) == 0:
                    curr_channel_coeffs[l] = np.array([0.]*(2*l+1))
                    continue
                curr_channel_coeffs[l] = hgm.hologram_coeff_l(curr_r[curr_ch],
                                                              curr_t[curr_ch],
                                                              curr_p[curr_ch],
                                                              r_h, k, l)
            channel_dicts.append(curr_channel_coeffs)
        # coefficients gathered for every channel for this sample residue
        # channels_dicts currently has the structure n_c x l x m
        # we can now swap n_c and l
        for l in range(l_max + 1):
            curr_coeffs[l] = np.stack([x[l] for x in channel_dicts])
        # now we want to add this dictionary to a list of all the dicts for
        # each sample aa
        data_dicts.append(curr_coeffs)

        # make a label for the current hologram
        curr_label = [0]*len(amino_acids)
        curr_label[protein.aa_to_ind[curr_aa]] = 1

        # if aa has been found increment the aa_ind
        if aa_found ==True:
            hologram_labels.append(curr_label)
            aa_ind += 1

    # after the above while loop has finished we now assume that all amino acids have
    # been sampled and their data is compiled in the list data_dicts in the order
    # N x l x n_c x m. We want to swap the indices N and l so we'll do that here
    for l in range(l_max + 1):
        hologram_coeffs[l] = np.stack([x[l] for x in data_dicts])

    hologram_coeffs_real = {}
    hologram_coeffs_imag = {}
    for l in range(l_max + 1):
        hologram_coeffs_real[l] = np.real(hologram_coeffs[l])
        hologram_coeffs_imag[l] = np.imag(hologram_coeffs[l])

    return hologram_coeffs_real,hologram_coeffs_imag,hologram_labels
                                                              

# returns the hologram coefficients of instances number of samples of each amino acid
# randomly sampled from the protein list provided
#
# note about structure of this function: we want the data organized in l x N x n_c x m
# In order to do this, we create a dictionary each time we take a sample. Simultaneously
# we create a list of these dictionaries so that our data is in the order N x l x n_c x m
# in order to reverse the first two indices, we stack the entries of each dictionary into
# one np array of length N and index this under teh appropriate l in our ultimate data
# structure
def get_amino_acid_el_shapes_from_protein_list_delta(protein_list,
                                               protein_dir,
                                               instances,
                                               noise_distance,
                                               r_h,
                                               k,
                                               l_max,
                                               center):


    
    # list of amino acids
    amino_acids = list(protein.aa_to_ind.keys())
    # a list of the amino acids we need to take samples of
    aa_to_sample = amino_acids*instances

    num_proteins = len(protein_list)
       
    # indices to keep track of location in the amino acids to sample list
    # as well as the protein list
    protein_ind = 0
    aa_ind = 0

    hologram_coeffs = {}
    hologram_labels = []

    parser = pdb.PDBParser(QUIET=True)

    data_dicts = []
    
    while aa_ind < len(aa_to_sample):
        if(aa_ind%20 == 0):
            print(aa_ind)
        # cool to keep track of whether or not the current amino acid was found
        aa_found = False

        # get the current aa and the current protein to check for this aa
        curr_aa = aa_to_sample[aa_ind]
        curr_protein = protein_list[protein_ind%num_proteins]
        curr_struct = parser.get_structure(curr_protein + 'struct',protein_dir + '/' + curr_protein + '.pdb')
        protein_ind += 1

        # take sample residue from protein. If the given aa doesn't exist in this protein
        # then note the aa is not found and move on to the next protein
        sample_res = sample_protein.sample_amino_acid_from_protein(curr_struct,curr_protein,curr_aa)
        if sample_res == None:
            continue
        aa_found = True

        # get the atomic coords of the residue in spherical form
        # also get the neighboring atomic coords if the noise distance is greater
        # than zero
        if center:
            curr_coords = get_res_atomic_coords(sample_res)
        else:
            curr_coords = [[[] for i in range(EL_CHANNEL_NUM)] for j in range(DIMENSIONS)]
        if noise_distance > 0:
            neighbor_coords = get_res_neighbor_atomic_coords(sample_res,noise_distance,curr_struct)
            for i in range(DIMENSIONS):
                for j in range(EL_CHANNEL_NUM):
                    curr_coords[i][j] = curr_coords[i][j] + neighbor_coords[i][j]

        curr_r,curr_t,curr_p = curr_coords

        # to be used once coefficients  are gathered for all channels for current res
        curr_coeffs = {}
        # list of the dicts for each channel for current res
        channel_dicts = []
        for curr_ch in range(EL_CHANNEL_NUM):
            curr_channel_coeffs = {}
            for l in range(l_max + 1):
                # if the current channel has no signal then append zero holographic signal
                if len(curr_r[curr_ch]) == 0:
                    curr_channel_coeffs[l] = np.array([0.]*(2*l+1))
                    continue
                curr_channel_coeffs[l] = hgm.delta(curr_r[curr_ch],
                                                   curr_t[curr_ch],
                                                   curr_p[curr_ch],
                                                   l)

            channel_dicts.append(curr_channel_coeffs)
        # coefficients gathered for every channel for this sample residue
        # channels_dicts currently has the structure n_c x l x m
        # we can now swap n_c and l
        for l in range(l_max + 1):
            curr_coeffs[l] = np.stack([x[l] for x in channel_dicts])
        # now we want to add this dictionary to a list of all the dicts for
        # each sample aa
        data_dicts.append(curr_coeffs)

        # make a label for the current hologram
        curr_label = [0]*len(amino_acids)
        curr_label[protein.aa_to_ind[curr_aa]] = 1

        # if aa has been found increment the aa_ind
        if aa_found ==True:
            hologram_labels.append(curr_label)
            aa_ind += 1

    # after the above while loop has finished we now assume that all amino acids have
    # been sampled and their data is compiled in the list data_dicts in the order
    # N x l x n_c x m. We want to swap the indices N and l so we'll do that here
    for l in range(l_max + 1):
        hologram_coeffs[l] = np.stack([x[l] for x in data_dicts])

    hologram_coeffs_real = {}
    hologram_coeffs_imag = {}
    for l in range(l_max + 1):
        hologram_coeffs_real[l] = np.real(hologram_coeffs[l])
        hologram_coeffs_imag[l] = np.imag(hologram_coeffs[l])

    return hologram_coeffs_real,hologram_coeffs_imag,hologram_labels
        

# returns the hologram coefficients of instances number of samples of each amino acid
# randomly sampled from the protein list provided
#
# note about structure of this function: we want the data organized in l x N x n_c x m
# In order to do this, we create a dictionary each time we take a sample. Simultaneously
# we create a list of these dictionaries so that our data is in the order N x l x n_c x m
# in order to reverse the first two indices, we stack the entries of each dictionary into
# one np array of length N and index this under teh appropriate l in our ultimate data
# structure
def get_amino_acid_el_shapes_from_protein_list_ks(protein_list,
                                               protein_dir,
                                               instances,
                                               noise_distance,
                                               r_h,
                                               ks,
                                               l_max,
                                               center):


    
    # list of amino acids
    amino_acids = list(protein.aa_to_ind.keys())
    # a list of the amino acids we need to take samples of
    aa_to_sample = amino_acids*instances

    num_proteins = len(protein_list)
       
    # indices to keep track of location in the amino acids to sample list
    # as well as the protein list
    protein_ind = 0
    aa_ind = 0

    hologram_coeffs = {}
    hologram_labels = []

    parser = pdb.PDBParser(QUIET=True)

    data_dicts = []
    
    while aa_ind < len(aa_to_sample):
        if(aa_ind%20 == 0):
            print(aa_ind)
        # cool to keep track of whether or not the current amino acid was found
        aa_found = False

        # get the current aa and the current protein to check for this aa
        curr_aa = aa_to_sample[aa_ind]
        curr_protein = protein_list[protein_ind%num_proteins]
        curr_struct = parser.get_structure(curr_protein + 'struct',protein_dir + '/' + curr_protein + '.pdb')
        protein_ind += 1

        # take sample residue from protein. If the given aa doesn't exist in this protein
        # then note the aa is not found and move on to the next protein
        sample_res = sample_protein.sample_amino_acid_from_protein(curr_struct,curr_protein,curr_aa)
        if sample_res == None:
            continue
        aa_found = True

        # get the atomic coords of the residue in spherical form
        # also get the neighboring atomic coords if the noise distance is greater
        # than zero
        if center:
            curr_coords = get_res_atomic_coords(sample_res)
        else:
            curr_coords = [[[] for i in range(EL_CHANNEL_NUM)] for j in range(DIMENSIONS)]
        if noise_distance > 0:
            neighbor_coords = get_res_neighbor_atomic_coords(sample_res,noise_distance,curr_struct)
            for i in range(DIMENSIONS):
                for j in range(EL_CHANNEL_NUM):
                    curr_coords[i][j] = curr_coords[i][j] + neighbor_coords[i][j]

        curr_r,curr_t,curr_p = curr_coords

        # to be used once coefficients  are gathered for all channels for current res
        curr_coeffs = {}
        # list of the dicts for each channel for current res
        channel_dicts = []
        for k in ks:
            for curr_ch in range(EL_CHANNEL_NUM):
                curr_channel_coeffs = {}
                for l in range(l_max + 1):
                    # if the current channel has no signal then append zero holographic signal
                    if len(curr_r[curr_ch]) == 0:
                        curr_channel_coeffs[l] = np.array([0.]*(2*l+1))
                        continue
                    curr_channel_coeffs[l] = hgm.hologram_coeff_l(curr_r[curr_ch],
                                                                  curr_t[curr_ch],
                                                                  curr_p[curr_ch],
                                                                  r_h, k, l)
                channel_dicts.append(curr_channel_coeffs)
        # coefficients gathered for every channel for this sample residue
        # channels_dicts currently has the structure n_c x l x m
        # we can now swap n_c and l
        for l in range(l_max + 1):
            curr_coeffs[l] = np.stack([x[l] for x in channel_dicts])
        # now we want to add this dictionary to a list of all the dicts for
        # each sample aa
        data_dicts.append(curr_coeffs)

        # make a label for the current hologram
        curr_label = [0]*len(amino_acids)
        curr_label[protein.aa_to_ind[curr_aa]] = 1

        # if aa has been found increment the aa_ind
        if aa_found ==True:
            hologram_labels.append(curr_label)
            aa_ind += 1

    # after the above while loop has finished we now assume that all amino acids have
    # been sampled and their data is compiled in the list data_dicts in the order
    # N x l x n_c x m. We want to swap the indices N and l so we'll do that here
    for l in range(l_max + 1):
        hologram_coeffs[l] = np.stack([x[l] for x in data_dicts])

    hologram_coeffs_real = {}
    hologram_coeffs_imag = {}
    for l in range(l_max + 1):
        hologram_coeffs_real[l] = np.real(hologram_coeffs[l])
        hologram_coeffs_imag[l] = np.imag(hologram_coeffs[l])
    print(hologram_coeffs_real[0].shape)
    return hologram_coeffs_real,hologram_coeffs_imag,hologram_labels
    
    
    
    
    

# returns the hologram coefficients of instances number of samples of each amino acid
# randomly sampled from the protein list provided
#
# note about structure of this function: we want the data organized in l x N x n_c x m
# In order to do this, we create a dictionary each time we take a sample. Simultaneously
# we create a list of these dictionaries so that our data is in the order N x l x n_c x m
# in order to reverse the first two indices, we stack the entries of each dictionary into
# one np array of length N and index this under teh appropriate l in our ultimate data
# structure
def get_amino_acid_aa_shapes_from_protein_list(protein_list,
                                            protein_dir,
                                            instances,
                                            noise_distance,
                                            r_h,
                                            k,
                                            l_max):

    
    # list of amino acids
    amino_acids = list(protein.aa_to_ind.keys())
    # a list of the amino acids we need to take samples of
    aa_to_sample = amino_acids*instances

    num_proteins = len(protein_list)
       
    # indices to keep track of location in the amino acids to sample list
    # as well as the protein list
    protein_ind = 0
    aa_ind = 0

    hologram_coeffs = {}
    hologram_labels = []

    parser = pdb.PDBParser(QUIET=True)

    data_dicts = []
    
    while aa_ind < len(aa_to_sample):
        if(aa_ind%20 == 0):
            print(aa_ind)
        # cool to keep track of whether or not the current amino acid was found
        aa_found = False

        # get the current aa and the current protein to check for this aa
        curr_aa = aa_to_sample[aa_ind]
        curr_protein = protein_list[protein_ind%num_proteins]

        curr_struct = parser.get_structure(curr_protein + 'struct',protein_dir + '/' + curr_protein + '.pdb')
        protein_ind += 1

        # take sample residue from protein. If the given aa doesn't exist in this protein
        # then note the aa is not found and move on to the next protein
        sample_res = sample_protein.sample_amino_acid_from_protein(curr_struct,curr_protein,curr_aa)
        if sample_res == None:
            continue
        aa_found = True

        # get the atomic coords of the residue in spherical form
        # also get the neighboring atomic coords if the noise distance is greater
        # than zero

        r = [[] for i in range(AA_CHANNEL_NUM)]
        t = [[] for i in range(AA_CHANNEL_NUM)]
        p = [[] for i in range(AA_CHANNEL_NUM)]

        curr_coords = [r,t,p]

        if noise_distance > 0:
            neighbor_coords = get_res_neighbor_aa_coords(sample_res,noise_distance,curr_struct)
            for i in range(DIMENSIONS):
                for j in range(AA_CHANNEL_NUM):
                    curr_coords[i][j] = curr_coords[i][j] + neighbor_coords[i][j]

        curr_r,curr_t,curr_p = curr_coords

        # to be used once coefficients  are gathered for all channels for current res
        curr_coeffs = {}
        # list of the dicts for each channel for current res
        channel_dicts = []
        for curr_ch in range(AA_CHANNEL_NUM):
            curr_channel_coeffs = {}
            for l in range(l_max + 1):
                # if the current channel has no signal then append zero holographic signal
                if len(curr_r[curr_ch]) == 0:
                    curr_channel_coeffs[l] = np.array([0.]*(2*l+1))
                    continue
                curr_channel_coeffs[l] = hgm.hologram_coeff_l(curr_r[curr_ch],
                                                              curr_t[curr_ch],
                                                              curr_p[curr_ch],
                                                              r_h, k, l)
            channel_dicts.append(curr_channel_coeffs)
        # coefficients gathered for every channel for this sample residue
        # channels_dicts currently has the structure n_c x l x m
        # we can now swap n_c and l
        for l in range(l_max + 1):
            curr_coeffs[l] = np.stack([x[l] for x in channel_dicts])
        # now we want to add this dictionary to a list of all the dicts for
        # each sample aa
        data_dicts.append(curr_coeffs)

        # make a label for the current hologram
        curr_label = [0]*len(amino_acids)
        curr_label[protein.aa_to_ind[curr_aa]] = 1

        # if aa has been found increment the aa_ind
        if aa_found ==True:
            hologram_labels.append(curr_label)
            aa_ind += 1

    # after the above while loop has finished we now assume that all amino acids have
    # been sampled and their data is compiled in the list data_dicts in the order
    # N x l x n_c x m. We want to swap the indices N and l so we'll do that here
    for l in range(l_max + 1):
        hologram_coeffs[l] = np.stack([x[l] for x in data_dicts])

    hologram_coeffs_real = {}
    hologram_coeffs_imag = {}
    for l in range(l_max + 1):
        hologram_coeffs_real[l] = np.real(hologram_coeffs[l])
        hologram_coeffs_imag[l] = np.imag(hologram_coeffs[l])

    return hologram_coeffs_real,hologram_coeffs_imag,hologram_labels
                                                              
        
    

                                                       


from Bio.PDB.DSSP import DSSP
        
    
# returns the hologram coefficients of instances number of samples of each amino acid
# randomly sampled from the protein list provided
#
# note about structure of this function: we want the data organized in l x N x n_c x m
# In order to do this, we create a dictionary each time we take a sample. Simultaneously
# we create a list of these dictionaries so that our data is in the order N x l x n_c x m
# in order to reverse the first two indices, we stack the entries of each dictionary into
# one np array of length N and index this under teh appropriate l in our ultimate data
# structure
def get_amino_acid_sample_from_protein_list(protein_list,
                                            protein_dir,
                                            instances):


    
    # list of amino acids
    amino_acids = list(protein.aa_to_ind.keys())
    # a list of the amino acids we need to take samples of
    aa_to_sample = amino_acids*instances

    num_proteins = len(protein_list)
       
    # indices to keep track of location in the amino acids to sample list
    # as well as the protein list
    protein_ind = 0
    aa_ind = 0
    
    parser = pdb.PDBParser(QUIET=True)

    aa_sample = []

    while aa_ind < len(aa_to_sample):
        if(aa_ind%20 == 0):
            print(aa_ind)
        # cool to keep track of whether or not the current amino acid was found
        aa_found = False

        # get the current aa and the current protein to check for this aa
        curr_aa = aa_to_sample[aa_ind]
        curr_protein = protein_list[protein_ind%num_proteins]
        curr_struct = parser.get_structure(curr_protein,protein_dir + '/' + curr_protein + '.pdb')
        protein_ind += 1

        # take sample residue from protein. If the given aa doesn't exist in this protein
        # then note the aa is not found and move on to the next protein
        sample_res = sample_protein.sample_amino_acid_from_protein(curr_struct,curr_protein,curr_aa)
        if sample_res == None:
            #print('sample res == None')
            continue
        if len([x for x in sample_res]) != protein.atoms_per_aa[curr_aa]:
            #print(len([x for x in sample_res]),protein.atoms_per_aa[curr_aa])
            continue
        aa_found = True
        resname = sample_res.get_resname()
        res_id = sample_res.get_full_id()
        resname = sample_res.get_resname()
        res_id = sample_res.get_full_id()
        try:
            dssp = DSSP(curr_struct[res_id[1]], curr_protein + '.pdb', dssp="/gscratch/stf/mpun/software/dssp-2.0.4-linux-amd64")
        except Exception as e:
            print(e)
            continue
        try:    
            res_dssp = dssp[(res_id[2],res_id[3])]
        except:
            print(res_id)
            continue
        aa_info = (resname, res_id, res_dssp)
        aa_sample.append(aa_info)
        if aa_found ==True:
            aa_ind += 1

    return aa_sample

    
# returns the hologram coefficients of instances number of samples of each amino acid
# randomly sampled from the protein list provided
#
# note about structure of this function: we want the data organized in l x N x n_c x m
# In order to do this, we create a dictionary each time we take a sample. Simultaneously
# we create a list of these dictionaries so that our data is in the order N x l x n_c x m
# in order to reverse the first two indices, we stack the entries of each dictionary into
# one np array of length N and index this under teh appropriate l in our ultimate data
# structure
def get_amino_acid_sample_from_protein_list_mmtf(protein_list,
                                            protein_dir,
                                            instances):


    
    # list of amino acids
    amino_acids = list(protein.aa_to_ind.keys())
    # a list of the amino acids we need to take samples of
    aa_to_sample = amino_acids*instances

    num_proteins = len(protein_list)
       
    # indices to keep track of location in the amino acids to sample list
    # as well as the protein list
    protein_ind = 0
    aa_ind = 0
    
    parser = pdb.mmtf.MMTFParser()

    aa_sample = []

    while aa_ind < len(aa_to_sample):
        if(aa_ind%20 == 0):
            print(aa_ind)
        # bool to keep track of whether or not the current amino acid was found
        aa_found = False

        # get the current aa and the current protein to check for this aa
        curr_aa = aa_to_sample[aa_ind]
        curr_protein = protein_list[protein_ind%num_proteins]
        curr_struct = parser.get_structure(protein_dir + '/' + curr_protein + '.mmtf')
        protein_ind += 1

        # take sample residue from protein. If the given aa doesn't exist in this protein
        # then note the aa is not found and move on to the next protein
        sample_res = sample_protein.sample_amino_acid_from_protein(curr_struct,curr_protein,curr_aa)
        if sample_res == None:
            continue
        if len([x for x in sample_res]) != protein.atoms_per_aa[curr_aa]:
            continue
        aa_found = True
        resname = sample_res.get_resname()
        res_id = sample_res.get_full_id()
        resname = sample_res.get_resname()
        res_id = sample_res.get_full_id()

        aa_info = (resname, res_id)
        aa_sample.append(aa_info)
        if aa_found ==True:
            aa_ind += 1

    return aa_sample

# returns the hologram coefficients of instances number of samples of each amino acid
# randomly sampled from the protein list provided
#
# note about structure of this function: we want the data organized in l x N x n_c x m
# In order to do this, we create a dictionary each time we take a sample. Simultaneously
# we create a list of these dictionaries so that our data is in the order N x l x n_c x m
# in order to reverse the first two indices, we stack the entries of each dictionary into
# one np array of length N and index this under teh appropriate l in our ultimate data
# structure
def get_amino_acid_el_shapes_from_protein_list_zernike(protein_list,
                                               protein_dir,
                                               instances,
                                               noise_distance,
                                               k,
                                               l_max,
                                               center):


    
    # list of amino acids
    amino_acids = list(protein.aa_to_ind.keys())
    # a list of the amino acids we need to take samples of
    aa_to_sample = amino_acids*instances

    num_proteins = len(protein_list)
       
    # indices to keep track of location in the amino acids to sample list
    # as well as the protein list
    protein_ind = 0
    aa_ind = 0

    hologram_coeffs = {}
    hologram_labels = []

    parser = pdb.PDBParser(QUIET=True)

    data_dicts = []
    
    while aa_ind < len(aa_to_sample):
        if(aa_ind%20 == 0):
            print(aa_ind)
        # cool to keep track of whether or not the current amino acid was found
        aa_found = False

        # get the current aa and the current protein to check for this aa
        curr_aa = aa_to_sample[aa_ind]
        curr_protein = protein_list[protein_ind%num_proteins]
        curr_struct = parser.get_structure(curr_protein + 'struct',protein_dir + '/' + curr_protein + '.pdb')
        protein_ind += 1

        # take sample residue from protein. If the given aa doesn't exist in this protein
        # then note the aa is not found and move on to the next protein
        sample_res = sample_protein.sample_amino_acid_from_protein(curr_struct,curr_protein,curr_aa)
        if sample_res == None:
            continue
        aa_found = True

        # get the atomic coords of the residue in spherical form
        # also get the neighboring atomic coords if the noise distance is greater
        # than zero
        if center:
            curr_coords = get_res_atomic_coords(sample_res)
        else:
            curr_coords = [[[] for i in range(EL_CHANNEL_NUM)] for j in range(DIMENSIONS)]
        if noise_distance > 0:
            neighbor_coords = get_res_neighbor_atomic_coords(sample_res,noise_distance,curr_struct)
            for i in range(DIMENSIONS):
                for j in range(EL_CHANNEL_NUM):
                    curr_coords[i][j] = curr_coords[i][j] + neighbor_coords[i][j]

        curr_r,curr_t,curr_p = curr_coords

        # to be used once coefficients  are gathered for all channels for current res
        curr_coeffs = {}
        # list of the dicts for each channel for current res
        channel_dicts = []
        for curr_ch in range(EL_CHANNEL_NUM):
            curr_channel_coeffs = {}
            for l in range(l_max + 1):
                # if the current channel has no signal then append zero holographic signal
                if len(curr_r[curr_ch]) == 0:
                    curr_channel_coeffs[l] = np.array([0.]*(2*l+1))
                    continue
                curr_channel_coeffs[l] = hgm.zernike_coeff_l(curr_r[curr_ch],
                                                             curr_t[curr_ch],
                                                             curr_p[curr_ch],
                                                             k, noise_distance+10., l)
            channel_dicts.append(curr_channel_coeffs)
        # coefficients gathered for every channel for this sample residue
        # channels_dicts currently has the structure n_c x l x m
        # we can now swap n_c and l
        for l in range(l_max + 1):
            curr_coeffs[l] = np.stack([x[l] for x in channel_dicts])
        # now we want to add this dictionary to a list of all the dicts for
        # each sample aa
        data_dicts.append(curr_coeffs)

        # make a label for the current hologram
        curr_label = [0]*len(amino_acids)
        curr_label[protein.aa_to_ind[curr_aa]] = 1

        # if aa has been found increment the aa_ind
        if aa_found ==True:
            hologram_labels.append(curr_label)
            aa_ind += 1

    # after the above while loop has finished we now assume that all amino acids have
    # been sampled and their data is compiled in the list data_dicts in the order
    # N x l x n_c x m. We want to swap the indices N and l so we'll do that here
    for l in range(l_max + 1):
        hologram_coeffs[l] = np.stack([x[l] for x in data_dicts])

    hologram_coeffs_real = {}
    hologram_coeffs_imag = {}
    for l in range(l_max + 1):
        hologram_coeffs_real[l] = np.real(hologram_coeffs[l])
        hologram_coeffs_imag[l] = np.imag(hologram_coeffs[l])

    return hologram_coeffs_real,hologram_coeffs_imag,hologram_labels
                                                              
# returns the hologram coefficients of instances number of samples of each amino acid
# randomly sampled from the protein list provided
#
# note about structure of this function: we want the data organized in l x N x n_c x m
# In order to do this, we create a dictionary each time we take a sample. Simultaneously
# we create a list of these dictionaries so that our data is in the order N x l x n_c x m
# in order to reverse the first two indices, we stack the entries of each dictionary into
# one np array of length N and index this under teh appropriate l in our ultimate data
# structure
def get_amino_acid_el_shapes_from_protein_list_zernike_ks(protein_list,
                                               protein_dir,
                                               instances,
                                               noise_distance,
                                               ks,
                                               l_max,
                                               center):


    print('Proper function called')
    # list of amino acids
    amino_acids = list(protein.aa_to_ind.keys())
    # a list of the amino acids we need to take samples of
    aa_to_sample = amino_acids*instances

    num_proteins = len(protein_list)
       
    # indices to keep track of location in the amino acids to sample list
    # as well as the protein list
    protein_ind = 0
    aa_ind = 0

    hologram_coeffs = {}
    hologram_labels = []

    parser = pdb.PDBParser(QUIET=True)

    data_dicts = []
    
    while aa_ind < len(aa_to_sample):
        if(aa_ind%20 == 0):
            print(aa_ind)
        # cool to keep track of whether or not the current amino acid was found
        aa_found = False

        # get the current aa and the current protein to check for this aa
        curr_aa = aa_to_sample[aa_ind]
        curr_protein = protein_list[protein_ind%num_proteins]
        curr_struct = parser.get_structure(curr_protein + 'struct',protein_dir + '/' + curr_protein + '.pdb')
        protein_ind += 1

        # take sample residue from protein. If the given aa doesn't exist in this protein
        # then note the aa is not found and move on to the next protein
        sample_res = sample_protein.sample_amino_acid_from_protein(curr_struct,curr_protein,curr_aa)
        if sample_res == None:
            continue
        aa_found = True

        # get the atomic coords of the residue in spherical form
        # also get the neighboring atomic coords if the noise distance is greater
        # than zero
        if center:
            curr_coords = get_res_atomic_coords(sample_res)
        else:
            curr_coords = [[[] for i in range(EL_CHANNEL_NUM)] for j in range(DIMENSIONS)]
        if noise_distance > 0:
            neighbor_coords = get_res_neighbor_atomic_coords(sample_res,noise_distance,curr_struct)
            for i in range(DIMENSIONS):
                for j in range(EL_CHANNEL_NUM):
                    curr_coords[i][j] = curr_coords[i][j] + neighbor_coords[i][j]

        curr_r,curr_t,curr_p = curr_coords

        # to be used once coefficients  are gathered for all channels for current res
        curr_coeffs = {}
        # list of the dicts for each channel for current res
        channel_dicts = []
        for k in ks:
            for curr_ch in range(EL_CHANNEL_NUM):
                curr_channel_coeffs = {}
                for l in range(l_max + 1):
                    # if the current channel has no signal then append zero holographic signal
                    if len(curr_r[curr_ch]) == 0:
                        curr_channel_coeffs[l] = np.array([0.]*(2*l+1))
                        continue
                    curr_channel_coeffs[l] = hgm.zernike_coeff_l(curr_r[curr_ch],
                                                                 curr_t[curr_ch],
                                                                 curr_p[curr_ch],
                                                                 k, noise_distance+5., l)
                    print('n = {} l = {}'.format(k,l))
                    if (k%2) == ((l+1)%2):
                        print('When n = {} and l = {} the channel {} coefficients are \n'.format(k,l,curr_ch))
                        print(curr_channel_coeffs)
                        print('\n')
                channel_dicts.append(curr_channel_coeffs)
        # coefficients gathered for every channel for this sample residue
        # channels_dicts currently has the structure n_c x l x m
        # we can now swap n_c and l
        for l in range(l_max + 1):
            curr_coeffs[l] = np.stack([x[l] for x in channel_dicts])
        # now we want to add this dictionary to a list of all the dicts for
        # each sample aa
        data_dicts.append(curr_coeffs)

        # make a label for the current hologram
        curr_label = [0]*len(amino_acids)
        curr_label[protein.aa_to_ind[curr_aa]] = 1

        # if aa has been found increment the aa_ind
        if aa_found ==True:
            hologram_labels.append(curr_label)
            aa_ind += 1

    # after the above while loop has finished we now assume that all amino acids have
    # been sampled and their data is compiled in the list data_dicts in the order
    # N x l x n_c x m. We want to swap the indices N and l so we'll do that here
    for l in range(l_max + 1):
        hologram_coeffs[l] = np.stack([x[l] for x in data_dicts])

    hologram_coeffs_real = {}
    hologram_coeffs_imag = {}
    for l in range(l_max + 1):
        hologram_coeffs_real[l] = np.real(hologram_coeffs[l])
        hologram_coeffs_imag[l] = np.imag(hologram_coeffs[l])
    print(hologram_coeffs_real[0].shape)
    return hologram_coeffs_real,hologram_coeffs_imag,hologram_labels
    
