from sklearn.neighbors import KDTree
import numpy as np
from pyrosetta.toolbox.extract_coords_pose import pose_coords_as_rows

def cartesian_to_spherical(r):
    # get cartesian coordinates
    x = r[:,0]
    y = r[:,1]
    z = r[:,2]

    # get spherical coords from cartesian
    r_mag = np.sqrt(np.sum(r*r,axis=-1))
    t = np.arccos(z/r_mag)
    p = np.arctan2(y,x)

    # return r,theta,phi
    sph_coords = np.array([r_mag,t,p])
    shape = sph_coords.shape
    return np.einsum('ij->ji',sph_coords)

def get_pdb_residue_info(
    pose,
    resnum,
):
    pi = pose.pdb_info()
    return (pi.chain(resnum), pi.number(resnum), pi.icode(resnum))


def get_neighbor_charges(central_res_id, 
                         all_atom_names, 
                         all_coords, 
                         all_res_names, 
                         all_charges,
                         r=10.):
    tree = KDTree(all_coords,leaf_size=2)
#     print(np.array(all_res_names) == central_res_id)
#     print(all_atom_names == )
    #print('ca ind called')
        
    ca_ind = np.where([x and y for x,y in zip(np.array(all_atom_names) == 'CA',
                                              np.logical_and.reduce(
                                                  np.array(all_res_names) == central_res_id,
                                                  axis=-1
                                              )
                                             )
                      ])
    #print(ca_ind)
    #print('Done called')
    ca_coord = all_coords[ca_ind]
    try:
        nn_inds = tree.query_radius(ca_coord, r=r, count_only=False)[0]
    except:
        #print(central_res_id)
        raise Exception('KDTrees query failed')
    
    #print('query called')
    not_central_nn_inds = [x for x in nn_inds if all_res_names[x] != central_res_id]
    nn_coords = all_coords[not_central_nn_inds]
    nn_charges = np.array(all_charges)[not_central_nn_inds]
    num_neighbors = len(not_central_nn_inds)
    #print(ca_coord)
    repeated_ca_coord = np.einsum('ij,nd->id',
              np.eye(num_neighbors,num_neighbors),
              ca_coord
              )
    nn_sph_coords = cartesian_to_spherical(nn_coords - repeated_ca_coord)
    #print(nn_sph_coords)
    return nn_sph_coords,nn_charges


def get_neighbor_coords_by_element(
        central_res_id, 
        all_atom_names, 
        all_coords, 
        all_res_names, 
        all_charges,
        r=10.,
        elements=['C','N','O','S']
):

    tree = KDTree(all_coords,leaf_size=2)
#     print(np.array(all_res_names) == central_res_id)
#     print(all_atom_names == )
    #print('ca ind called')
        
    ca_ind = np.where([x and y for x,y in zip(np.array(all_atom_names) == 'CA',
                                              np.logical_and.reduce(
                                                  np.array(all_res_names) == central_res_id,
                                                  axis=-1
                                              )
                                             )
                      ])
    #print(ca_ind)
    #print('Done called')
    ca_coord = all_coords[ca_ind]
    try:
        nn_inds = tree.query_radius(ca_coord, r=r, count_only=False)[0]
    except:
        #print(central_res_id)
        raise Exception('KDTrees query failed')
    
    #print('query called')
    not_central_nn_inds = [x for x in nn_inds if all_res_names[x] != central_res_id]
    nn_coords = all_coords[not_central_nn_inds]
    nn_atoms = np.array(
        [
            ''.join(
                [y for y in x if not y.isdigit()][0])
            for x in np.array(all_atom_names)[not_central_nn_inds]
        ]
    )
    

    
    num_neighbors = len(not_central_nn_inds)
    #print(ca_coord)
    repeated_ca_coord = np.einsum('ij,nd->id',
              np.eye(num_neighbors,num_neighbors),
              ca_coord
              )
    nn_sph_coords = cartesian_to_spherical(nn_coords - repeated_ca_coord)

    elementwise_sph_coords = []
    for el in elements:
        elementwise_sph_coords.append(nn_sph_coords[nn_atoms == el])
    #print(elementwise_sph_coords)
    
    #print(nn_sph_coords)
    return elementwise_sph_coords,elements



def get_res_neighbor_charge_coords(
    res_id,
    pose,
    d=10.
):
    
    
    coords = pose_coords_as_rows(pose)
    charges = []
    atom_names = []
    res_names = []
    for i in range(1,pose.size()+1):
        for j in range(1,len(pose.residue(i).atoms())+1):
            charges.append(pose.residue_type(i).atom_charge(j))
            atom_names.append(pose.residue_type(i).atom_name(j).replace(' ',''))    
            res_names.append(get_pdb_residue_info(pose,i))
    # if coords.shape[0] != charges.shape[0]:
    #     print('\nCoords and charges have different shapes')
    #     print('Coords shape is ',coords.shape)
    print('Charges shape is ',np.array(charges).shape)
    print(res_id)
    
    try:
        nn_coords,nn_charges = get_neighbor_charges(
            res_id,
            atom_names,
            coords,
            res_names,
            charges
        )
    except Exception as e:
        print(e)
        raise Exception('Charge gathering failed in get_res_neighbor_charge_coords');
    if 0. in nn_coords[:,0]:
        print('Error: atom lies at the origin. Check for multiple models in pdb file')
        raise Exception('Exception: atom lies at the origin. Check for multiple models in pdb file')
        print(pose)
    return nn_coords,nn_charges
    


def get_res_neighbor_element_coords(
        res_id,
        pose,
        d=10.,
        elements=['C','N','O','S']
):
    
    
    coords = pose_coords_as_rows(pose)
    charges = []
    atom_names = []
    res_names = []
    for i in range(1,pose.size()+1):
        for j in range(1,len(pose.residue(i).atoms())+1):
            charges.append(pose.residue_type(i).atom_charge(j))
            atom_names.append(pose.residue_type(i).atom_name(j).replace(' ',''))    
            res_names.append(get_pdb_residue_info(pose,i))
    # if coords.shape[0] != charges.shape[0]:
    #     print('\nCoords and charges have different shapes')
    #     print('Coords shape is ',coords.shape)
    #print('Charges shape is ',np.array(charges).shape)
    #print(res_id)
    
    try:
        nn_coords,nn_charges = get_neighbor_coords_by_element(
            res_id,
            atom_names,
            coords,
            res_names,
            charges,
            elements=elements
        )
    except Exception as e:
        print(e)
        raise Exception('Charge gathering failed in get_res_neighbor_charge_coords');
    if 0. in nn_coords[0][:,0]:
        print('Error: atom lies at the origin. Check for multiple models in pdb file')
        raise Exception('Exception: atom lies at the origin. Check for multiple models in pdb file')
        print(pose)
    return nn_coords,nn_charges
    

