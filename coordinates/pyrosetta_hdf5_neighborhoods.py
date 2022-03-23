import h5py
from functools import partial
import numpy as np
from sklearn.neighbors import KDTree
import geo2 as geo

# slice array along given indices
def slice_array(arr,inds):
    return arr[inds]

# given a set of neighbor coords, slice all info in the npProtein along neighbor inds
def get_neighborhoods(neighbor_inds,npProtein):
    return list(map(partial(slice_array,inds=neighbor_inds),npProtein))

def get_unique_chains(protein):

    valid_res_types = [b'A', b'C', b'D', b'E', b'F', b'G', b'H', b'I', b'K', b'L',
                       b'M', b'N', b'P', b'Q', b'R', b'S', b'T', b'V', b'W', b'Y']

    # get sequences and chain sequences
    seq = protein[:,0][
        np.logical_or.reduce([protein[:,0] == x for x in valid_res_types])
    ]
    chain_seq = protein[:,2][
        np.logical_or.reduce([protein[:,0] == x for x in valid_res_types])
    ]

    # get chains and associated residue sequences
    chain_seqs = {}
    for c in np.unique(chain_seq):
        chain_seqs[c] = b''.join(seq[chain_seq == c])
    
    # cluster chains by matching residue sequences
    chain_matches = {}
    for c1 in chain_seqs.keys():
        for c2 in chain_seqs.keys():
            chain_matches[(c1,c2)] = chain_seqs[c1] == chain_seqs[c2]
            
    unique_chains = []
    unique_chain_seqs = []
    for chain in chain_seqs.keys():
        if chain_seqs[chain] not in unique_chain_seqs:
            unique_chains.append(chain)
            unique_chain_seqs.append(chain_seqs[chain])

    return unique_chains


def get_neighborhoods_from_protein(np_protein,r_max=10.,uc=True):
    
    atom_names = np_protein['atom_names']
    real_locs = (atom_names != b'')
    atom_names = atom_names[real_locs]
    coords = np_protein['coords'][real_locs]
    ca_locs = (atom_names == b' CA ')
    if uc:
        chains = np_protein['res_ids'][real_locs][:,2]
        unique_chains = get_unique_chains(np_protein['res_ids'])
        nonduplicate_chain_locs = np.logical_or.reduce(
            [chains == x for x in unique_chains]
        )
        ca_locs = np.logical_and(
            ca_locs,
            nonduplicate_chain_locs
        )

    #ca_inds = np.squeeze(np.argwhere(atom_names == b' CA '))
    ca_coords = coords[ca_locs]
    ca_res_ids = np_protein['res_ids'][real_locs][ca_locs]
    tree = KDTree(coords,leaf_size=2)
    nh_ids = np_protein[3][real_locs][ca_locs]
    neighbors_list = tree.query_radius(ca_coords, r=r_max, count_only=False)
    get_neighbors_custom = partial(

        npProtein=[np_protein[x] for x in range(1,7)]
    )
    res_ids = np_protein[3][real_locs]
#     print(res_ids)
    
    # remove central residue
    for i,nh_id,neighbor_list in zip(np.arange(len(nh_ids)),nh_ids,neighbors_list):
#         print(nh_id)
        #print(neighbor_list)
        #print(np.logical_or.reduce(res_ids[0] != nh_id,axis=-1),'\n')
        neighbors_list[i] = [x for x in neighbor_list if
                             np.logical_or.reduce(res_ids[x] != nh_id,axis=-1)]
    neighborhoods = list(map(get_neighbors_custom,neighbors_list))
    
    for nh,nh_id,ca_coord in zip(neighborhoods,nh_ids,ca_coords):
        # convert to spherical coordinates
        #print(nh[3].shape,nh[3].dtype)
        #print(ca_coord,type(ca_coord))
        #print('\t',np.array(geo.cartesian_to_spherical(nh[3] - ca_coord)).shape,np.array(geo.cartesian_to_spherical(nh[3] - ca_coord)).dtype)
        #print('\t',np.array(geo.cartesian_to_spherical(nh[3])).shape,np.array(geo.cartesian_to_spherical(nh[3])).dtype)
        
        nh[3] = np.array(geo.cartesian_to_spherical(nh[3] - ca_coord))
        nh.insert(0,nh_id)

    return neighborhoods

# given a matrix, pad it with empty array
def pad(arr,padded_length=100):
    try:
        # get dtype of input array
        dt = arr[0].dtype
    except IndexError as e:
        print(e)
        print(arr)
        raise Exception
    # shape of sub arrays and first dimension (to be padded)
    shape = arr.shape[1:]
    orig_length = arr.shape[0]

    # check that the padding is large enough to accomdate the data
    if padded_length < orig_length:
        print('Error: Padded length of {}'.format(padded_length),
              'is smaller than original length of array {}'.format(orig_length))

    # create padded array
    padded_shape = (padded_length,*shape)
    mat_arr = np.empty(padded_shape, dtype=dt)
    
    # if type is string fill array with empty strings
    if np.issubdtype(bytes, dt):
        mat_arr.fill(b'')

    # add data to padded array
    mat_arr[:orig_length] = np.array(arr)
    
    return mat_arr

def pad_neighborhood(
        res_id,
    ragged_structure,
    padded_length=100
):
    
    
    pad_custom = partial(pad,padded_length=padded_length)

    max_atoms=padded_length
    dt = np.dtype([
        ('res_id', 'S5', (6)),
        ('atom_names', 'S4', (max_atoms)),
        ('elements', 'S1', (max_atoms)),
        ('res_ids', 'S5', (max_atoms,6)),
        ('coords', 'f8', (max_atoms,3)),
        ('SASAs', 'f8', (max_atoms)),
        ('charges', 'f8', (max_atoms)),
    ])

    mat_structure = np.empty(dtype=dt,shape=())
    padded_list = list(map(pad_custom,ragged_structure))
    mat_structure['res_id'] = res_id
    for i,val in enumerate(dt.names[1:]):
        # print(i,val)
        # print(padded_list[i].shape)
        # print(mat_structure.shape)
        # print(mat_structure[0].shape)
        # print(mat_structure[0][val].shape)
        mat_structure[val] = padded_list[i]

    return mat_structure

def pad_neighborhoods(
        neighborhoods,
        padded_length=600
):
    padded_neighborhoods = []
    for i,neighborhood in enumerate(neighborhoods):
        #print('Zeroeth entry',i,neighborhood[0])
        padded_neighborhoods.append(
            pad_neighborhood(
                neighborhood[0],
                [neighborhood[i] for i in range(1,7)],
                padded_length=padded_length
            )
        )
    
    #[padded_neighborhood.insert(0,nh[0]) for nh,padded_neighborhood in zip(neighborhoods,padded_neighborhoods)]
    #[padded_neighborhood['res_id'] = nh[0] for nh,padded_neighborhood in zip(neighborhoods,padded_neighborhoods)]
    padded_neighborhoods = np.array(padded_neighborhoods,dtype=padded_neighborhoods[0].dtype)
    return padded_neighborhoods
