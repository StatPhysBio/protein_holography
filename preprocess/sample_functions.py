#
# File for protein sampling functions
#

import sys
sys.path.append('/gscratch/spe/mpun/protein_holography/hologram')
import protein
import Bio.PDB as pdb
import sample_protein
import numpy as np

amino_acids = list(protein.aa_to_ind.keys())

def sample_equally(pdb_list,pdb_dir,instances):
    # get a list of all amino acids needed for the sample
    aa_to_sample = amino_acids * instances
    
    num_pdbs = len(pdb_list)

    pdb_ind = 0
    aa_ind = 0

    parser = pdb.PDBParser(QUIET=True)
    
    aa_sampled = []

    while aa_ind < len(aa_to_sample):
        if(aa_ind % 20 == 0):
            print(aa_ind)

        aa_found = False

        curr_aa = aa_to_sample[aa_ind]
        curr_pdb = pdb_list[pdb_ind % num_pdbs]
        curr_struct = parser.get_structure(curr_pdb,
                                           pdb_dir + '/' + curr_pdb + '.pdb')
        pdb_ind += 1

        sample_res = sample_protein.sample_amino_acid_from_protein(
            curr_struct,curr_pdb,curr_aa
        )
        
        if sample_res == None:
            continue
        if len([x for x in sample_res]) != protein.atoms_per_aa[curr_aa]:
            continue
        aa_found = True
        resname = sample_res.get_resname()
        res_id = sample_res.get_full_id()
 
        # convert tuple form res id to np array form res id
        new_res_id = []
        new_res_id.append(resname.encode())
        new_res_id.append(res_id[0].encode())
        new_res_id.append(res_id[1])
        new_res_id.append(res_id[2].encode())
        new_res_id.append(res_id[3][0].encode())
        new_res_id.append(res_id[3][1])
        new_res_id.append(res_id[3][2].encode())
        new_res_id = np.array(new_res_id)

        aa_info = (resname, res_id)
        aa_sampled.append(new_res_id)
        if aa_found == True:
            aa_ind += 1

    return aa_sampled

                                           
