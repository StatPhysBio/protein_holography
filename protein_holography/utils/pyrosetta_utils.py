import sys
sys.path.append('/gscratch/stf/mpun/software/PyRosetta4.Release.python38.linux.release-317/')
import pyrosetta
from pyrosetta.rosetta import core, protocols, numeric, basic, utility
from protein_holography.coordinates.get_structural_info import get_padded_structural_info as c_struct
from protein_holography.coordinates.get_neighborhoods import get_padded_neighborhoods as c_nh
from protein_holography.coordinates.get_zernikegrams import get_zernikegrams as c_z
import numpy as np

# this one would be good for packing and minimization
scorefxn = pyrosetta.create_score_function('ref2015.wts')

# this one should be used if we are letting bond lengths/angles vary, as in fastrelax above
scorefxn_cartesian = pyrosetta.create_score_function('ref2015_cart.wts')

default_flags = '-ignore_unrecognized_res 1 -include_current -ex1 -ex2 '
wet_flags = '-ignore_unrecognized_res 1' \
            '-include_current -ex1 -ex2' \
            '-ignore_waters 0'

#max_atoms = 3000

def c_struct_string(pose_name):
    pose = pyrosetta.pose_from_pdb(pose_name)
    return c_struct(pose,padded_length=max_atoms)

def c_struct_shear(tup):

    pose_name = tup[0]
    site = tup[1]
    delta = tup[2]
    pose = pyrosetta.pose_from_pdb(pose_name)
    
    # get native angles
    native_phi = pose.phi(site)
    native_psi = pose.psi(site-1)
    
    # set backbone angles
    pose.set_phi(site,native_phi + delta)
    pose.set_psi(site-1,native_psi - delta)
    
    return c_struct(pose,padded_length=max_atoms)


def c_struct_mut(tup,pose=None):
    print('Getting psoe')
    if pose==None:
        pose_name = tup[0]
        pose = pyrosetta.pose_from_pdb(pose_name)
    mutation_inds = tup[1]
    mutant_aas = tup[2]
    print('Got info')
    if len(mutation_inds) == 0:
        return c_struct(pose,padded_length=max_atoms)
    print('making seqpos')
    seqposs = [get_pose_residue_number(pose, 'A', x) for x in mutation_inds]
    # assert pose.residue(seqpos).name1() == 'L'
    print('Making mutations')
    mutations = {seqpos:mutation for seqpos,mutation in zip(seqposs,mutant_aas)}
    make_mutations(mutations, pose)
    print('Mutations made')
#     # this one would be good for packing and minimization
#     scorefxn = pyrosetta.create_score_function('ref2015.wts')

#     # this one should be used if we are letting bond lengths/angles vary, as in fastrelax above
#     scorefxn_cartesian = pyrosetta.create_score_function('ref2015_cart.wts')
    print('relaxing')
    relaxation_sites = find_calpha_neighbors(seqposs, 10.0, pose)
    print('repacking')
    repack_residues(scorefxn, relaxation_sites, pose)

    return c_struct(pose,padded_length=max_atoms)

def c_struct_mut_cartesian(tup,max_atoms=3000,save_pdb=None):

    pose_name = tup[0]
    pose = pyrosetta.pose_from_pdb(pose_name)
    chain = tup[1]
    mutation_inds = tup[2]
    mutant_aas = tup[3]
    if len(mutation_inds) == 0:
        print('---\n NO mutations made\n-----')
        return c_struct(pose,padded_length=max_atoms)
    seqposs = [get_pose_residue_number(pose, c, x) for x,c in zip(mutation_inds,chain)]
    # assert pose.residue(seqpos).name1() == 'L'
    mutations = {seqpos:mutation for seqpos,mutation in zip(seqposs,mutant_aas)}
    make_mutations(mutations, pose)

#     # this one would be good for packing and minimization
#     scorefxn = pyrosetta.create_score_function('ref2015.wts')

#     # this one should be used if we are letting bond lengths/angles vary, as in fastrelax above
#     scorefxn_cartesian = pyrosetta.create_score_function('ref2015_cart.wts')

    relaxation_sites = find_calpha_neighbors(seqposs, 10.0, pose)

    fastrelax_positions(
        scorefxn_cartesian,
        #relaxation_sites,
        np.concatenate([[seqpos-1, seqpos, seqpos+1] for seqpos in seqposs]),
        relaxation_sites,
        pose
    )
    print(pose_name,'relaxed')
    if save_pdb != None:
        pose.dump_pdb(pose_name)
    return c_struct(pose,padded_length=max_atoms)




def repack_residues(
    scorefxn,
    positions, # 1-indexed pose numbering
    pose,
):
    ''' Repack the sidechains at the residues in "positions" 
    '''

    tf = core.pack.task.TaskFactory()
    tf.push_back(core.pack.task.operation.InitializeFromCommandline()) # use -ex1 and -ex2 rotamers if requested

    # dont allow any design
    op = core.pack.task.operation.RestrictToRepacking()
    tf.push_back(op)

    # freeze residues not in the positions list
    op = core.pack.task.operation.PreventRepacking()
    for i in range(1,pose.size()+1):
        if i not in positions:
            op.include_residue(i)
        else:
            print('repacking at residue', i)
    tf.push_back(op)
    packer = protocols.minimization_packing.PackRotamersMover()
    packer.task_factory(tf)
    packer.score_function(scorefxn)

    # show the packer task
    print(tf.create_task_and_apply_taskoperations(pose))
    
    packer.apply(pose)

def fastrelax_positions(
        scorefxn,
        backbone_flexible_positions,
        sidechain_flexible_positions,
        pose,
        nrepeats = 1,
):
    ''' "Relax" iterates between repacking and gradient-based minimization
    here we are doing "cartesian" relax, which allows bond lengths and angles to change slightly
    (the positions of the atoms are the degrees of freedom, rather than the internal coordinates)
    So the scorefxn should have terms to constrain these near ideal values, eg ref2015_cart.wts
    '''
    # movemap:
    mm = pyrosetta.rosetta.core.kinematics.MoveMap()
    mm.set_bb(False)
    mm.set_chi(False)
    mm.set_jump(False)

    for i in backbone_flexible_positions:
        mm.set_bb(i, True)

    for i in sidechain_flexible_positions:
        mm.set_chi(i, True)

    fr = protocols.relax.FastRelax(scorefxn_in=scorefxn,
                                   standard_repeats=nrepeats)
    fr.cartesian(True)
    fr.set_movemap(mm)
    fr.set_movemap_disables_packing_of_fixed_chi_positions(True)

    # For non-Cartesian scorefunctions, use "dfpmin_armijo_nonmonotone"
    fr.min_type("lbfgs_armijo_nonmonotone")
    fr.apply(pose)

def find_calpha_neighbors(
    core_positions,
    distance_threshold,
    pose
):
    ''' This function finds neighbors of the residues in "core_positions" based on Calpha-Calpha distance
    '''
    # include all the 'core' positions as neighbors (of themselves, e.g.)
    nbr_positions = set(core_positions)

    distance_threshold_squared = distance_threshold**2
    
    for i in range(1, pose.size()+1): # stupid Rosetta 1-indexing
        rsd1 = pose.residue(i)
        try:
            rsd1_CA = rsd1.xyz("CA") # access by string is a *little* slow; could use integer indices
            for j in core_positions:
                rsd2 = pose.residue(j)
                if rsd1_CA.distance_squared(rsd2.xyz("CA")) <= distance_threshold_squared:
                    nbr_positions.add(i)
                    break
        except:
            continue
    return nbr_positions

def get_pdb_residue_info(
    pose,
    resnum,
):
    pi = pose.pdb_info()
    return (pi.chain(resnum), pi.number(resnum), pi.icode(resnum))

def get_pose_residue_number(
    pose,
    chain,
    resnum,
    icode=' ',
):
    return pose.pdb_info().pdb2pose(chain, resnum, icode)

def make_mutations(
    mutations,
    pose,
):
    ''' Make sequence changes and repack the mutated positions
    
    mutations is a dictionary mapping from pose number to new 1-letter aa
    mutations is 1-indexed
    
    Note that we don't specify the score function! I guess the packer here is
    using a default fullatom scorefunction... Huh
    '''
    oldseq = pose.sequence()

    tf = core.pack.task.TaskFactory()
    #tf.push_back(core.pack.task.operation.InitializeFromCommandline()) # potentially include extra rotamers

    # freeze non-mutated
    op = core.pack.task.operation.PreventRepacking()
    for i in range(1,pose.size()+1):
        if i not in mutations:
            op.include_residue(i)
    tf.push_back(op)

    # force desired sequence at mutations positions
    for i, aa in mutations.items():
        op = core.pack.task.operation.RestrictAbsentCanonicalAAS()
        op.include_residue(i)
        op.keep_aas(aa)
        tf.push_back(op)
        print('make mutation:', i, oldseq[i-1], '-->', aa)

    packer = protocols.minimization_packing.PackRotamersMover()
    packer.task_factory(tf)

    # show the packer task
    print(tf.create_task_and_apply_taskoperations(pose))

    packer.apply(pose)
