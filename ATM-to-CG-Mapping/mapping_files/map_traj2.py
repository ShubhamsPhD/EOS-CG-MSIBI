import time
import pdb
import warnings
import itertools
from multiprocessing import Pool
from collections import OrderedDict
import os
from optparse import OptionParser
import numpy as np
import mdtraj
import mbuild as mb
import cg_mapping
import xml.etree.ElementTree as ET
import copy as cp

PATH_TO_MAPPINGS='/Users/sd3008/cg_mapping/mapping_files'
HOOMD_FF="/Users/sd3008/cgsimulations/RunOnDMOG/lipids/cg_molecules/cg-force-field.xml"

def _load_mapping(mapfile=None,reverse=False):
    """ Load a forward mapping

    Parameters
    ----------
    mapfile : str
        Path to mapping file

    Returns
    -------
    mapping_dict : OrderedDict()
        OrderedDict (CG bead index : [beadtype, list of atom indices])

    Notes
    -----
    mapping files are ":" delimited
    col0: bead index
    col1: bead type
    col2: atom indices

        """

    mapping_dict = OrderedDict()
    bonding_info = []
    with open(mapfile,'r') as f:
        for line in f:
            if line.rstrip():
                if 'bond' in line.split(":")[0]:
                    atom_i = line.split(":")[1].split()[0].rstrip()
                    atom_j = line.split(":")[1].split()[1].rstrip()
                    bonding_info.append((atom_i, atom_j))

                else:
                    mapping_dict.update({line.split(":")[0].rstrip():
                            [line.split(":")[1].rstrip(), eval(line.split(":")[2])]})
    return mapping_dict, bonding_info



def _create_CG_topology(topol=None, all_CG_mappings=None, water_bead_mapping=4,
        all_bonding_info=None):
    """ Create CG topology from given topology and mapping

    Parameters
    ---------
    topol : mdtraj Topology
    all_CG_mappings : dict
        maps residue names to respective CG
        mapping dictionaries(CG index, [beadtype, atom indices])
    water_bead_mapping : int
        specifies how many water molecules get mapped to a water CG bead
    all_bonding_info : dict
        maps residue names to bonding info arrays
        np.ndarray (n, 2)


    Returns
    -------
    CG_topology_map : tuple of CG_bead()
    CG_topology : mdtraj topology
        Need to fill in more details for topology creation

    """
    CG_topology_map = []
    CG_topology = mdtraj.Topology()
    CG_beadindex = 0
    water_counter = 0
    # Loop over all residues
    for residue in topol.residues:
        if not (residue.name == 'SOL'):
            # Obtain the correct molecule mapping based on the residue
            molecule_mapping = all_CG_mappings[residue.name]
            temp_residue = CG_topology.add_residue(residue.name, CG_topology.add_chain())
            temp_CG_indices = []
            temp_CG_atoms = []
            temp_CG_beads = [None]*len(molecule_mapping.keys())
            CG_atoms = []

            # Make a list of atoms in the residue
            atoms = []
            for atom in residue.atoms:
                atoms.append(atom)

            for key in molecule_mapping.keys():
                for index in molecule_mapping[key][1]:
                    temp_CG_atoms.append(atoms[index])
                new_bead = cg_mapping.CG_bead(beadindex=0,
                                   beadtype=molecule_mapping[key][0],
                                   resname=residue.name,
                                   atom_indices=[atom.index for atom in temp_CG_atoms])
                temp_CG_atoms = []
                temp_CG_beads[int(key)] = new_bead

            # Add beads to topology by adding atoms
            for index, bead in enumerate(temp_CG_beads):
                bead.beadindex = index
                CG_topology_map.append(bead)
                new_CG_atom = CG_topology.add_atom(bead.beadtype, None, temp_residue)
                CG_atoms.append(new_CG_atom)

            # Add bonds to topolgoy
            bonding_info = all_bonding_info[residue.name]
            for (index_i, index_j) in bonding_info:
                CG_topology.add_bond(CG_atoms[int(index_i)], CG_atoms[int(index_j)])

        else:
            water_counter +=1
            if water_counter % water_bead_mapping == 0:
                temp_residue = CG_topology.add_residue("tip3p", CG_topology.add_chain())
                new_bead = cg_mapping.CG_bead(beadindex=0, beadtype="W",
                        resname='tip3p')
                CG_topology_map.append(new_bead)
                CG_topology.add_atom("W", None, temp_residue)





    return CG_topology_map, CG_topology

def _map_waters(traj, water_start, frame_index):
    """ Worker function to parallelize mapping waters via kmeans

    Parameters
    ----------
    traj : mdtraj trajectory
        full atomstic trajectory
    frame index : int
        parallelizing calculation frame by frame
    water_start : int
        counter denoting which index in the CG coordiantes is water

    """
    from sklearn import cluster
    start = time.time()
    frame = traj[frame_index]
    # Get atom indices of all water oxygens
    waters = frame.topology.select('resname tip3p and name O1')

    # Get coordinates and number of all water oxygens
    n_aa_water = len(waters)
    aa_water_xyz = frame.atom_slice(waters).xyz[0,:,:]

    # Number of CG water molecules based on mapping scheme
    water_bead_mapping = 4
    n_cg_water = int(n_aa_water /  water_bead_mapping)
    # Water clusters are a list (n_cg_water) of empty lists
    water_clusters = [[] for i in range(n_cg_water)]

    # Perform the k-means clustering based on the AA water xyz
    k_means = cluster.KMeans(n_clusters=n_cg_water)
    k_means.fit(aa_water_xyz)

    # Each cluster index says which cluster an atom belongs to
    for atom_index, cluster_index in enumerate(k_means.labels_):
        # Sort each water atom into the corresponding cluster
        # The item being added should be an atom index
        water_clusters[cluster_index].append(waters[atom_index])


    single_frame_coms = []
    # For each cluster, compute enter of mass
    for cg_index, water_cluster in enumerate(water_clusters):
        com = mdtraj.compute_center_of_mass(frame.atom_slice(water_cluster))
        single_frame_coms.append((frame_index, cg_index+water_start, com))
        #CG_xyz[frame_index, cg_index + water_start,:] = com
    end = time.time()
    print("K-means for frame {}: {}".format(frame_index, end-start))

    return single_frame_coms

def _convert_xyz(traj=None, CG_topology_map=None, water_bead_mapping=4,parallel=True):
    """Take atomistic trajectory and convert to CG trajectory

    Parameters
    ---------
    traj : mdtraj Trajectory
        Atomistic trajectory
    CG_topology : list
        list of CGbead()
    parallel : boolean
        True if using parallelized, false if using serial

    Returns
    ------
    CG_xyz : np.ndarray(n_frame, n_CG_beads, 3)


    """
    # Iterate through the CG_topology
    # For each bead, get the atom indices
    # Then slice the trajectory and compute hte center of mass for that particular bead
    entire_start = time.time()
    CG_xyz = np.ndarray(shape=(traj.n_frames, len(CG_topology_map),3))
    water_indices = []
    print("Converting non-water atoms into beads over all frames")
    start = time.time()
    for index, bead in enumerate(CG_topology_map):
        if index % 250 == 0:
            print("On bead {} of {}       ".format(index, len(CG_topology_map)), end='\r')
        if 'tip3p' not in bead.resname:
            # Handle non-water residuse with center of mass calculation over all frames
            atom_indices = bead.atom_indices
            # Two ways to compute center of mass, both are pretty fast
            bead_coordinates = mdtraj.compute_center_of_mass(traj.atom_slice(atom_indices))
            CG_xyz[:, index, :] = bead_coordinates
        else:
            # Handle waters by initially setting the bead coordinates to zero
            # Remember which coarse grain indices correspond to water
            water_indices.append(index)
            CG_xyz[:,index,:] = np.zeros((traj.n_frames,3))
    print()

    end = time.time()
    print("Converting took: {}".format(end-start))

    # Figure out at which coarse grain index the waters start
    print("Converting water beads via k-means")
    start = time.time()
    if len(water_indices)>0:
        water_start = min(water_indices)


        # Perform kmeans, frame-by-frame, over all water residues
        # Workers will return centers of masses of clusters, frame index, and cg index
        # Master will assign to CG_xyz
        if parallel:
            all_frame_coms = []
            with Pool() as p:
                all_frame_coms = p.starmap(_map_waters, zip(itertools.repeat(traj),
                    itertools.repeat(water_start), range(traj.n_frames)))

            end = time.time()
            print("K-means and converting took: {}".format(end-start))

            print("Writing to CG-xyz")
            start = time.time()
            for snapshot in all_frame_coms:
                for element in snapshot:
                    CG_xyz[element[0],element[1],:] = element[2]
            end =  time.time()
            print("Writing took: {}".format(end-start))

            return CG_xyz

        else:
            from sklearn import cluster
            for frame_index, frame in enumerate(traj):
                # Get atom indices of all water oxygens
                waters = traj.topology.select('(resname == tip3p) and name O1')

                # Get coordinates and number of all water oxygens
                n_aa_water = len(waters)
                aa_water_xyz = traj.atom_slice(waters).xyz[frame_index,:,:]

                # Number of CG water molecules based on mapping scheme
                n_cg_water = int(n_aa_water /  water_bead_mapping)
                # Water clusters are a list (n_cg_water) of empty lists
                water_clusters = [[] for i in range(n_cg_water)]

                # Perform the k-means clustering based on the AA water xyz
                start = time.time()
                print("Clustering for frame {}".format(frame_index))
                k_means = cluster.KMeans(n_clusters=n_cg_water)
                k_means.fit(aa_water_xyz)
                end = time.time()
                print("Clustering one frame took: {}".format(end-start))

                # Each cluster index says which cluster an atom belongs to
                print("Assigning water atoms to water clusters for frame {}".format(frame_index))
                start = time.time()
                for atom_index, cluster_index in enumerate(k_means.labels_):
                    # Sort each water atom into the corresponding cluster
                    # The item being added should be an atom index
                    water_clusters[cluster_index].append(waters[atom_index])
                end = time.time()
                print("Assigning took: {}".format(end-start))


                # For each cluster, compute enter of mass
                print("Computing cluster centers for frame {}".format(frame_index))
                start = time.time()
                for cg_index, water_cluster in enumerate(water_clusters):
                    com = mdtraj.compute_center_of_mass(traj.atom_slice(water_cluster)[frame_index])
                    CG_xyz[frame_index, cg_index + water_start,:] = com
                end = time.time()
                print("Computing took: {}".format(end-start))

    entire_end = time.time()
    print("XYZ conversion took: {}".format(entire_end - entire_start))
    return CG_xyz

def compute_avg_box(traj):
    """ Compute average box lengths"""

    return np.mean(traj.unitcell_lengths, axis=0)

def fix_hoomd(filename):
    # now get rind of number after ring in types (i.e., ring1 -> ring)
    for i in range(1, 1+4):
        os.system("perl -pi -e 's/ring{0}\n/ring\n/g' {1}".format(i, filename))

    # fix angles for CER NP and CER AP
    tree = ET.parse(filename)
    root = tree.getroot()
    remove_list = []
    for index, child in enumerate(root[0]):
        if child.tag == 'angle':
            old = child.text.split("\n")
            new = []
            for line in old:
                if line.split(" ")[0] == 'mhead2-tail-oh3':
                    oldline = cp.deepcopy(line)
                    newline1 = oldline.replace('mhead2-tail-oh3', 'mhead2-tail-oh3-a')
                    new.append(newline1)
                    oldline = cp.deepcopy(line)
                    newline2 = oldline.replace('mhead2-tail-oh3', 'mhead2-tail-oh3-b')
                    new.append(newline2)
                else:
                    new.append(line)
            new = "\n".join(new)
            root[0][index].text = new
        if child.tag in {"pair_coeffs", "bond_coeffs", "angle_coeffs", "dihedral_coeffs"}:
            remove_list.append(child)
    for child in remove_list:
        root[0].remove(child)
    tree.write(filename)



parser = OptionParser()
parser.add_option("-f", action="store", type="string", dest = "trajfile", default='traj_wrapped.xtc')
parser.add_option("-c", action="store", type="string", dest = "topfile", default=None)
parser.add_option("-o", action="store", type="string", dest = "output", default='cg-traj')
(options, args) = parser.parse_args()


if options.topfile:
    traj = mdtraj.load(options.trajfile, top=options.topfile)
else:
    traj = mdtraj.load(options.trajfile)
topol = traj.topology
start=time.time()

traj = traj.atom_slice(traj.top.select('not resname tip3p'))

print('Loaded trajectory with {} frames and {} residues'.format(traj.n_frames, traj.top.n_residues))

# Huge dictionary of dictionaries, keys are molecule names
# Values are the molecule's mapping dictionary
# could be made more pythonic
all_CG_mappings = OrderedDict()
all_bonding_info = OrderedDict()

import glob
for filename in glob.glob("{}/*map".format(PATH_TO_MAPPINGS)):
    molecule_mapping, molecule_bonding = _load_mapping(mapfile=filename)
    resname = filename.split("/")[-1].split(".")[0]
    all_CG_mappings.update({resname : molecule_mapping})
    all_bonding_info.update({resname : molecule_bonding})

CG_topology_map, CG_topology = _create_CG_topology(topol=topol, all_CG_mappings=all_CG_mappings, all_bonding_info=all_bonding_info)
CG_xyz = _convert_xyz(traj=traj, CG_topology_map=CG_topology_map, parallel=True)

CG_traj = mdtraj.Trajectory(CG_xyz, CG_topology, time=traj.time,
        unitcell_lengths=traj.unitcell_lengths, unitcell_angles = traj.unitcell_angles)

# Because we are resizing the box based on the average box over the trajectory,
# we need to scale the coordinates appropriately, since they are taken only
# from the final frame.

CG_traj.xyz /= 6
CG_traj.unitcell_lengths /= 6

CG_traj.save('{}.xtc'.format(options.output))
CG_traj[-1].save('{}.gro'.format(options.output))
CG_traj[-1].save('{}.h5'.format(options.output))

# Save as a hoomd file
#for index, atom in enumerate(CG_traj.top.atoms):
#    if atom.name == 'W':
#        CG_traj.top.atom(index).name = "water"
#    if atom.name == 'tail2':
#        CG_traj.top.atom(index).name = "ter2"
#    if atom.name == 'taild':
#        CG_traj.top.atom(index).name = "tail"

#system = mb.Compound()
#system.from_trajectory(CG_traj[-1])

#for particle in system.particles():
#    particle.name = "_"+ particle.name.strip()

#box = mb.Box(lengths=CG_traj.unitcell_lengths[-1])

#filename = '{}.hoomdxml'.format(options.output)
#system.save(filename, forcefield_files=HOOMD_FF, box=box, overwrite=True, ref_distance=0.1, foyerkwargs={'assert_dihedral_params':False})
#fix_hoomd(filename)

end=time.time()
print(end-start)


