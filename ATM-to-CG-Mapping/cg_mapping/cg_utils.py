import mdtraj
import sys
import math
import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import pdb
import networkx as nx
from networkx import NetworkXNoPath

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from collections import Counter
#from msibi.utils.find_exclusions import find_1_n_exclusions

class State(object):
    """ Container to store basic thermodynamic information

    Parameters
    ---------
    k_b : float
        Boltzmann constant
    T : float
        Temperature
    traj : MDTraj Trajectory

        """

    def __init__(self, k_b=8.314e-3, T=305, traj=None):
        self._k_b = k_b
        self._T = T
        self._traj = traj
        self._bondgraph = nx.Graph()
        self._bondgraph.add_nodes_from([a.index for a in traj.topology.atoms])
        bonds = [b for b in traj.topology.bonds]
        bonds_by_index = [(b[0].index, b[1].index) for b in bonds]
        self._bondgraph.add_edges_from(bonds_by_index)


    @property
    def k_b(self):
        return self._k_b

    @property
    def T(self):
        return self._T

    @property
    def traj(self):
        return self._traj

    @property
    def bondgraph(self):
        return self._bondgraph

    @k_b.setter
    def k_b(self, k_b):
        self._k_b = k_b

    @T.setter
    def T(self, T):
        self._T = T

    @traj.setter
    def traj(self, traj):
        self._traj = traj
        self._bondgraph = nx.Graph()
        self._bondgraph.add_nodes_from([a.index for a in traj.topology.atoms])
        bonds = [b for b in traj.topology.bonds]
        bonds_by_index = [(b[0].index, b[1].index) for b in bonds]
        self._bondgraph.add_edges_from(bonds_by_index)


    def __str__(self):
        return("k_B = {} \nT = {}\ntraj = {}".format(self._k_b, self._T, self._traj))


    def gaussian(self, x,  x0, w, A):
        """ Generic gaussian function

        Parameters
        ----------
        x0 : float
            Mean around which gaussian is centered
        w : float
            width of gaussian
        A : float
            Total area of gaussian

        Returns
        -------
        Probability

        Notes
        -----
        Equation form follows Milano, Goudeau, and Muller-Plathe (2005)
        """
        return A * np.exp(-2*(x-x0)**2/(w**2))

    def gaussian_to_energy(self, x, constant, a, x0):
        """ Compute energy distribution from gaussian distribution

        Parameters
        ----------
        constant : float
        a : coefficient on harmonic term
        x0 : reference value

        Returns
        -------

        Notes
        -----
        Equation form follows Milano, Goudeau, and Muller-Plathe (2005)
        a is only half the value of the force constant, and needs to be
        doubled
        """
        return constant + 0.5*a*(x-x0)**2


    def harmonic_energy(self, x_val, x0, force_constant):
        """ Generic harmonic energy evaluation

        Parameters
        ----------
        x_val : float
            Independent variable for which we are calculating energy
        x0 : float
            harmonic reference
        force_constant : float
            harmonic force constant

        Returns
        -------
        energy_eval
        """
        return 0.5*force_constant*((x_val-x0)**2)


    def harmonic_force(self, x_val, x0, force_constant):
        """ Generic harmonic force evaluation

        Parameters
        ----------
        x_val : float
            Independent variable for which we are calculating energy
        x0 : float
            harmonic reference
        force_constant : float
            harmonic force constant

        Returns
        -------
        force_eval
        """
        return force_constant*(x_val-x0)

    def fit_to_gaussian(self, independent_vars, dependent_vars, energy_fit=False):
        """ Fit values to gaussian distribution

        Parameters
        ----------
        independent_vars : list()
            Likely the bond lengths or angles
        dependent_vars: list()

        Returns
        -------
        harmonic_parameters : dict()
            force_constant, x0

        Notes
        -----
            """

        # Fit probabilities to gaussian probabilities
        if not energy_fit:
            params, covar = curve_fit(self.gaussian, independent_vars,
                    dependent_vars)#, method='dogbox', bounds = [(0,0,-np.inf),
                        #(np.inf, np.inf, np.inf)])
            x0 = params[0]
            w = params[1]
            A = params[2]

            # Extract spring constant
            force_constant = 4 * self._k_b * self._T/w**2
            #bonded_parameters={'force_constant': force_constant, 'x0': x0}
        # Fit gaussian energies
        else:
            params, covar = curve_fit(self.gaussian_to_energy, independent_vars,
                    dependent_vars, method='dogbox', bounds=[(-np.inf,0,0),
                        (np.inf, np.inf, np.inf)])
            constant = params[0]
            force_constant = params[1]
            x0 = params[2]

        bonded_parameters={'force_constant': force_constant, 'x0': x0}
        return bonded_parameters

    def compute_bond_parameters(self, atomtype_i, atomtype_j, plot=False):
        """
        Calculate bonded parameters from a trajectory
        Compute the probability distribution of bond lengths
        Curve fit to gaussian distribution

        Parameters
        ---------
        atomtype_i : str
            First atomtype
        atomtype_j : str
            Second atomtype

        Returns
        -------
        force_constant : float
            Harmonic bond constant
        x0 : float
          Bond reference

        """

        topol = self._traj.topology
        target_pair = (atomtype_i, atomtype_j)
        bonded_pairs = []
        if len([(i,j) for i,j in topol.bonds]) == 0:
            sys.exit("No bonds detected, check your input files")

        for (i, j) in topol.bonds:
            if set((i.name, j.name)) == set(target_pair):
                bonded_pairs.append((i.index, j.index))

        if len(bonded_pairs) == 0:
            #print("No {}-{} bonds detected".format(atomtype_i, atomtype_j))
            return None

        # Compute distance between bonded pairs
        bond_distances = np.asarray(mdtraj.compute_distances(self._traj, bonded_pairs))

        fig,ax =  plt.subplots(1,1)
        # 51 bins, 50 probabilities
        all_probabilities, bins, patches = ax.hist(bond_distances.flatten(), 50, normed=1)
        if plot:
            ax.set_xlabel("Distance (nm)")
            ax.set_ylabel("Probability")
            plt.savefig("{}-{}_bond_distribution.jpg".format(atomtype_i, atomtype_j))
            plt.close()


        # Need to compute energies from the probabilities
        # For each probability, compute energy and assign it appropriately
        all_energies = []
        all_distances = []

        for index, probability in enumerate(all_probabilities):
            first_bin = bins[index]
            second_bin = bins[index+1]
            bin_width = second_bin - first_bin
            if probability - 1e-6 <=0:
                probability = 1e-6
            energy = -self._k_b * self._T * np.log(probability)
            distance = np.mean((bins[index], bins[index+1]))
            all_energies.append(energy)
            all_distances.append(distance)
        # Shift energies to positive numbers
        min_shift = min(all_energies)
        all_energies = [energy - min_shift for energy in all_energies]
        min_index = np.argmin(all_energies)
        bonded_parameters = self.fit_to_gaussian(all_distances, all_probabilities)
        '''
        converged = False
        i = 2
        while not converged:
            try:
                # Slice data to be center the fit around the minima
                sliced_distances = all_distances[min_index-i: min_index+i]
                sliced_energies = all_energies[min_index-i: min_index+i]
                sliced_probabilities = all_probabilities[min_index-i: min_index+i]

                #bonded_parameters = self.fit_to_gaussian(sliced_distances, sliced_energies)
                bonded_parameters = self.fit_to_gaussian(sliced_distances, sliced_probabilities)
                converged=True

            except RuntimeError:
                i +=1
                if min_index + i >= 50 or min_index -i <= 0:
                    #bonded_parameters = self.fit_to_gaussian(all_distances, all_energies)
                    bonded_parameters = self.fit_to_gaussian(all_distances, all_probabilities)
                    converged=True
        '''

        predicted_energies = self.harmonic_energy(all_distances, **bonded_parameters)
        if plot:
            fig, axarray = plt.subplots(2,1,sharex=True)
            axarray[0].plot(all_distances, predicted_energies, c='darkgray', label="Predicted")
            axarray[1].plot(all_distances, all_energies, c='black', label="Target")
            axarray[0].legend()
            axarray[1].legend()
            axarray[1].set_xlabel("Distance (nm)")
            axarray[0].set_ylabel("Energy (kJ/mol)")
            axarray[1].set_ylabel("Energy (kJ/mol)")
            plt.savefig("{}-{}_bond_energies.jpg".format(atomtype_i, atomtype_j))
            plt.close()
        return bonded_parameters




    def compute_angle_parameters(self, atomtype_i, atomtype_j, atomtype_k, plot=False):
        """
        Calculate angle parameters from a trajectory
        Compute the probability distribution of angles
        Curve fit to gaussian distribution

        Parameters
        ---------
        atomtype_i : str
            First atomtype
        atomtype_j : str
            Second atomtype
        atomtype_k : str
            Third atomtype

        Returns
        -------
        force_constant : float
            Harmonic bond constant
        x0 : float
          Bond reference

        Notes
        -----
        Considers both angles i-j-k and k-j-i

        """

        all_triplets = []
        #participating_bonds = []
        if len([(i,j) for i,j in self._traj.topology.bonds]) == 0:
            sys.exit("No bonds detected, check your input files")
        # Find all the central atoms
        central_atoms = [a.index for a in self._traj.topology.atoms if
                atomtype_j in a.name]
        # For each central atom, try to build an i-j-k triplet
        # by finding neighbors
        for atom_j in central_atoms:
            for atom_i, atom_k in itertools.product(self._bondgraph.neighbors(atom_j),repeat=2):
                if atomtype_i in self._traj.topology.atom(atom_i).name and \
                        atomtype_k in self._traj.topology.atom(atom_k).name and \
                        atom_i != atom_k:
                            all_triplets.append([atom_i, atom_j, atom_k])

        if len(all_triplets) == 0:
            return None


        # Compute angle between triplets
        all_angles_rad = np.asarray(mdtraj.compute_angles(self._traj, all_triplets))


        fig,ax =  plt.subplots(1,1)
        # 51 bins, 50 probabilities
        vals, bins, patches = ax.hist([value for value in all_angles_rad.flatten() if not math.isnan(value)], 50, normed=1)
        ax.set_xlabel("Angle (rad)")
        ax.set_ylabel("Probability")
        if plot:
            plt.savefig("{}-{}-{}_angle_distribution.jpg".format(atomtype_i, atomtype_j,
            atomtype_k))
            plt.close()


        # Need to compute energies from the probabilities
        # For each probability, compute energy and assign it appropriately
        all_energies = []
        all_angles = []
        all_probabilities = []

        for index, probability in enumerate(vals):
            first_bin = bins[index]
            second_bin = bins[index+1]
            bin_width = second_bin - first_bin
            if probability - 1e-6 <=0:
                probability = 1e-6
            angle = np.mean((bins[index], bins[index+1]))
            scaled_probability = probability/np.sin(angle)
            energy = -self._k_b * self._T * np.log(scaled_probability)
            all_energies.append(energy)
            all_angles.append(angle)
            all_probabilities.append(scaled_probability)

        if plot:
            fig, ax = plt.subplots(1,1)
            ax.plot(all_angles, all_probabilities, label="Scaled probabilities")
            ax.set_xlabel("Angle (rad)")
            ax.set_ylabel("Probability")
            ax.legend()
            plt.savefig("{}-{}-{}-scaled_probabilities.jpg".format(atomtype_i, atomtype_j, atomtype_k))
            plt.close()

        # Shift energies to positive numbers
        min_shift = min(all_energies)
        all_energies = [energy - min_shift for energy in all_energies]
        min_index = np.argmin(all_energies)

        bonded_parameters = self.fit_to_gaussian(all_angles, all_probabilities)
        '''
        converged = False
        i = 2
        while not converged:
            try:
                #bonded_parameters = self.fit_to_gaussian(all_angles[min_index-i:min_index+i], all_energies[min_index-i:min_index+i])
                if min_index + i >= 50 or min_index -i <=0:
                        # Before fitting, may need to reflect energies about a particular angle
                        mirror_angles = np.zeros_like(all_angles)
                        mirror_energies = np.zeros_like(all_energies)
                        mirror_probabilities = np.zeros_like(all_probabilities)
                        for i, val in enumerate(all_angles):
                            mirror_energies[i] = all_energies[-i-1]
                            mirror_probabilities[i] = all_energies[-i-1]
                            mirror_angles[i] = 2*all_angles[-1] - all_angles[-i-1]

                        all_angles.extend(mirror_angles)
                        all_energies.extend(mirror_energies)
                        all_probabilities.extend(mirror_probabilities)
                        bonded_parameters = self.fit_to_gaussian(all_angles, all_probabilities)
                        converged=True


                else:
                    bonded_parameters = self.fit_to_gaussian(all_angles[min_index-i:min_index+i], all_probabilities[min_index-i:min_index+i])
                    if bonded_parameters['force_constant'] > 0.01:
                        converged = True
                    else:
                        converged = False
                        i += 1
            except RuntimeError:
                i+=1
                converged = False
            '''
                #try:
                #    bonded_parameters = self.fit_to_gaussian(all_angles[min_index-i:min_index+i], all_energies[min_index-i:min_index+i], energy_fit=True)
                #    if bonded_parameters['force_constant'] > 0.01:
                #        converged = True
                #    else:
                #        converged = False
                #        i += 1
                #except RuntimeError:
                #    i += 1
                #    if min_index + i >= 50 or min_index -i <=0:
                #        # Before fitting, may need to reflect energies about a particular angle
                #        mirror_angles = np.zeros_like(all_angles)
                #        mirror_energies = np.zeros_like(all_energies)
                #        mirror_probabilities = np.zeros_like(all_probabilities)
                #        for i, val in enumerate(all_angles):
                #            mirror_energies[i] = all_energies[-i-1]
                #            mirror_probabilities[i] = all_energies[-i-1]
                #            mirror_angles[i] = 2*all_angles[-1] - all_angles[-i-1]
                #
                #        all_angles.extend(mirror_angles)
                #        all_energies.extend(mirror_energies)
                #        all_probabilities.extend(mirror_probabilities)

                #        #bonded_parameters = self.fit_to_gaussian(all_angles, all_energies)
                #        try:
                #            bonded_parameters = self.fit_to_gaussian(all_angles, all_probabilities)
                #            converged=True
                #        except RuntimeError:
                #            bonded_parameters = self.fit_to_gaussian(all_angles, all_energies, energy_fit=True)
                #        converged = True

        predicted_energies = self.harmonic_energy(all_angles, **bonded_parameters)
        if plot:
            fig, axarray = plt.subplots(2,1,sharex=True)
            axarray[0].plot(all_angles, predicted_energies, c='darkgray', label="Predicted")
            axarray[1].plot(all_angles, all_energies, c='black', label="Target")
            axarray[0].legend()
            axarray[1].legend()
            axarray[1].set_xlabel("Angle (rad)")
            axarray[0].set_ylabel("Energy (kJ/mol)")
            axarray[1].set_ylabel("Energy (kJ/mol)")
            plt.savefig("{}-{}-{}_angle_energies.jpg".format(atomtype_i, atomtype_j, atomtype_k))
            plt.close()


        return bonded_parameters





    def compute_rdf(self, atomtype_i, atomtype_j, output,
            bin_width=0.01, exclude_up_to=3):
        """
        Compute RDF between pair of atoms, save to text

        Parameters
        ---------
        atomtype_i : str
            First atomtype
        atomtype_j : str
            Second atomtype
        output : str
            Filename
        exclude_up_to : int
            Exclude up to this many bonded terms for RDF calculation
            i.e., exclude_up_to=2 means 1-2 and 1-3 bonds are excluded from RDF

        Notes
        -----
        Be VERY aware of the exclusion terms for computing RDFs

            """

        """ Mine
        pairs = traj.topology.select_pairs(selection1='name {}'.format(atomtype_i),
                selection2='name {}'.format(atomtype_j))
        (first, second) = mdtraj.compute_rdf(traj, pairs, [0, 2], bin_width=0.01 )
        np.savetxt('{}.txt'.format(output), np.column_stack([first,second]))
        """


        pairs = self._traj.topology.select_pairs("name '{0}'".format(atomtype_i),
                                 "name '{0}'".format(atomtype_j))
        if exclude_up_to is not None:
            to_delete = find_1_n_exclusions(self._traj.topology, pairs, exclude_up_to)
            pairs = np.delete(pairs, to_delete, axis=0)
        (first, second) = mdtraj.compute_rdf(self._traj, pairs, [0,2], bin_width=bin_width)
        np.savetxt('{}.txt'.format(output), np.column_stack([first,second]))

