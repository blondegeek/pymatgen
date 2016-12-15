# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, print_function, unicode_literals
from __future__ import absolute_import

from math import *
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.core.sites import PeriodicSite
from pymatgen.io.cif import CifWriter
import numpy as np

"""
This module provides the classes needed to analyze the change in polarization
during a ferroelectric transition. Currently, this only handles data from VASP calculations.
"""

__author__ = "Tess Smidt"
__copyright__ = "Copyright 2016, The Materials Project"
__version__ = "1.0"
__email__ = "tsmidt@berkeley.edu"
__status__ = "Development"
__date__ = "July 31, 2016"


class PolarizationChange(object):
    """
    Calculates the change in polarization from a series of polarization calculations along a ferroelectric transition.
    The structures are assumed to be an interpolation from a nonpolar to polar structure, with the Outcars in the same
    order.

    A good primer for polarization theory is:
    http://physics.rutgers.edu/~dhv/pubs/local_preprint/dv_fchap.pdf

    Also, from the VASP documentation:
    http://cms.mpi.univie.ac.at/vasp/Berry_phase/node3.html

    The challenging bit is that polarization is a many valued function, so we need to make sure we are comparing the
    same "branch" through the ferroelectric transition.

    VASP outputs the polarization as a dot product with the three reciprocal lattice vectors.


    """

    def __init__(self, structures, outcars):
        self.structures = structures
        if all(isinstance(o, Outcar) for o in outcars):
            self.outcars = [o.to_dict for o in outcars]
        elif all(isinstance(o, dict) for o in outcars):
            # also insert test for polarization p_elec and p_ion fields
            self.outcars = outcars
        else:
            print("Please give list of Outcar or dicts generated from Outcar")

    def get_pelecs_and_pions(self, convert_to_muC_per_cm2=False):
        p_elecs = np.matrix([o['p_elec'] for o in self.outcars])
        p_ions = np.matrix([o['p_ion'] for o in self.outcars])

        volumes = [s.lattice.volume for s in self.structures]

        # if convert_to_muC_per_cm2:
        #     volumes = [s.lattice.volume for s in self.structures]
        #     e_to_muC = -1.6021766e-13
        #     cm2_to_A2 = 1e16
        #     for i in range(p_elecs.shape[0]):
        #         for j in range(p_elecs.shape[1]):
        #             p_elecs[i, j] = p_elecs[i, j] / volumes[i] * (e_to_muC * cm2_to_A2)
        #             p_ions[i, j] = p_ions[i, j] / volumes[i] * (e_to_muC * cm2_to_A2)

        p_elecs = np.matrix(p_elecs).T
        p_ions = np.matrix(p_ions).T

        if convert_to_muC_per_cm2:
            e_to_muC = -1.6021766e-13
            cm2_to_A2 = 1e16
            units = 1.0 / np.matrix(volumes)
            units *= e_to_muC * cm2_to_A2

            p_elecs = np.multiply(units, p_elecs)
            p_ions = np.multiply(units, p_ions)

        p_elecs, p_ions = p_elecs.T, p_ions.T

        return p_elecs, p_ions

    def get_same_branch_polarization_data(self, convert_to_muC_per_cm2=False, abc=True):
        """
        Get same branch polarization for given polarization data.

        convert_to_muC_per_cm2: convert polarization from electron * Angstroms to microCoulomb per centimeter**2
        abc: return polarization in coordinates of a,b,c (versus x,y,z)
        """

        from pymatgen.core.lattice import Lattice

        p_elec, p_ion = self.get_pelecs_and_pions()
        p_tot = p_elec + p_ion

        L = len(p_elec)
        volumes = [s.lattice.volume for s in self.structures]
        d_structs = []
        sites = []

        factor = 2.0

        for i in range(L):
            l = self.structures[i].lattice
            # frac_coord = np.divide(np.matrix(p_tot[i]), np.matrix([l.a, l.b, l.c]))
            frac_coord = np.divide(np.matrix(p_tot[i]), np.matrix([l.a, l.b, l.c]))*factor
            # new stuff
            angles = l.angles
            l_new = Lattice.from_lengths_and_angles((l.a / factor, l.b / factor, l.c / factor), angles)
            d = Structure(l_new, ["C"], [np.matrix(frac_coord).A1])
            # d = Structure(l, ["C"], [np.matrix(frac_coord).A1])
            d_structs.append(d)
            site = d[0]
            if i == 0:
                prev_site = [0, 0, 0]
            else:
                prev_site = sites[-1].coords
            new_site = d.get_nearest_site(prev_site, site)
            sites.append(new_site[0])

        adjust_pol = []

        for s,d in zip(sites,d_structs):
            l = d.lattice
            adjust_pol.append(np.multiply(s.frac_coords, np.matrix([l.a / factor, l.b / factor, l.c / factor])).A1 * factor)
            # adjust_pol.append(np.multiply(s.frac_coords, np.matrix([l.a, l.b, l.c])).A1)

        volumes = np.matrix(volumes)

        adjust_pol = np.matrix(adjust_pol)

        if convert_to_muC_per_cm2:
            e_to_muC = -1.6021766e-13
            cm2_to_A2 = 1e16
            units = 1.0 / np.matrix(volumes)
            units *= e_to_muC * cm2_to_A2
            adjust_pol = np.multiply(units.T, adjust_pol)

        return adjust_pol

    def get_polarization_change(self):
        tot = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=True)
        return (tot[-1] - tot[0])