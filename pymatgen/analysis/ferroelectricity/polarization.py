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

        p_elecs, p_ions = self.get_pelecs_and_pions(convert_to_muC_per_cm2=False)

        L = len(p_elecs)

        # divide volumes before adjusting quantum?

        volumes = [s.lattice.volume for s in self.structures]
        polar_volume = volumes[-1]
        smallest_vol = min(volumes)

        shifted_p_elec_cart = []
        shifted_p_ion_cart = []

        sms = []

        for i in range(L):
            # Current assumptions:
            #   Structure.get_primitive_structure() does not rotate xyz coordinates
            #   Currently assuming that it does not displace xyz coordinates, this might be WRONG.

            # For from abc coords as given by VASP to fractional coords
            l = self.structures[i].lattice
            e_frac_coord = np.divide(np.matrix(p_elecs[i]),np.matrix([l.a, l.b, l.c]))
            i_frac_coord = np.divide(np.matrix(p_ions[i]),np.matrix([l.a, l.b, l.c]))

            se = Structure(l, ["C"], [np.matrix(e_frac_coord).A1])
            si = Structure(l, ["C"], [np.matrix(i_frac_coord).A1])

            # get cart coords from structure
            site = se[0]
            p_e_cart = site.to_unit_cell
            p_e_cart = site.coords

            site = si[0]
            p_i_cart = site.to_unit_cell
            p_i_cart = site.coords

            #save lattice unit vectors
            sm = np.matrix(self.structures[i].lattice.matrix)
            sm /= np.linalg.norm(sm, axis=1)
            sms.append(sm)

            shifted_p_elec_cart.append(p_e_cart)
            shifted_p_ion_cart.append(p_i_cart)

        s_pe = Structure(smallest_vol.lattice, ["C"] * L, shifted_p_elec_cart, coords_are_cartesian=True)
        s_pi = Structure(smallest_vol.lattice, ["C"] * L, shifted_p_ion_cart, coords_are_cartesian=True)

        for i in range(L):
            if i == 0:
                continue
                # site_e, _ = s_pe.get_nearest_site([0, 0, 0], s_pe[i])
                # site_i, _ = s_pi.get_nearest_site([0, 0, 0], s_pi[i])
            else:
                site_e, _ = s_pe.get_nearest_site(s_pe[i - 1].coords, s_pe[i])
                site_i, _ = s_pi.get_nearest_site(s_pi[i - 1].coords, s_pi[i])
            s_pe.translate_sites(i, site_e.coords - s_pe[i].coords, frac_coords=False, to_unit_cell=False)
            s_pi.translate_sites(i, site_i.coords - s_pi[i].coords, frac_coords=False, to_unit_cell=False)

        shifted_p_elec = s_pe.cart_coords
        shifted_p_ion = s_pi.cart_coords

        if abc:
            shifted_p_elec = [(sms[i].I * (np.matrix(shifted_p_elec[i]).T)).T.tolist()[0] for i in range(L)]
            shifted_p_ion = [(sms[i].I * (np.matrix(shifted_p_ion[i]).T)).T.tolist()[0] for i in range(L)]

            # Should instead use cart coord to place back in original calculated unit cell

            # shifted_p_elec = np.multiply(np.matrix(s_pe.frac_coords),np.matrix([s_pe.lattice.a, s_pe.lattice.b, s_pe.lattice.c]))
            # shifted_p_ion = np.multiply(np.matrix(s_pi.frac_coords),np.matrix([s_pi.lattice.a, s_pi.lattice.b, s_pi.lattice.c]))

        CifWriter(s_pe).write_file(filename="s_pe.cif")
        CifWriter(s_pi).write_file(filename="s_pi.cif")

        shifted_p_elec_T = np.matrix(shifted_p_elec).T
        shifted_p_ion_T = np.matrix(shifted_p_ion).T

        volumes = np.matrix(volumes)

        if convert_to_muC_per_cm2:
            e_to_muC = -1.6021766e-13
            cm2_to_A2 = 1e16
            units = 1.0 / np.matrix(volumes)
            units *= e_to_muC * cm2_to_A2

            shifted_p_elec_T = np.multiply(units, shifted_p_elec_T)
            shifted_p_ion_T = np.multiply(units, shifted_p_ion_T)

        # Shift so everything starts at zero
        shifted_p_elec_T -= shifted_p_elec_T[:, 0]
        shifted_p_ion_T -= shifted_p_ion_T[:, 0]

        return shifted_p_elec_T.T, shifted_p_ion_T.T, None

    def get_polarization_change(self):
        shifted_p_elec, shifted_p_ion, shifted_total = self.get_same_branch_polarization_data(
            convert_to_muC_per_cm2=True)
        tot = shifted_p_elec + shifted_p_ion
        return (tot[-1] - tot[0])