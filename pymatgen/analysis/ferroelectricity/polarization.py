# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, print_function, unicode_literals
from __future__ import absolute_import

from math import *
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Outcar
from pymatgen.core.sites import PeriodicSite
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
    def __init__(self,structures,outcars):
        self.structures = structures
        if all(isinstance(o, Outcar) for o in outcars):
            self.outcars = [o.to_dict for o in outcars]
        elif all(isinstance(o, dict) for o in outcars):
            # also insert test for polarization p_elec and p_ion fields
            self.outcars = outcars
        else:
            print("Please give list of Outcar or dicts generated from Outcar")

    def get_pelecs_and_pions(self,convert_to_muC_per_cm2=False):
        p_elecs = np.matrix([o['p_elec'] for o in self.outcars])
        p_ions = np.matrix([o['p_ion'] for o in self.outcars])


        if convert_to_muC_per_cm2:
            volumes = [s.lattice.volume for s in self.structures]
            e_to_muC = -1.6021766e-13
            cm2_to_A2 = 1e16
            for i in range(p_elecs.shape[0]):
                for j in range(p_elecs.shape[1]):
                    p_elecs[i, j] = p_elecs[i, j] / volumes[i] * (e_to_muC * cm2_to_A2)
                    p_ions[i, j] = p_ions[i, j] / volumes[i] * (e_to_muC * cm2_to_A2)

        return p_elecs,p_ions

    def get_same_branch_polarization_data(self, convert_to_muC_per_cm2=False, half_quantum=False):

        p_elecs, p_ions = self.get_pelecs_and_pions(convert_to_muC_per_cm2=False)

        L = len(p_elecs)

        p_tot = p_elecs + p_ions

        # Assuming primitive cells do not rotate xyz coords
        primitives = [s.get_primitive_structure() for s in self.structures]

        shifted_p_elec = []
        shifted_p_ion = []
        shifted_p_elec_cart = []
        shifted_p_ion_cart = []

        for i in range(L):
            p_e = np.matrix(p_elecs[i]).T
            p_i = np.matrix(p_ions[i]).T
            sm = np.matrix(self.structures[i].lattice.matrix)
            sm /= np.linalg.norm(sm,axis=1)
            p_e_cart = (sm * p_e).T.tolist()[0]
            p_i_cart = (sm * p_i).T.tolist()[0]

            p_e_prim = PeriodicSite("C", p_e_cart, primitives[i].lattice,
                                    coords_are_cartesian=True).to_unit_cell.coords
            p_i_prim = PeriodicSite("C", p_i_cart, primitives[i].lattice,
                                    coords_are_cartesian=True).to_unit_cell.coords

            shifted_p_elec_cart.append(p_e_prim)
            shifted_p_ion_cart.append(p_i_prim)

            p_e_final = (sm.I * (np.matrix(p_i_prim).T)).T.tolist()[0]
            p_i_final = (sm.I * (np.matrix(p_e_prim).T)).T.tolist()[0]

            shifted_p_elec.append(p_e_final)
            shifted_p_ion.append(p_i_final)

        volumes = [s.get_primitive_structure().lattice.volume for s in self.structures]

        from pymatgen.io.cif import CifWriter
        smallest_prim = primitives[volumes.index(min(volumes))]

        s_pe = Structure(smallest_prim.lattice,["C"]*L,shifted_p_elec_cart,coords_are_cartesian=True)
        s_pi = Structure(smallest_prim.lattice,["C"]*L,shifted_p_ion_cart,coords_are_cartesian=True)

        dists_e, images_e = [], []
        dists_i, images_i = [], []

        new_cart_es = []
        new_cart_ie = []

        for i in range(L):
            if i == 0:
                dist_e, image_e = s_pe[i].distance_and_image_from_frac_coords([0,0,0])
                dist_i, image_i = s_pi[i].distance_and_image_from_frac_coords([0,0,0])
            else:
                dist_e, image_e = s_pe[i].distance_and_image(s_pe[i-1])
                dist_i, image_i = s_pi[i].distance_and_image(s_pe[i-1])
            dists_e.append(dist_e)
            dists_i.append(dist_i)
            images_e.append(image_e)
            images_i.append(image_i)

            s_pe.translate_sites(i, -1 * np.array(image_e), frac_coords=True, to_unit_cell=False)
            s_pi.translate_sites(i, -1 * np.array(image_i), frac_coords=True, to_unit_cell=False)

        print(dists_e)
        print(images_e)
        print(dists_i)
        print(images_i)

        CifWriter(s_pe).write_file(filename="s_pe.cif")
        CifWriter(s_pi).write_file(filename="s_pi.cif")

        shifted_p_elec_T = np.matrix(shifted_p_elec).T
        shifted_p_ion_T = np.matrix(shifted_p_ion).T

        volumes = np.matrix(volumes)

        # intervals = np.matrix([[s.lattice.a,
        #                         s.lattice.b,
        #                         s.lattice.c]
        #                        for s in self.structures])
        #
        #
        #
        # # This should be 1 if I have the theory right, but 0.5 seems to work better.
        # # I'm likely missing something important.
        # intervals *= 1
        #
        # #volumes = np.matrix([s.lattice.volume for s in self.structures])
        #
        #
        # shifted_p_elec_T = [shiftList(
        #     p_elecs.T[i].tolist()[0],
        #     start=0.0,
        #     intervals=intervals.T[i].tolist()[0]) for i in range(3)]
        #
        # shifted_p_ion_T = [shiftList(
        #     p_ions.T[i].tolist()[0],
        #     start=0.0,
        #     intervals=intervals.T[i].tolist()[0]) for i in range(3)]
        #
        # shifted_p_tot_T = [shiftList(
        #     p_tot.T[i].tolist()[0],
        #     start=0.0,
        #     intervals=intervals.T[i].tolist()[0]) for i in range(3)]
        #
        # shifted_p_elec_T = np.matrix(shifted_p_elec_T)
        # shifted_p_ion_T = np.matrix(shifted_p_ion_T)
        # shifted_p_tot_T = np.matrix(shifted_p_tot_T)

        if convert_to_muC_per_cm2:
            e_to_muC = -1.6021766e-13
            cm2_to_A2 = 1e16
            units = 1.0 / np.matrix(volumes)
            units *= e_to_muC * cm2_to_A2

            shifted_p_elec_T = np.multiply(units, shifted_p_elec_T)
            shifted_p_ion_T = np.multiply(units, shifted_p_ion_T)
            #shifted_p_tot_T = np.multiply(units, shifted_p_tot_T)

        # Shift so everything starts at zero
        shifted_p_elec_T -= shifted_p_elec_T[:, 0]
        shifted_p_ion_T -= shifted_p_ion_T[:, 0]
        #shifted_p_tot_T -= shifted_p_tot_T[:, 0]

        return shifted_p_elec_T.T, shifted_p_ion_T.T, None

    def get_polarization_change(self):
        shifted_p_elec, shifted_p_ion, shifted_total = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=True)
        tot = shifted_p_elec + shifted_p_ion
        return (tot[-1]-tot[0])

def shift(compare, value, interval):
    """
    Given two numbers 'compare' and 'value' figure out how many
    'interval's should be added or subtracted to achieve closest distance.

    compare (float) -- number to compare to
    value (float) -- number to add or subtract an interger number of intervals
    interval (float) -- number of interval jumps
    """

    n = (float(compare)-float(value))/ float(interval)
    return value + round(n)*interval


def shiftList(shiftlist, start=0.0, intervals=1):
    """
    Given a list, shift it such that the points are closest to the previous point

    shiftlist (list of floats) -- list of floats to shift
    start (float) -- value for first point to get closest to
    interval (list of floats or float) -- quantized spacing between points
    """

    if type(intervals) == float or type(intervals) == int:
        intervals = [intervals for i in range(len(shiftlist))]

    new = []
    prev = start
    for i, j in enumerate(shiftlist):
        new.append(shift(prev, j, intervals[i]))
        prev = new[-1]
    return new
