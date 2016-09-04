# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, print_function, unicode_literals
from __future__ import absolute_import

from math import *
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Outcar
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

    def get_pelecs_and_pions(self):
        p_elecs = np.matrix([o['p_elec'] for o in self.outcars])
        p_ions = np.matrix([o['p_ion'] for o in self.outcars])

        return p_elecs,p_ions

    def get_same_branch_polarization_data(self, convert_to_muC_per_cm2=False, half_quantum=False):

        p_elecs, p_ions = self.get_pelecs_and_pions()

        p_elecs_T = p_elecs.T
        p_ions_T = p_ions.T

        intervals = np.matrix([[s.lattice.a,
                                s.lattice.b,
                                s.lattice.c]
                               for s in self.structures])
        intervals_T = intervals.T
 #       intervals_T = intervals.T*0.5

        shifted_p_elec_T = []
        shifted_p_ion_T = []
        shifted_total_T = []

        for i in range(3):
            if half_quantum == True:
                # this should either be plus OR minus half a quantum
                start = intervals_T[i].tolist()[0][0]
            else:
                start = 0.0
            shifted_p_elec_T.append(shiftList(p_elecs_T[i].tolist()[0],start=start,intervals = intervals_T[i].tolist()[0]))
            shifted_p_ion_T.append(shiftList(p_ions_T[i].tolist()[0],start=start,intervals = intervals_T[i].tolist()[0]))
            shifted_total_T.append(shiftList((p_ions_T[i]+p_elecs_T[i]).tolist()[0],start=start,intervals = intervals_T[i].tolist()[0]))

        shifted_p_elec_T = np.matrix(shifted_p_elec_T)
        shifted_p_ion_T = np.matrix(shifted_p_ion_T)
        shifted_total_T = np.matrix(shifted_total_T)

        if convert_to_muC_per_cm2:
            volumes = [s.lattice.volume for s in self.structures]
            e_to_muC = -1.6021766e-13
            cm2_to_A2 = 1e16
            for i in range(shifted_p_elec_T.shape[0]):
                for j in range(shifted_p_elec_T.shape[1]):
                    shifted_p_elec_T[i,j] = shifted_p_elec_T[i,j]/volumes[i]*(e_to_muC*cm2_to_A2)
                    shifted_p_ion_T[i,j] = shifted_p_ion_T[i,j]/volumes[i]*(e_to_muC*cm2_to_A2)
                    shifted_total_T[i,j] = shifted_total_T[i,j]/volumes[i]*(e_to_muC*cm2_to_A2)

        return shifted_p_elec_T.T, shifted_p_ion_T.T, shifted_total_T.T

    def get_polarization_change(self):
        shifted_p_elec, shifted_p_ion, shifted_total = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=True)
        return shifted_total[-1]-shifted_total[0]

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
    Given a list, shift it such that the points are closest to the preceeding point

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
