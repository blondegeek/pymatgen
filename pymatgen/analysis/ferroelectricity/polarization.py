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
from pymatgen.core.lattice import Lattice
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


# PBE ZVAL
PBE_ZVAL={'Ac':11,
'B':3,
'C_GW_new':4,
'Co':9,
'Er_2':8,
'Ga_d':13,
'H.58':0,
'Hf_sv_GW':12,
'Li':1,
'Mo':6,
'Nd_3':11,
'Os_sv_GW':16,
'Pt':10,
'Rh_pv':15,
'Se':6,
'Tc_pv':13,
'V_pv':11,
'Ag':11,
'B_GW':3,
'C_h':4,
'Co_GW':9,
'Er_3':9,
'Ga_d_GW':13,
'H.66':0,
'Hg':12,
'Li_AE_GW':3,
'Mo_pv':12,
'Ne':8,
'P':5,
'Pt_GW':10,
'Rh_pv_GW':15,
'Se_GW':6,
'Tc_sv':15,
'V_sv':13,
'Ag_GW':11,
'B_h':3,
'C_s':4,
'Co_pv':15,
'Eu':17,
'Ga_h':13,
'H.75':0,
'Ho':21,
'Li_GW':1,
'Mo_sv':14,
'Ne_GW':8,
'P_GW':5,
'Pt_pv':16,
'Rh_sv_GW':17,
'Si':4,
'Tc_sv_GW':15,
'V_sv_GW':13,
'Ag_pv':17,
'B_s':3,
'Ca_pv':8,
'Co_sv':17,
'Eu_2':8,
'Ga_pv_GW':19,
'H1.25':1,
'Ho_3':9,
'Li_sv':3,
'Mo_sv_GW':14,
'Ne_GW_soft':8,
'P_h':5,
'Pt_pv_GW':16,
'Rn':8,
'Si_GW':4,
'Te':6,
'W':6,
'Al':3,
'Ba_sv':10,
'Ca_sv':10,
'Co_sv_GW':17,
'Eu_3':9,
'Ga_sv_GW':21,
'H1.33':1,
'I':7,
'Li_sv_GW':3,
'N':5,
'Ni':10,
'Pa':13,
'Pt_sv_GW':18,
'Ru':8,
'Si_sv_GW':12,
'Te_GW':6,
'W_pv':12,
'Al_GW':3,
'Ba_sv_GW':10,
'Ca_sv_GW':10,
'Cr':6,
'F':7,
'Gd':18,
'H1.5':1,
'I_GW':7,
'Lu':25,
'N_GW':5,
'Ni_GW':10,
'Pa_s':11,
'Pu':16,
'Ru_pv':14,
'Sm':16,
'Th':12,
'W_sv_GW':14,
'Al_sv_GW':11,
'Be':2,
'Cd':12,
'Cr_pv':12,
'F_GW':7,
'Gd_3':9,
'H1.66':1,
'In':3,
'Lu_3':9,
'N_GW_new':5,
'Ni_pv':16,
'Pb':4,
'Pu_s':16,
'Ru_pv_GW':14,
'Sm_3':11,
'Th_s':10,
'Xe':8,
'Am':17,
'Be_GW':2,
'Cd_GW':12,
'Cr_sv':14,
'F_GW_new':7,
'Ge':4,
'H1.75':1,
'In_d':13,
'Mg':2,
'N_h':5,
'Ni_sv_GW':18,
'Pb_d':14,
'Ru_sv':16,
'Sn':4,
'Ti':4,
'Xe_GW':8,
'Ar':8,
'Be_sv':4,
'Cd_pv_GW':18,
'Cr_sv_GW':14,
'F_h':7,
'Ge_GW':4,
'H_AE':1,
'In_d_GW':13,
'Mg_GW':2,
'N_s':5,
'Np':15,
'Pb_d_GW':14,
'Ru_sv_GW':16,
'Sn_d':14,
'Ti_pv':10,
'Y_sv':11,
'Ar_GW':8,
'Be_sv_GW':4,
'Cd_sv_GW':20,
'Cs_sv':9,
'F_s':7,
'Ge_d':14,
'H_GW':1,
'Ir':9,
'Mg_pv':8,
'N_s_GW':5,
'Np_s':15,
'Pd':10,
'Ra_sv':10,
'S':6,
'Sn_d_GW':14,
'Ti_sv':12,
'Y_sv_GW':11,
'As':5,
'Bi':5,
'Ce':12,
'Cs_sv_GW':9,
'Fe':8,
'Ge_d_GW':14,
'H_h':1,
'Ir_sv_GW':17,
'Mg_pv_GW':8,
'Na':1,
'O':6,
'Pd_GW':10,
'Rb_pv':7,
'S_GW':6,
'Sr_sv':10,
'Ti_sv_GW':12,
'Yb':24,
'As_GW':5,
'Bi_GW':5,
'Ce_3':11,
'Cu':11,
'Fe_GW':8,
'Ge_h':14,
'H_h_GW':1,
'K_pv':7,
'Mg_sv':10,
'Na_pv':7,
'O_GW':6,
'Pd_pv':16,
'Rb_sv':9,
'S_h':6,
'Sr_sv_GW':10,
'Tl':3,
'Yb_2':8,
'As_d':15,
'Bi_d':15,
'Ce_GW':12,
'Cu_GW':11,
'Fe_pv':14,
'Ge_sv_GW':22,
'H_s':1,
'K_sv':9,
'Mg_sv_GW':10,
'Na_sv':9,
'O_GW_new':6,
'Pm':15,
'Rb_sv_GW':9,
'Sb':5,
'Ta':5,
'Tl_d':13,
'Zn':12,
'At':7,
'Bi_d_GW':15,
'Ce_h':12,
'Cu_pv':17,
'Fe_sv':16,
'H':1,
'He':2,
'K_sv_GW':9,
'Mn':7,
'Na_sv_GW':9,
'O_h':6,
'Pm_3':11,
'Re':7,
'Sb_GW':5,
'Ta_pv':11,
'Tm':23,
'Zn_GW':12,
'At_d':17,
'Br':7,
'Cl':7,
'Cu_pv_GW':17,
'Fe_sv_GW':16,
'H.25':0,
'He_GW':2,
'Kr':8,
'Mn_GW':7,
'Nb_pv':11,
'O_s':6,
'Po':6,
'Re_pv':13,
'Sb_d_GW':15,
'Ta_sv_GW':13,
'Tm_3':9,
'Zn_pv_GW':18,
'Au':11,
'Br_GW':7,
'Cl_GW':7,
'Dy':20,
'Fr_sv':9,
'H.33':0,
'Hf':4,
'Kr_GW':8,
'Mn_pv':13,
'Nb_sv':13,
'O_s_GW':6,
'Po_d':16,
'Re_sv_GW':15,
'Sc':3,
'Tb':19,
'U':14,
'Zn_sv_GW':20,
'Au_GW':11,
'C':4,
'Cl_h':7,
'Dy_3':9,
'Ga':3,
'H.42':0,
'Hf_pv':10,
'La':11,
'Mn_sv':15,
'Nb_sv_GW':13,
'Os':8,
'Pr':13,
'Rh':9,
'Sc_sv':11,
'Tb_3':9,
'U_s':14,
'Zr_sv':12,
'Au_pv_GW':17,
'C_GW':4,
'Cm':18,
'Er':22,
'Ga_GW':3,
'H.5':0,
'Hf_sv':12,
'La_s':9,
'Mn_sv_GW':15,
'Nd':14,
'Os_pv':14,
'Pr_3':11,
'Rh_GW':9,
'Sc_sv_GW':11,
'Tc':7,
'V':5,
'Zr_sv_GW':12}


def calc_ionic(pos, s, zval, center=None, tiny=0.001):
    """
    Function for calculating the ionic dipole moment for a site.

    pos (pymatgen.core.site.Site) : pymatgen Site
    s (pymatgen.core.structure.Structure) : pymatgen Structure
    zval (float) : number of core electrons of pseudopotential
    center (np.array with shape [3,1]) : dipole center used by VASP
    tiny (float) : tolerance for determining boundary of calculation.
    """
    # lattice vector lenghts
    norms = s.lattice.lengths_and_angles[0]
    # Define center of dipole moment. If not set default is used.
    center = np.array([-100.0,-100.0,-100.0]) if center == None else center
    # Following VASP dipol.F. SUBROUTINE POINT_CHARGE_DIPOL
    temp = (pos.frac_coords - center + 10.5) % 1 - 0.5
    for i in range(3):
        if abs(abs(temp[i]) - 0.5) < tiny/norms[i]:
            temp[i] = 0.0
    # Convert to Angstroms from fractional coords before returning.
    return np.dot(np.transpose(s.lattice.matrix), -temp*zval)


def get_total_ionic_dipole(structure, species_potcar_dict, pseudo_dict = None, center = None, tiny= 0.001):
    """
    Get the total ionic dipole moment for a structure.

    structure: pymatgen Structure
    species_potcar_dict: dictionary for species and pseudopotential name.
        Example {‘Li’ : ‘Li' , ’Nb’: ‘Nb_sv', ‘O’: ‘O’}
    pseudo_dict: default is PBE_ZVAL
    center (np.array with shape [3,1]) : dipole center used by VASP
    tiny (float) : tolerance for determining boundary of calculation.
    """

    pseudo_dict = PBE_ZVAL if pseudo_dict == None else pseudo_dict
    center = np.array([-100.0, -100.0, -100.0]) if center == None else center

    tot_ionic = []
    for site in structure:
        zval = pseudo_dict[species_potcar_dict[str(site.specie)]]
        tot_ionic.append(calc_ionic(site, structure, zval, center, tiny))
    return np.sum(tot_ionic,axis=0)

class Polarization(object):
    """
    Revised object for getting polarization
    """
    def __init__(self, p_elecs, p_ions, structures):
        if len(p_elecs) != len(p_ions) or len(p_elecs) != len(structures):
            raise ValueError("The number of electronic polarization and ionic polarization values must be equal.")
        self.p_elecs = np.matrix(p_elecs)
        self.p_ions = np.matrix(p_ions)
        self.structures = structures

    def get_pelecs_and_pions(self, convert_to_muC_per_cm2=False):

        if not convert_to_muC_per_cm2:
            return self.p_elecs, self.p_ions

        if convert_to_muC_per_cm2:
            p_elecs = np.matrix(self.p_elecs).T
            p_ions = np.matrix(self.p_ions).T

            volumes = [s.lattice.volume for s in self.structures]
            e_to_muC = -1.6021766e-13
            cm2_to_A2 = 1e16
            units = 1.0 / np.matrix(volumes)
            units *= e_to_muC * cm2_to_A2

            p_elecs = np.multiply(units, p_elecs)
            p_ions = np.multiply(units, p_ions)

            p_elecs, p_ions = p_elecs.T, p_ions.T

            return p_elecs, p_ions

    def get_same_branch_polarization_data(self, convert_to_muC_per_cm2=False):
        """
        Get same branch polarization for given polarization data.

        convert_to_muC_per_cm2: convert polarization from electron * Angstroms to microCoulomb per centimeter**2
        abc: return polarization in coordinates of a,b,c (versus x,y,z)
        """

        p_elec, p_ion = self.get_pelecs_and_pions()
        p_tot = p_elec + p_ion
        p_tot = np.matrix(p_tot)

        lattices = [s.lattice for s in self.structures]
        volumes = np.matrix([s.lattice.volume for s in self.structures])

        L = len(p_elec)

        # convert polarizations and lattice lengths prior to adjustment
        if convert_to_muC_per_cm2:
            e_to_muC = -1.6021766e-13
            cm2_to_A2 = 1e16
            units = 1.0 / np.matrix(volumes)
            units *= e_to_muC * cm2_to_A2
            # Convert the total polarization
            p_tot = np.multiply(units.T, p_tot)
            # adjust lattices
            for i in range(L):
                lattice = lattices[i]
                l,a = lattice.lengths_and_angles
                lattices[i] = Lattice.from_lengths_and_angles(np.array(l)*units.A1[i],a)

        d_structs = []
        sites = []

        for i in range(L):
            l = lattices[i]
            frac_coord = np.divide(np.matrix(p_tot[i]), np.matrix([l.a, l.b, l.c]))
            d = Structure(l, ["C"], [np.matrix(frac_coord).A1])
            d_structs.append(d)
            site = d[0]
            if i == 0:
                # Adjust nonpolar polarization to be closest to zero.
                # This is compatible with both a polarization of zero or a half quantum.
                prev_site = [0, 0, 0]
                # An alternative method which leaves the nonpolar polarization where it is.
                #sites.append(site)
                #continue
            else:
                prev_site = sites[-1].coords
            new_site = d.get_nearest_site(prev_site, site)
            sites.append(new_site[0])

        adjust_pol = []
        for s, d in zip(sites, d_structs):
            l = d.lattice
            adjust_pol.append(np.multiply(s.frac_coords, np.matrix([l.a, l.b, l.c])).A1)
        adjust_pol = np.matrix(adjust_pol)

        return adjust_pol

    def get_lattice_quanta(self, convert_to_muC_per_cm2 = True):
        """
        Returns the quanta along a, b, and c for all structures.
        """
        lattices = [s.lattice for s in self.structures]
        volumes = np.matrix([s.lattice.volume for s in self.structures])

        L = len(self.structures)

        # convert polarizations and lattice lengths prior to adjustment
        if convert_to_muC_per_cm2:
            e_to_muC = -1.6021766e-13
            cm2_to_A2 = 1e16
            units = 1.0 / np.matrix(volumes)
            units *= e_to_muC * cm2_to_A2
            # adjust lattices
            for i in range(L):
                lattice = lattices[i]
                l,a = lattice.lengths_and_angles
                lattices[i] = Lattice.from_lengths_and_angles(np.array(l)*units.A1[i],a)

        quanta = np.matrix([np.array(l.lengths_and_angles[0]) for l in lattices])

        return quanta

    def rms_norm_same_branch_polarization(self, tol=0.05):
        p_tot = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=True)
        mag = self.get_polarization_change().A1
        eps = 1e-8

        from scipy.optimize import curve_fit
        popt_a, pcov_a = curve_fit(line, range(p_tot.shape[0]), p_tot[:, 0].A1)
        popt_b, pcov_b = curve_fit(line, range(p_tot.shape[0]), p_tot[:, 1].A1)
        popt_c, pcov_c = curve_fit(line, range(p_tot.shape[0]), p_tot[:, 2].A1)

        rms_a = np.mean((p_tot[:, 0].A1 - line(np.array(range(p_tot.shape[0])), *popt_a)) ** 2) / abs(mag[0] + eps)
        rms_b = np.mean((p_tot[:, 1].A1 - line(np.array(range(p_tot.shape[0])), *popt_b)) ** 2) / abs(mag[1] + eps)
        rms_c = np.mean((p_tot[:, 2].A1 - line(np.array(range(p_tot.shape[0])), *popt_c)) ** 2) / abs(mag[2] + eps)

        def check_tol(x):
            if x < tol:
                return True
            return False

        return map(check_tol, [rms_a, rms_b, rms_c])

    def get_polarization_change(self):
        tot = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=True)
        return (tot[-1] - tot[0])

    def same_branch_splines(self):
        from scipy.interpolate import UnivariateSpline
        tot = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=True)
        L = tot.shape[0]
        try:
            sp_a = UnivariateSpline(range(L),tot[:,0].A1)
        except:
            sp_a = None
        try:
            sp_b = UnivariateSpline(range(L),tot[:,1].A1)
        except:
            sp_b = None
        try:
            sp_c = UnivariateSpline(range(L),tot[:,2].A1)
        except:
            sp_c = None
        return sp_a, sp_b, sp_c

    def smoothness(self):
        tot = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=True)
        L = tot.shape[0]
        try:
            sp = self.same_branch_splines()
        except:
            print("Something went wrong.")
            return None
        sp_latt = [sp[i](range(L)) for i in range(3)]
        diff = [sp_latt[i] - tot[:,i].A1 for i in range(3)]
        rms = [np.sqrt(np.sum(np.square(diff[i])) / L) for i in range(3)]
        #rms_mag_norm = [rms[i] / (max(tot[:,i].A1) - min(tot[:,i].A1)) for i in range(3)]
        return rms

    def is_smooth(self, rms_mag_norm_tol = 1e-2):
        """
        Returns whether spline fitted to adjusted a, b, c polarizations are smooth relative to a tolerance.
        """
        tot = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=True)
        L = tot.shape[0]
        try:
            sp = self.same_branch_splines()
        except:
            print("Something went wrong.")
            return None
        sp_latt = [sp[i](range(L)) for i in range(3)]
        diff = [sp_latt[i] - tot[:,i].A1 for i in range(3)]
        rms = [np.sqrt(np.sum(np.square(diff[i])) / L) for i in range(3)]
        rms_mag_norm = [rms[i] / (max(tot[:,i].A1) - min(tot[:,i].A1)) for i in range(3)]
        return [rms_mag_norm[i] <= rms_mag_norm_tol for i in range(3)]


class EnergyTrend(object):
    def __init__(self,energies):
        self.energies = energies

    def spline(self):
        from scipy.interpolate import UnivariateSpline
        sp = UnivariateSpline(range(len(self.energies)),self.energies, k=4)
        return sp

    def smoothness(self):
        energies = self.energies
        try:
            sp = self.spline()
        except:
            print("Energy spline failed.")
            return None
        spline_energies = sp(range(len(energies)))
        diff = spline_energies - energies
        rms = np.sqrt(np.sum(np.square(diff))/len(energies))
        #rms_mag_norm = rms / (max(energies) - min(energies))
        return rms

    def is_smooth(self, rms_mag_norm_tol = 1e-2):
        energies = self.energies
        try:
            sp = self.spline()
        except:
            print("Energy spline failed.")
            return None
        spline_energies = sp(range(len(energies)))
        diff = spline_energies - energies
        rms = np.sqrt(np.sum(np.square(diff))/len(energies))
        rms_mag_norm = rms / (max(energies) - min(energies))
        return rms_mag_norm <= rms_mag_norm_tol

    def endpoints_minima(self, slope_cutoff = 5e-3):
        energies = self.energies
        try:
            sp = self.spline()
        except:
            print("Energy spline failed.")
            return None
        der = sp.derivative()
        spline_energies = sp(range(len(energies)))
        der_energies = der(range(len(energies)))
        return {"polar" : abs(der_energies[-1]) <= slope_cutoff,
                "nonpolar" : abs(der_energies[0]) <= slope_cutoff}


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

    def get_same_branch_polarization_data(self, convert_to_muC_per_cm2=False, abc=True, factor=2):
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

        #factor = 2.0

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

    def rms_norm_same_branch_polarization(self, tol = 0.05):
        p_tot = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=True)
        mag = self.get_polarization_change().A1
        eps = 1e-8

        from scipy.optimize import curve_fit
        popt_a, pcov_a = curve_fit(line, range(p_tot.shape[0]), p_tot[:, 0].A1)
        popt_b, pcov_b = curve_fit(line, range(p_tot.shape[0]), p_tot[:, 1].A1)
        popt_c, pcov_c = curve_fit(line, range(p_tot.shape[0]), p_tot[:, 2].A1)

        rms_a = np.mean((p_tot[:, 0].A1 - line(np.array(range(p_tot.shape[0])), *popt_a)) ** 2) / abs(mag[0] + eps)
        rms_b = np.mean((p_tot[:, 1].A1 - line(np.array(range(p_tot.shape[0])), *popt_b)) ** 2) / abs(mag[1] + eps)
        rms_c = np.mean((p_tot[:, 2].A1 - line(np.array(range(p_tot.shape[0])), *popt_c)) ** 2) / abs(mag[2] + eps)

        def check_tol(x):
            if x < tol:
                return True
            return False

        return map(check_tol,[rms_a,rms_b,rms_c])

    def get_polarization_change(self):
        tot = self.get_same_branch_polarization_data(convert_to_muC_per_cm2=True)
        return (tot[-1] - tot[0])

def line(x, a, b):
    return a * x + b