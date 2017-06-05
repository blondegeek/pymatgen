# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals, print_function
import logging
import math
import itertools
import warnings
from collections import OrderedDict

import numpy as np
from matplotlib import patches
import matplotlib.pyplot as plt

from monty.json import jsanitize

from pymatgen.util.plotting import pretty_plot, \
    add_fig_kwargs, get_ax3d_fig_plt


"""
This module implements the plotter for Polarization and EnergyTrend.
"""

__author__ = "Tess Smidt"
__copyright__ = "Copyright 2017, The Materials Project"
__version__ = "0.1"
__email__ = "blondegeek@gmail.com"
__date__ = "May 27, 2017"


logger = logging.getLogger(__name__)

class PolarizationPlotter(object):
    def __init__(self, polarization):
        self.polarization = polarization

    def get_plot(self, with_quanta=True, abs_ylim_min=5, convert_to_muC_per_cm2=True):
        # Get polarization data
        same_branch = self.polarization.get_same_branch_polarization_data(convert_to_muC_per_cm2=convert_to_muC_per_cm2)
        quanta = self.polarization.get_lattice_quanta(convert_to_muC_per_cm2=convert_to_muC_per_cm2)
        splines = self.polarization.same_branch_splines(convert_to_muC_per_cm2=True)

        # Set plot preferences
        plt.rc('xtick', labelsize=24)
        plt.rc('ytick', labelsize=24)
        try:
            plt.rc('text', usetex=True)
        except:
            # Fall back on non Tex if errored.
            plt.rc('text', usetex=False)

        font = {'family': 'normal',
                'weight': 'bold',
                'size': 30}
        plt.rc('font', **font)

        # Get figure and axes
        fig = plt.figure(figsize=(20, 6))

        axes = [plt.subplot(1, 3, i) for i in range(1, 4)]

        lattice = "abc"

        for i,ax in enumerate(axes):
            # Make symmetric axis range around zero.
            ax_max = max(max(same_branch[:,i].A1),  abs_ylim_min)
            ax_min = min(min(same_branch[:,i].A1), -abs_ylim_min)
            ax_range = max(abs(ax_min),ax_max)
            ax.set_ylim((-ax_range, ax_range))
            ax.set_xlim((0, len(same_branch[:,i])-1))

            # Make labels
            if i==0:
                if convert_to_muC_per_cm2:
                    ax.set_ylabel('$\mu C / cm^2$ along a, b, and c')
                else:
                    ax.set_ylabel('$e \AA$ along a, b, and c')
            ax.set_xlabel('Nonpolar to Polar')

            # Plot same branch polarization with quanta
            num_copies_quanta = 2 * int(np.ceil((ax_max - ax_min)/quanta[0,i]))
            for j in range(-num_copies_quanta, num_copies_quanta+1):
                color = 'ro' if j==0 else 'bo'
                ax.plot(same_branch[:,i]+j*quanta[:,i],color)

            # Plot same branch polarization spline
            if splines[i]:
                xs = np.linspace(0, len(same_branch[:,i])-1, 1000)
                ax.plot(xs, splines[i](xs),'r')

        return plt

    def show(self, with_quanta=True, abs_ylim_min=5, convert_to_muC_per_cm2=True):
        plt = self.get_plot(with_quanta=with_quanta,
                            abs_ylim_min=abs_ylim_min,
                            convert_to_muC_per_cm2=convert_to_muC_per_cm2)
        plt.show()

    def save_plot(self, filename, img_format="eps", with_quanta=True,
                  abs_ylim_min=5, convert_to_muC_per_cm2=True):
        plt = self.get_plot(with_quanta=with_quanta,
                            abs_ylim_min=abs_ylim_min,
                            convert_to_muC_per_cm2=convert_to_muC_per_cm2)
        plt.savefig(filename, format=img_format)


# For misc trends across the ferroelectric distortion
class TrendPlotter(object):
    """
    Plot single or multiple trends across distortion.
    """
    def __init__(self,data, group_dict=None, splines=None):
        self.data = np.array(data)
        self.group_dict =  group_dict if group_dict is not None else {}
        self.splines = splines

    def _make_ticks(self):
        # Remove numbers of interpolation?
        pass

    def get_plot(self, xlabel='Nonpolar to Polar', ylabel='Energy (eV per atom)', abs_ylim_min=5):
        try:
            plt.rc('text', usetex=True)
        except:
            # Fall back on non Tex if errored.
            plt.rc('text', usetex=False)

        plt.xlabel(xlabel, fontsize=24)
        plt.ylabel(ylabel, fontsize=24)

        # ax_max = max(np.max(self.data), abs_ylim_min)
        # ax_min = min(np.min(self.data), -abs_ylim_min)
        # ax_range = max(abs(ax_min), ax_max)
        # plt.ylim((-ax_range, ax_range))

        if len(self.data.shape) == 2:
            for i in range(self.data.shape[0]):
                color = 'bo'
                if self.group_dict.get(i,None):
                    color = self.group_dict[i].get('color','bo')
                plt.plot(self.data[i], color)
        elif len(self.data.shape) == 1:
            color = 'bo'
            plt.plot(self.data, color)

        return plt

    def show(self, xlabel='Nonpolar to Polar', ylabel='Energy (eV per atom)'):
        plt = self.get_plot(xlabel=xlabel, ylabel=ylabel)
        plt.show()

    def save_plot(self, filename, img_format="eps", xlabel='Nonpolar to Polar',
                  ylabel='Energy (eV per atom)'):
        plt = self.get_plot(xlabel=xlabel, ylabel=ylabel)
        plt.savefig(filename, format=img_format)