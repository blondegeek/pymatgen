# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

from __future__ import division, unicode_literals
from itertools import combinations_with_replacement

"""
This module implements symmetry-related structure forms.
"""


__author__ = "Shyue Ping Ong"
__copyright__ = "Copyright 2012, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Shyue Ping Ong"
__email__ = "shyuep@gmail.com"
__date__ = "Mar 9, 2012"

import numpy as np
from pymatgen.core.structure import Structure


class SymmetrizedStructure(Structure):
    """
    This class represents a symmetrized structure, i.e. a structure
    where the spacegroup and symmetry operations are defined. This class is
    typically not called but instead is typically obtained by calling
    pymatgen.symmetry.analyzer.SpacegroupAnalyzer.get_symmetrized_structure.

    Args:
        structure (Structure): Original structure
        spacegroup (Spacegroup): An input spacegroup from SpacegroupAnalyzer.
        equivalent_positions: Equivalent positions from SpacegroupAnalyzer.

    .. attribute: equivalent_indices

        indices of structure grouped by equivalency
    """

    def __init__(self, structure, spacegroup, equivalent_positions):
        super(SymmetrizedStructure, self).__init__(
            structure.lattice, [site.species_and_occu for site in structure],
            structure.frac_coords, site_properties=structure.site_properties)
        self._spacegroup = spacegroup
        u, inv = np.unique(equivalent_positions, return_inverse=True)
        self.site_labels = equivalent_positions
        self.equivalent_indices = [[] for i in range(len(u))]
        self._equivalent_sites = [[] for i in range(len(u))]
        for i, inv in enumerate(inv):
            self.equivalent_indices[inv].append(i)
            self._equivalent_sites[inv].append(self.sites[i])

    @property
    def spacegroup(self):
        return self._spacegroup

    @property
    def equivalent_sites(self):
        """
        All the sites grouped by symmetry equivalence in the form of [[sites
        in group1], [sites in group2], ...]
        """
        return self._equivalent_sites

    def find_equivalent_sites(self, site):
        """
        Finds all symmetrically equivalent sites for a particular site

        Args:
            site (PeriodicSite): A site in the structure

        Returns:
            ([PeriodicSite]): List of all symmetrically equivalent sites.
        """
        for sites in self.equivalent_sites:
            if site in sites:
                return sites

        raise ValueError("Site not in structure")

    def get_equiv_by_specie(self):
        """
        Returns dictionary with { specie : list of equiv positions}

        Note, that this will also work for structures with oxidation states.
        However, the oxidation state will be part of the index string. Use the
        remove_oxidation_state method on on the SymmetrizedStructure object
        and then use this method if you would prefer to not have oxidation
        states included.

        Returns:
            { specie : [equiv_index, ...] , ...} 
        """

        equiv_by_specie = {}
        for i in self.composition.elements:
            equiv_by_specie.update({str(i):[]})

        for i in set(self.site_labels):
            site = self[i]
            equiv_by_specie[str(site.specie)].append(i)

        return equiv_by_specie

    def get_specie_labels(self):
        """
        Labels site by element and equivalent site number. Useful for generating
        traditional crystal structure tables.

        The index of the equivalent position for a given element 
        name gives atom number. For equiv_by_specie,
              {'Ir':[0,4,6],'O':[8,10]} 
        get_specie_labels would return atom labels
              { 0 : 'Ir1', 4 : 'Ir2', 6 : 'Ir3', 8 : 'O1', 10 : 'O2'}

        Note, that this will also work for structures with oxidation states.
        However, the oxidation state will be part of the string. Use the
        remove_oxidation_state method on on the SymmetrizedStructure object
        and then use this method if you would prefer to not have oxidation
        states included.

        Returns:
            { equiv_site : label, ...}
        """

        equiv_by_specie = self.get_equiv_by_specie()

        specie_labels = {}
        for i in set(self.site_labels):
            label = str(self[i].specie)
            label += str(equiv_by_specie[str(self[i].specie)].index(i)+1)
            specie_labels.update({i : label})

        return specie_labels

    def get_distinct_bonds(self,r,dr,sigfig=3):
        """
        Returns symmetrically distinct bond lengths up to a certain radius.

        Args:
            r (float): Inner radius of shell.
            dr (float): Width of shell.
            sigfig (int):
                Significant figures to distinguishing between bond lengths.
        """

        equiv_by_specie = self.get_equiv_by_specie
        specie_labels = self.get_specie_labels()

        # Get all pairwise species combinations
        species = map(str,self.composition.elements)
        #bonds = set(combinations_with_replacement(species,2))

        bonds = set()

        for index in set(self.site_labels):
            neigh = self.get_neighbors_in_shell(self[index].coords,r,dr,
                                             include_index=True)
            for n in neigh:
                equiv = tuple(sorted([index,self.site_labels[n[2]]]))
                dist = unicode.format('{0:.'+unicode(sigfig)+'f}',
                                      round(n[1],sigfig))
                bonds.update(((equiv,dist)))
        return bonds

#    def get_distinct_angles(self):
