
from __future__ import division, unicode_literals, print_function

from pymatgen.util.testing import PymatgenTest
from pymatgen.core.structure import IStructure, Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.structure import SymmetrizedStructure
import random
import warnings
import os

class SymmetrizedStructureTest(PymatgenTest):

    def setUp(self):
        coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
        self.lattice = Lattice([[3.8401979337, 0.00, 0.00],
                                [1.9200989668, 3.3257101909, 0.00],
                                [0.00, -2.2171384943, 3.1355090603]])
        self.struct = Structure(self.lattice, ["Si"] * 2, coords)
        self.symmstruct = SpacegroupAnalyzer(self.struct).get_symmetrized_structure()

    def test_distinct_bonds(self):
        bonds = self.symmstruct.get_distinct_bonds(2,1)
        self.assertEqual(bonds,set([(('Si1', 'Si1'), u'2.352')]))

    def test_distinct_angles(self):
        angles = self.symmstruct.get_distinct_angles(2,1)
        self.assertEqual(angles,set([(('Si1', 'Si1'), 'Si1', u'109.471')]))

if __name__ == '__main__':
    import unittest2 as unittest
    unittest.main()
