"""
Test meshing
"""

import pytest

from pytest import approx

from comsol_mesh.meshing import *


class TestMesh:
    def test_volume(self):
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        tet_indices = np.array([[0, 1, 2, 3]])
        mesh = Mesh(points, tet_indices)
        assert mesh.volume() == approx(0.5)

        mesh = Mesh(2 * points, tet_indices)
        assert mesh.volume() == approx(0.5 * 2 ** 3)
        
        mesh = Mesh(-points, tet_indices)
        assert mesh.volume() == approx(0.5)

        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ])
        mesh = Mesh(points, tet_indices)
        assert mesh.volume() == approx(3)