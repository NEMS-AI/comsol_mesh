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
        assert mesh.volume() == approx(1 / 6)

        mesh = Mesh(2 * points, tet_indices)
        assert mesh.volume() == approx(1 / 6 * 2 ** 3)
        
        mesh = Mesh(-points, tet_indices)
        assert mesh.volume() == approx(1 / 6)

        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3]
        ])
        mesh = Mesh(points, tet_indices)
        assert mesh.volume() == approx(1)


class TestSurface:
    def test_areas(self):
        # Test MeshSurface
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

        tet_indices = np.array([]).reshape(0, 4)
        tri_indices = np.array([
            [0, 1, 2],
            [1, 3, 2]
        ])

        mesh = Mesh(points, tet_indices) 
        surf = Surface(mesh, tri_indices)
        assert surf.tri_areas == approx([0.5, 0.5])

    def test_normals(self):
        # Test MeshSurface
        points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])

        tet_indices = np.array([]).reshape(0, 4)
        tri_indices = np.array([
            [0, 1, 2],
            [1, 3, 2]
        ])

        mesh = Mesh(points, tet_indices) 
        surf = Surface(mesh, tri_indices)
        assert surf.tri_normals == approx(np.array([
                [ 0.,  0.,  1.],
          
                [ 0.,  0.,  1.]
        ])) 