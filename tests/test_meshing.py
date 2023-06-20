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

    def test_cube_volume(self):
        # Test cube integration
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ])
        tet_indices = np.array([
            [0, 1, 2, 4],
            [2, 4, 5, 6],
            [1, 2, 4, 5],
            [1, 2, 3, 7],
            [2, 5, 6, 7],
            [1, 2, 5, 7]
        ])
        cube_mesh = Mesh(points, tet_indices)
        assert cube_mesh.volume() == approx(1.0)


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


class TestField:
    def test_integrate(self):
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        tet_indices = np.array([[0, 1, 2, 3]])
        mesh = Mesh(points, tet_indices)

        # Test uniform
        values = np.array([1, 1, 1, 1]).reshape(4, 1)
        field = Field(mesh, values)
        assert field.integrate() == approx(1 / 6)

        # Test asymmetric
        values = np.array([1, -1, 6, 2]).reshape(4, 1)
        field = Field(mesh, values)
        assert field.integrate() == approx(1 / 3)

        # Test (2,) shape
        values = np.array([
            [1, 1],
            [1, -1],
            [1, 6],
            [1, 2]
        ])
        field = Field(mesh, values)
        assert field.integrate() == approx(np.array([1 / 6, 1 / 3]))

    def test_integrate_product(self):
        # Test 1: Test scalar
        # ----------------------------------------------------------------------
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        tet_indices = np.array([[0, 1, 2, 3]])
        mesh = Mesh(points, tet_indices)

        values1 = np.array([1, 1, 1, 1]).reshape(4, 1)
        field1 = Field(mesh, values1)

        values2 = np.array([1, 1, 1, 1]).reshape(4, 1)
        field2 = Field(mesh, values2)
        assert field1.integrate_product(field2) == approx(1 / 6)

        # Test 2: Test vector
        # ----------------------------------------------------------------------
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        tet_indices = np.array([[0, 1, 2, 3]])
        mesh = Mesh(points, tet_indices)

        values = np.array([1, 2, -1, 6]).reshape(4, 1)
        field = Field(mesh, values)
        assert field.integrate_product(field) == approx(53 / 60)

        # Test 3: Test vector
        # ----------------------------------------------------------------------
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
        tet_indices = np.array([[0, 1, 2, 3]])
        mesh = Mesh(points, tet_indices)

        values = np.array([
            [1, 1], [1, 2], [1, -1], [1, 6]
        ])
        field = Field(mesh, values)
        assert field.integrate_product(field) == approx(np.array([1/6, 53/60]))
        
    def test_eval_surface(self):
        # Define mesh
        points = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ])
        tet_indices = np.array([]).reshape((0, 4))
        mesh = Mesh(points, tet_indices)

        # Define surface
        tri_indices = np.array([[0, 1, 2]])
        surface = Surface(mesh, tri_indices)

        # Define field
        values = np.array([1, 1, 1]).reshape((3, 1))
        field = Field(mesh, values)

        xis_set = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([1/3, 1/3, 1/3])
        ]

        for xis in xis_set:
            assert field.eval_surface(surface, 0, xis) == approx(np.array([1.0]))
            
        # Test f(x, y) = x + 2y 
        values = np.array([0, 1, 2]).reshape((3, 1))
        field = Field(mesh, values)
        expected = [0.0, 1.0, 2.0, 1.0]

        for (xis, expect) in zip(xis_set, expected):
            assert field.eval_surface(surface, 0, xis) == approx(np.array([expect]))
    
