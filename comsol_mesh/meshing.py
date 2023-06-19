"""
Methods for unstructured tetrahedral meshes.

Includes utitilties for interpolation, integration and random point selection.
"""

import numpy as np

from scipy.spatial import KDTree

from .parsers import COMSOLObjects, COMSOLField


class Mesh:
    """Unstructured 3-dimensional tetrahedral mesh
    
    Parameters
    ----------
    points : (n_points, 3) ndarray
        vertices of tetrahedral mesh
    tet_indices : (n_tetrahedra, 4) int ndarray
        point indices of tetrahedra in mesh
    """
    def __init__(self, points, tet_indices):
        self.points = points
        self.tet_indices = tet_indices

        # Build KDTree for fast nearest point method
        self._pt_tree = KDTree(points)

    @property
    def n_points(self):
        return self.points.shape[0]
    
    @property
    def n_tetrahedra(self):
        return self.tet_indices.shape[0]
    
    @classmethod
    def from_comsol_obj(cls, comsol_obj):
        """Return mesh object using information from COMSOL object

        Parameters
        ----------
        comsol_obj : dict
            dictionary containing information about COMSOL object

        Returns
        -------
        Mesh
        """
        lowest_vertex_index = comsol_obj['lowest_vertex_index']
        points = comsol_obj['vertices']

        # Find tetrahedral object
        for type_dict in comsol_obj['types']:
            if type_dict['type_name'] == 'tet':
                tet_type_dict = type_dict
                break
        else:
            raise ValueError('No tetrahedral type in COMSOL mesh')

        tet_indices = tet_type_dict['element_indices'] - lowest_vertex_index
        return cls(points, tet_indices)
    
    def volume(self):
        """Return volume of mesh"""
        points = self.points
        tet_indices = self.tet_indices

        vol_acc = 0.0

        for i in range(self.n_tetrahedra):
            ps = points[tet_indices[i, :], :]  # (4, 3) ndarray of points
            us = ps[1:, :] - ps[0, :]          # (3, 3) ndarray of point displacements
            tet_vol = abs(0.5 * np.linalg.det(us))  # volume of tetrahedra
            vol_acc += tet_vol

        return vol_acc

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'n_points={self.n_points}, n_tetrahedra={self.n_tetrahedra})'
        )


class MeshSurface:
    """Surface of unstructured 3-dimensional mesh
    
    A surface is defined as a collection of triangles

    Parameters
    ----------
    mesh : Mesh
        unstructured tetrahedral mesh on which surface is defined
    tri_indices : (n_triangles, 3) int ndarray
        ordered indices of triangles
    """
    def __init__(self, mesh, tri_indices):
        self.mesh = mesh
        self.tri_indices = tri_indices
        
        self.tri_areas = self._triangle_areas(mesh, tri_indices)

    @staticmethod
    def _triangle_areas(self, mesh, tri_indices):
        """Return areas of surface triangles
        
        Parameters
        ----------
        mesh : Mesh
            unstructured tetrahedral mesh on which surface is defined
        tri_indices : (n_triangles, 3) int ndarray
            ordered indices of triangles

        Returns
        -------
        areas : (n_triangles,) float ndarray
            array of triangle areas
        """
        n_triangles = tri_indices.shape[0]
        areas = np.empty(n_triangles)

        for i in range(n_triangles):
            j1, j2, j3 = tri_indices[i]
            p1 = mesh.points[j1, :]
            p2 = mesh.points[j2, :]
            p3 = mesh.points[j3, :]

            area = 0.5 * np.cross(p3 - p1, p2 - p1)
            areas[i] = area
        return areas

    def random_triangle_sample(self, n_samples):
        """Return random sampling of triangle indices weighted by area
        
        Parameters
        ----------
        n_samples : int
            number of samples

        Returns
        -------
        (n_samples,) int ndarray
            indices of random triangles
        """
        # TODO: Implement
        pass

    
    def project(self, point):
        # TODO: Implement some utility for projecting points onto the mesh surface
        pass


class MeshField:
    """Field over 3-dimensional tetrahedral mesh"""

    def __init__(self, mesh, values):
        self.mesh = mesh
        self.values = values

    @classmethod
    def from_comsol_field(cls, mesh, comsol_field, tol=1e-10):
        """Return field over mesh from the values of a COMSOLField

        The values of the field for each point in the mesh is determined by
        taken the closest point in the comsol_field object.

        Parameters
        ----------
        comsol_field : COMSOLField
            field of values on a collection of points from CSV file
        tol : float
            maximum acceptable distance between a mesh point and closest field 
            point

        Raises
        ------
        ValueError :
            if the distance between a mesh point and the closest field point is
            above `tol`

        Returns
        -------
        MeshField
            mesh field
        """
        cf_pt_tree = KDTree(comsol_field.points)
        dists, idxs = cf_pt_tree.query(mesh.points)

        if np.any(dists > tol):
            raise ValueError(
                'Mesh point is too far from nearest COMSOLField point.'
                ' Exceeds tolerance!'
            )
        
        return cls(mesh=mesh, values=comsol_field.values[idxs])
    
    def integrate(self):
        """Return integral of field over mesh volume"""
        pass

    def integrate_product(self, g):
        """Return integrate product of field and another field over mesh volume"""
        pass

    def L2_norm(self):
        """Return the L2 norm
        
        The L2 norm of a field is defined as 
            
            L_2(f)^2 := ∫ | f(x) |^2 dV(x),

        where the integral is taken over the volume of the mesh.
        """
        pass
    


# ------------------------------------------------------------------------------
# Eliminate this code and turn this into a method for projecting points onto a
# surface.
# ------------------------------------------------------------------------------


def triangle_pcoordinates(p, ps):
    """Return triangle coordinates (xi1, xi2, xi3) for point p in triangle (p1, p2, p3)
    
    Parameters
    ----------
    p : (d,) ndarray
        point in triangle
    ps : (3, d) ndarray
        points as rows of the matrix
    
    Returns
    -------
    (3,) ndarray
        triangular coordinates
    """
    A = np.empty((3, 2))
    coords = np.empty(3)
    
    A[:, 0] = p1 - p3
    A[:, 1] = p2 - p3
    b = p - p3
    
    coords[:2] = np.linalg.lstsq(A, b, rcond=False)[0]
    coords[2] = 1 - coords[0] - coords[1]
    return coords


def triangle_interpolate(p, ps, us):
    """Return linear interpolation at point p given values at nodes
    
    Point is not projected onto the mesh
    
    Parameters
    ----------
    p : (d,) ndarray
        point in triangle
    ps : (3, d) ndarray
        points as rows of the matrix
    us : (3, df) ndarray
        values of function at triangle nodes
        
    Returns
    -------
    (df,) ndarray
        value of df-dimensional linear interpolation at point p
    """
    coords = np.linalg.lstsq(ps.T, p, rcond=False)[0]
    
    dims = np.ones(us.ndim, int)
    dims[0] = -1
    
    value = np.sum(coords.reshape(dims) * us, axis=0)
    return value
        
    
class MeshInterp:
    """Structure for linear interpolation over mesh points
    
    The values at each mesh point can be an arbitrary array
    
    Parameters
    ----------
    points : (n, d) ndarray
        points as rows of the matrix
    mesh_values : (n, *dfs) ndarray
        values of function at mesh nodes
    """
    
    def __init__(self, points, mesh_values):
        # Check Arguments
        
        if mesh_values.ndim < 2:
            raise ValueError('mesh_values should be at least 2-dimensional')
        
        n_mesh_values, *mesh_df = mesh_values.shape
        n_pts, pt_dim = points.shape
            
        if n_pts != n_mesh_values:
            raise ValueError("size of first dimension of mesh_values should match num. pts.")
        
        # Store data & build KDTree
        self._points = points
        self.tree = sc.spatial.KDTree(points)
        self.mesh_values = mesh_values
        self.mesh_df = mesh_df
        self.pt_dim = pt_dim
    
    def __call__(self, ps):
        """Return function on mesh at points ps
        
        Linear interpolation is used to compute the value of the function value 
        at the point.
        
        Parameters
        ----------
        ps : (m, pt_dim) ndarray 
            points to project and evaluate modes at
        
        Returns
        -------
        values : (m, *mesh_df) ndarray
            values of function on the mesh
        """
        
        # Check argument
        if ps.ndim != 2:
            raise ValueError('point array should have 2 dimensions')
        elif ps.shape[1] != self.pt_dim:
            raise ValueError(f'points should have dimension {self.pt_dim}')
        
        m, _ = ps.shape
        values = np.empty((m, *self.mesh_df))
        
        for i in range(m):
            p = ps[i, :]
            _, idxs = self.tree.query(p, k=3)
            
            mesh_pts = self._points[idxs, :]
            mesh_pts_us = self.mesh_values[idxs, ...]
            values[i, :] = triangle_interpolate(p, mesh_pts, mesh_pts_us)
        
        return values