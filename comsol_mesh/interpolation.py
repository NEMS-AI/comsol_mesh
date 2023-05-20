"""
Methods for interpolating and performing basic quadrature over unstructured meshes
"""

import numpy as np
import scipy as sc


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