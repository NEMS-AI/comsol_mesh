"""
Methods for unstructured tetrahedral meshes.

Includes utitilties for interpolation, integration and random point selection.
"""

import numpy as np

from textwrap import indent
from itertools import zip_longest
from collections import defaultdict
from scipy.spatial import KDTree
from .parsers import COMSOLObjects, COMSOLField


def is_broadcastable(shp1, shp2):
    """Return True if shapes are broadcastable as numpy arrays"""
    for a, b in zip(shp1[::-1], shp2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def broadcast_shape(shp1, sph2):
    """Return the shape of broadcast product of these arrays"""
    if not is_broadcastable(shp1, sph2):
        raise ValueError('Arrays are not broadcastable')
    return tuple(
        max(a, b) 
        for a, b in zip_longest(shp1, sph2, fillvalue=1)
    )


# ------------------------------------------------------------------------------
# Mesh
# ------------------------------------------------------------------------------
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
            tet_vol = abs(np.linalg.det(us) / 6)  # volume of tetrahedra
            vol_acc += tet_vol

        return vol_acc

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'n_points={self.n_points}, n_tetrahedra={self.n_tetrahedra})'
        )


# ------------------------------------------------------------------------------
# Surface
# ------------------------------------------------------------------------------
class Surface:
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
        
        # Compute triangle areas and normals
        tri_areas, tri_normals = self._triangle_properties(mesh, tri_indices)
        self.tri_areas = tri_areas
        self.tri_normals = tri_normals

    @property
    def n_triangles(self):
        return self.tri_indices.shape[0]
    
    @staticmethod
    def _triangle_properties(mesh, tri_indices):
        """Return areas and normals of surface triangles
        
        Parameters
        ----------
        mesh : Mesh
            unstructured tetrahedral mesh on which surface is defined
        tri_indices : (n_triangles, 3) int ndarray
            ordered indices of triangles

        Returns
        -------
        (areas, normals) : 
            areas : (n_triangles,) float ndarray
                array of triangle areas
            normals : (n_triangles, 3) float ndarray
                array of triangle normals
        """
        n_triangles = tri_indices.shape[0]
        areas = np.empty(n_triangles)
        normals = np.empty((n_triangles, 3))

        for i in range(n_triangles):
            j1, j2, j3 = tri_indices[i]
            p1 = mesh.points[j1, :]
            p2 = mesh.points[j2, :]
            p3 = mesh.points[j3, :]

            nn = np.cross(p2 - p1, p3 - p1)
            nn_norm = np.linalg.norm(nn)

            areas[i] = 0.5 * nn_norm
            normals[i, :] = nn / nn_norm
        return areas, normals

    def _random_triangle_idxs(self, n_samples):
        """Return the indices of random triangles weighted by area"""
        tri_areas = self.tri_areas
        total_area = np.sum(tri_areas)
        probs = tri_areas / total_area
        
        rand_tri_idxs = np.random.choice(self.n_triangles, size=n_samples, p=probs)
        return rand_tri_idxs

    def _random_triangle_points(self, n_samples):
        """Return random samples of points in canonical triangle
        
        A triangle is defined by 3 points (p₁, p₂, p₃) and points within the 
        triangle can be expressed uniquely as,

            p = p₁ξ₁ + p₂ξ₂+ p₃ξ₃
        
        where 0 ≤ ξ_i and ξ₁+ξ₂+ξ₃ = 1. This function returns random points
        within a triangle by yields random samples of (ξ₁, ξ₂, ξ₃).

        Parameters
        ----------
        n_samples : int
            number of samples

        Returns
        -------
        param_samples : (n_samples, 3)
            random samples of (ξ₁, ξ₂, ξ₃) sampled from uniform distribution
        """
        param_samples = np.empty((n_samples, 3))
    
        for i in range(n_samples):
            xi1, xi2 = np.random.uniform(size=2)
            if xi1 + xi2 < 1:
                param_samples[i, 0] = xi1
                param_samples[i, 1] = xi2
                param_samples[i, 2] = 1 - xi1 - xi2
            else:
                param_samples[i, 0] = 1 - xi1
                param_samples[i, 1] = 1 - xi2
                param_samples[i, 2] = xi1 + xi2 - 1
        
        return param_samples

    def random_point_sample(self, n_samples):
        """Return random points on surface weighted by area
        
        Parameters
        ----------
        n_samples : int
            number of samples

        Returns
        -------
        rand_points : (n_samples, 3) float ndarray
            random points on the surface
        """
        rand_points = np.empty((n_samples, 3))
        rand_tri_idxs = self._random_triangle_idxs(n_samples)
        rand_pt_params = self._random_triangle_points(n_samples)

        for i in range(n_samples):
            pt_indices = self.tri_indices[rand_tri_idxs[i]]
            ps = self.mesh.points[pt_indices, :]  # (3, 3) ndarray of points
            xis = rand_pt_params[i, :]   # (3,) ndarray of (ξ₁, ξ₂, ξ₃)
            rand_points[i, :] = xis @ ps

        return rand_points

    def random_value_sample(self, field, n_samples):
        """Return random values of field on surface weighted by area
        
        Parameters
        ----------
        n_samples : int
            number of samples

        Returns
        -------
        points, values :
            points : (n_samples, 3) float ndarray
                random points on the surface
            values : (n_samples, *field_shape) float ndarray
                values of the field at each random point
        """
        field_shape = field.field_shape
        
        rand_points = np.empty((n_samples, 3))
        rand_values = np.empty((n_samples, *field_shape))
        rand_tri_idxs = self._random_triangle_idxs(n_samples)
        rand_xis = self._random_triangle_points(n_samples)

        for i in range(n_samples):
            pt_indices = self.tri_indices[rand_tri_idxs[i]]
            ps = self.mesh.points[pt_indices, :]  # (3, 3) ndarray of points
            xis = rand_xis[i, :]   # (3,) ndarray of (ξ₁, ξ₂, ξ₃)
            rand_points[i, :] = xis @ ps
            rand_values[i, ...] = field.eval_surface(
                surface=self, 
                tri_idx=rand_tri_idxs[i],
                xis=rand_xis[i, :]
            )

        return rand_points, rand_values

    def points(self):
        """Return coordinates of points on surface
        
        Returns 
        -------
        points : (n_points, 3) float ndarray
            coordinates of points on the surface
        """
        point_idxs = np.unique(self.tri_indices)
        return self.mesh.points[point_idxs, :]
    
    def project(self, point):
        # TODO: Implement some utility for projecting points onto the mesh surface
        pass

    def __repr__(self):
        fields = ',\n'.join([
            f'mesh={self.mesh!r}',
            f'n_triangles={self.n_triangles}'
        ])

        return f'{self.__class__.__name__}(\n' + indent(fields, ' ' * 4) + '\n)'


def surfaces_from_comsol_obj(mesh, comsol_obj):
    """Return surfaces defined by COMSOL object
    
    A COMSOL object typically has several types which given as a list of dicts
    under the 'types' key. Examples of types are; 
        vtx - vertices (or points)
        edg - edges
        tri - triangles
        tet - tetrahedra
    
    The triangles (tri) type defines the surfaces of the COMSOL mesh object and
    this function converts this representation into a list of surfaces on the
    provided mesh object.
    
    Parameters
    ----------
    mesh : Mesh
        mesh to define surfaces on
    comsol_obj : dict
        dictionary defining a COMSOL object
        
    Returns
    -------
    surfaces : list[Surface]
        list of surfaces defined on the mesh
    """
    # Find first type with 'tri' type name
    tri_dict = [
        t_dict 
        for t_dict in comsol_obj['types'] 
        if t_dict['type_name'] == 'tri'
    ][0]
    
    # Build list of triangles associated with each geometry id
    geom_entry_idxs = tri_dict['geom_entry_idxs']
    geom_mapping = defaultdict(list)

    for i, geom_id in enumerate(geom_entry_idxs):
        geom_mapping[geom_id].append(i)

    # Build list of surfaces
    tri_indices_all = tri_dict['element_indices']
    lowest_vertex_id = comsol_obj['lowest_vertex_index']
    surfaces = []
    
    for geom_id in sorted(geom_mapping.keys()):
        tri_indices = tri_indices_all[geom_mapping[geom_id], :] - lowest_vertex_id
        s = Surface(mesh, tri_indices)
        surfaces.append(s)
        
    return surfaces

# ------------------------------------------------------------------------------
# Field
# ------------------------------------------------------------------------------
class Field:
    """Field over 3-dimensional tetrahedral mesh
    
    Attributes
    ----------
    mesh : Mesh
        mesh field is defined on
    values : (n_points, *field_shape) ndarray
        values of field at each mesh point where `n_points` is the number of
        points in the mesh. Field shape must be tuple at least length 1
    field_shape : tuple[int]
        shape of the field array at each mesh point (read only)
    """

    def __init__(self, mesh, values):
        """Return MeshField

        Parameters
        ----------
        mesh : Mesh
            mesh field is defined on
        values : (n_points, *field_shape) ndarray
            values of field at each mesh point where `n_points` is the number of
            points in the mesh. Field shape must be tuple at least length 1
            
        Raises
        ------
        ValueError : 
            - if value array has fewer than 2 dimensions
            - if number of values given does not match number of points in mesh
        """
        if len(values.shape) < 2:
            raise ValueError('Value array must have at least 2 dimensions')
        if values.shape[0] != mesh.n_points:
            raise ValueError(
                f'First axis of `values` array ({values.shape[0]}) should match'
                f' number of points in mesh ({mesh.n_points}).'
            )
        self.mesh = mesh
        self.values = values

    @property
    def field_shape(self):
        return self.values.shape[1:]
    
    def integrate(self):
        """Return integral of field over mesh volume
        
        Returns
        -------
        (*field_shape) ndarray
            integral of field over mesh volume
        """
        points = self.mesh.points
        tet_indices = self.mesh.tet_indices
        acc = np.zeros(self.field_shape)

        for i in range(self.mesh.n_tetrahedra):
            ps = points[tet_indices[i, :], :]  # (4, 3) ndarray of points
            us = ps[1:, :] - ps[0, :]  # (3, 3) ndarray of point displacements
            J = abs(np.linalg.det(us))  # Jacobian
            # (*field_shape,) ndarray of element linear integral
            el_int = J / 24 * np.sum(self.values[tet_indices[i, :], ...], axis=0)
            acc += el_int

        return acc

    def integrate_product(self, g):
        """Return integrate product of field and another field over mesh volume
        
        Parameters
        ----------
        g : MeshField
            another field over the same mesh, `field_shape` of field and g
            must be broadcastable 
        
        Raises
        ------
        ValueError :
            - if `g` is not defined on the same mesh object
            - if field shapes `self` and `g` are not broadcastable

        Returns
        -------
        : (*broadcast_shape) ndarray 
            intergral of the product of field and g. The shape of the integral
            result is the broadcast result of
        """
        if g.mesh != self.mesh:
            raise ValueError('g must be defined on the same mesh')
        if not is_broadcastable(g.field_shape, self.field_shape):
            raise ValueError(
                f'The field shapes of self and g must be broadcastable as numpy'
                f' arrays. {self.field_shape} and {g.field_shape} are not'
                f' broadcastable.'
            )

        points = self.mesh.points
        tet_indices = self.mesh.tet_indices
        acc = np.zeros(broadcast_shape(self.field_shape, g.field_shape))

        for i in range(self.mesh.n_tetrahedra):
            ps = points[tet_indices[i, :], :]  # (4, 3) ndarray of points
            us = ps[1:, :] - ps[0, :]  # (3, 3) ndarray of point displacements
            J = abs(np.linalg.det(us))  # Jacobian

            fs = self.values[tet_indices[i, :], ...]  # (4, self.field_shape)
            gs = g.values[tet_indices[i, :], ...]     # (4, g.field_shape)
            
            for i in range(4):
                for j in range(4):
                    if i <= j:
                        acc += J / 60 * fs[i, ...] * gs[j, ...]
        return acc

    def L2_norm(self):
        """Return the L2 norm
        
        The L2 norm of a field is defined as 
            
            L_2(f)^2 := ∫ | f(x) |^2 dV(x),

        where the integral is taken over the volume of the mesh.

        Return
        ------
        float :
            L2 norm of field over mesh
        """
        components = self.integrate_product(self)
        l2_sq = components.sum()
        l2_norm = np.sqrt(l2_sq)
        return l2_norm

    def eval_surface(self, surface, tri_idx, xis):
        """Return value of field at surface point
        
        Parameters
        ----------
        surface : Surface
            surface to evaluate field on
        tri_idx : int
            index of surface triangle containing evaluation point
        xis : (3,) float ndarray
            parameters of point in triangle (ξ₁, ξ₂, ξ₃)

        Returns
        -------
        value : (*field_shape) float ndarray
            value of field at evaluation point
        """
        xis = np.asanyarray(xis)
        pt_idxs = surface.tri_indices[tri_idx]  # (3,) ndarray indices of triangle points
        fs = self.values[pt_idxs, ...]  # (3, *field_shape) ndarray field values
        value = np.tensordot(xis, fs, 1)
        return value

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
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}(mesh={self.mesh!r},'
            f' field_shape={self.field_shape})'
        )


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