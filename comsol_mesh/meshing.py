"""
Methods for unstructured tetrahedral meshes.

Includes utitilties for interpolation, integration and random point selection.
"""

from parsers import COMSOLFile, COMSOLField


class Mesh:
    """Unstructured 3-dimensional tetrahedral mesh
    
    Parameters
    ----------
    points : (n_points, 3) ndarray
        vertices of tetrahedral mesh
    tet_indices : (n_tetrahedra, 4) ndarray
        point indices of tetrahedra in mesh
    """
    def __init__(self, points, tet_indices):
        self.points = points
        self.tet_indices = tet_indices

    @property
    def n_points(self):
        return self.points.shape[0]
    
    @property
    def n_tetrahedra(self):
        return self.tet_indices.shape[0]
    
    @classmethod
    def from_comsol_file(cls, comsol_file: COMSOLFile):
        lowest_vertex_index = comsol_file['lowest_vertex_index']
        points = comsol_file['vertices']

        # Find tetrahedral object
        for type_dict in comsol_obj['types']:
            if type_dict['type_name'] == 'tet':
                tet_type_dict = type_dict
                break
        else:
            raise ValueError('No tetrahedral type in COMSOL object')

        tet_indices = tet_type_dict['element_indices'] - lowest_vertex_index
        return cls(points, tet_indices)


class MeshField:
    """Field over 3-dimensional tetrahedral mesh"""

    def __init__(self, mesh, values):
        self.mesh = mesh
        self.values = values

    @classmethod
    def from_comsol_field(cls, comsol_field: COMSOLField):
        values = comsol+
