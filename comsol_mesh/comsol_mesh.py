"""
Methods for importing meshes and analyses from COMSOL outputs
"""

from .interpolation import MeshInterp
from .parser_eigenanalysis import Eigenmodes
from .parsers import COMSOLMeshParser, COMSOLFile

class TetMesh:
    """Unstructured 3-dimensional tetrahedral mesh
    
    Parameters
    ----------
    vertices : (n_vertices, 3) ndarray
        vertices of tetrahedral mesh
    tetrahedra : (n_tetrahedra, 4) ndarray
        vertex indices of tetrahedra in mesh
    """
    def __init__(self, vertices, tetrahedra):
        self.vertices = vertices
        self.tetrahedra = tetrahedra

    @property
    def n_vertices(self):
        return self.vertices.shape[0]
    
    @property
    def n_tetrahedra(self):
        return self.tetrahedra.shape[0]
    
    @classmethod
    def from_comsol_obj(cls, comsol_obj):
        lowest_vertex_index = comsol_obj['lowest_vertex_index']
        vertices = comsol_obj['vertices']

        # Find tetrahedral object
        for type_dict in comsol_obj['types']:
            if type_dict['type_name'] == 'tet':
                tet_type_dict = type_dict
                break
        else:
            raise ValueError('No tetrahedral type in COMSOL object')

        element_indices = tet_type_dict['element_indices'] - lowest_vertex_index
        return cls(vertices, element_indices)
    

# STUB!!!!!!
class TetMeshFunction:
    """Function over 3-dimensional tetrahedral mesh"""

    def __init__(self, tet_mesh, values):
        self.tet_mesh = tet_mesh
        self.values = values

        # Instantiate interpolant

    @classmethod
    def from_point_array(cls, tet_mesh, values):
        pass

