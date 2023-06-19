"""
Test COMSOL mesh parser
"""
from comsol_mesh.parsers import *

from pprint import pprint

OBJ_TYPES = [
    {
        'element_indices': np.array([[0, 5, 4, 1],
                                [2, 4, 6, 1],
                                [5, 4, 1, 6],
                                [3, 5, 1, 6],
                                [8, 5, 6, 4]]),
        'geom_entry_idxs': [1, 1, 1, 1, 2],
        'n_elements': 5,
        'n_vertices_per_element': 4,
        'type_name': 'tet'
    },
    {
        'element_indices': np.array([[3, 5, 6, 7, 9, 10],
                                [8, 6, 5, 11, 10, 9]]),
        'geom_entry_idxs': [1, 2],
        'n_elements': 2,
        'n_vertices_per_element': 6,
        'type_name': 'prism'
    },
    {
        'element_indices': np.array([[0],
                                [1],
                                [2],
                                [4],
                                [5],
                                [6],
                                [7],
                                [9],
                                [10],
                                [11]]),
        'geom_entry_idxs': [0, 1, 3, 2, 4, 5, 7, 6, 9, 8],
        'n_elements': 10,
        'n_vertices_per_element': 1,
        'type_name': 'vtx'
    },
    {
        'element_indices': np.array([[ 0,  1],
                                [ 1,  2],
                                [ 1,  3],
                                [ 0,  4],
                                [ 4,  2],
                                [ 0,  5],
                                [ 4,  5],
                                [ 4,  6],
                                [ 2,  6],
                                [ 3,  7],
                                [ 4,  8],
                                [ 9,  7],
                                [ 5,  9],
                                [ 9, 10],
                                [ 7, 10],
                                [ 6, 10],
                                [ 8, 11],
                                [11, 10],
                                [ 9, 11]]),
        'geom_entry_idxs': [0, 3, 4, 1, 5, 2, 6, 7, 9, 4, 8, 12, 10, 14, 15, 11, 8, 16, 13],
        'n_elements': 19,
        'n_vertices_per_element': 2,
        'type_name': 'edg'
    },
    {
        'element_indices': np.array([[0, 1, 4],
                                [2, 4, 1],
                                [0, 5, 1],
                                [0, 4, 5],
                                [3, 1, 5],
                                [2, 6, 4],
                                [2, 1, 6],
                                [3, 6, 1],
                                [5, 6, 4],
                                [8, 5, 4],
                                [8, 4, 6],
                                [7, 9, 10],
                                [11, 10, 9]]),
        'geom_entry_idxs': [0, 0, 1, 2, 1, 4, 3, 3, 5, 6, 7, 8, 9],
        'n_elements': 13,
        'n_vertices_per_element': 3,
        'type_name': 'tri'
    },
    {
        'element_indices': np.array([[5, 9, 3, 7],
                                [3, 7, 6, 10],
                                [5, 9, 6, 10],
                                [5, 8, 9, 11],
                                [8, 6, 11, 10]]),
        'geom_entry_idxs': [1, 3, 5, 6, 7],
        'n_elements': 5,
        'n_vertices_per_element': 4,
        'type_name': 'quad'
    }
]


class TestCOMSOLParser:
    def test_parse_header(self):
        path = 'tests/data/mesh_example_intro.mphtxt'
        with open(path) as stream:
            parser = COMSOLFileParser(path, stream)
            header = parser.parse_header()
            
            assert set(header.keys()) == set(['version', 'tags', 'types'])
            assert header['version'] == (0, 1)
            assert header['tags'] == ['mesh1']
            assert header['types'] == ['obj']

    def test_parse_object(self):
        path = 'tests/data/mesh_example_intro.mphtxt'
        with open(path) as stream:
            # Read to start of object
            for i in range(11):
                stream.readline()

            parser = COMSOLFileParser(path, stream)
            obj = parser.parse_object()
            
            # Check header & vertices
            assert set(obj.keys()) == set([
                    'class', 'version', 'sdim', 'n_vertices', 
                    'lowest_vertex_index', 'vertices', 'types'
                    ])
            assert obj['class'] == 'Mesh'
            assert obj['version'] == 4
            assert obj['sdim'] == 3
            assert obj['n_vertices'] == 12
            assert obj['lowest_vertex_index'] == 0

            vertices_expect = np.array([
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [1, 1, 1],
                [2, 0, 1],
                [1, 1, 0],
                [2, 0, 0],
                [2, 1, 1],
                [2, 1, 0]
                ], 
                dtype=float
            )

            assert np.allclose(obj['vertices'], vertices_expect)

            for type_dict, expect in zip(obj['types'], OBJ_TYPES):
                for key in type_dict.keys():
                    if key == 'element_indices':
                        assert np.allclose(type_dict[key], expect[key])
                    else:
                        assert type_dict[key] == expect[key]

    def test_parser(self):
        path = 'tests/data/comsol_meshes/mesh_example_intro.mphtxt'
        obj = COMSOLFileParser.parse(path)
