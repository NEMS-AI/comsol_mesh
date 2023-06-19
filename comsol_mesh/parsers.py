"""
Methods for parsing output from COMSOL files
"""
import re
import numpy as np
import pandas as pd

from dataclasses import dataclass


class COMSOLMesh:
    """Container for COMSOL mesh file header and objects"""
    def __init__(self, header, objects=[]):
        self.header = header
        self.objects = []

    def append_object(self, obj):
        self.objects.append(obj)

    @property
    def n_objects(self):
        return len(self.header['types'])
    
    def __repr__(self):
        return f'{self.__class__.__name__}(n_object={self.n_objects})'


class COMSOLMeshParser:
    """Parser for COMSOL mesh .mphtxt files

    Examples
    --------
    Parsing a file
    >>> path = 'example.mphtxt'
    >>> comsol_mesh = COMSOLParser.parse(path)
    """

    def __init__(self, path, stream):
        self.path = path
        self.stream = stream
        self.line_index = 0
        self.last_line = None

    @classmethod
    def parse(cls, path):
        with open(path) as stream:
            parser = cls(path, stream)
            header = parser.parse_header()
            mesh = COMSOLFile(header)
            
            for i in range(mesh.n_objects):
                obj = parser.parse_object()
                mesh.append_object(obj)
        
        return mesh

    def parse_header(self):
        """Parse header of COMSOL .mphtxt file"""
        stream = self.stream
        header = dict()

        # Parse version number
        try:
            line = self.next_line()
            v_major, v_minor = map(int, line.split())
            header['version'] = (v_major, v_minor)
        except:
            raise COMSOLParserError('HEADER: Improper version format', self)

        # Parse tags
        try:
            line = self.next_line()
            n_tags = int(re.match(r'(\d+).*', line).group(1))
            
            tags = []
            for i in range(n_tags):
                line = self.next_line()
                tag = re.match(r'\d+ (.*)', line).group(1)
                tags.append(tag)
            header['tags'] = tags
        except:
            raise COMSOLParserError(f'HEADER: Improper tag format on', self)
        
        # Parse object types
        try:
            line = self.next_line()
            n_types = int(re.match(r'(\d+).*', line).group(1))

            types = []
            for i in range(n_types):
                line = self.next_line()
                types.append(re.match(r'\d+ (.*)', line).group(1))

            header['types'] = types
        except:
            raise COMSOLParserError('HEADER: Improper type format', self)
        return header

    def parse_int(self):
        return int(re.match(r'(\d+)( #.*)?', self.next_line()).group(1))

    def parse_object(self):
        """Return object dict for object definition COMSOL .mphtxt file"""
        obj_dict = dict()

        # Check header for object
        line = self.next_line()
        if line != '0 0 1':
            raise COMSOLParserError('OBJ HEADER: Improper format', self)

        # Read class type
        try:
            obj_dict['class'] = re.match(r'\d+ (\w+) #.*', self.next_line()).group(1)
            obj_dict['version'] = self.parse_int()
            obj_dict['sdim'] = sdim = self.parse_int()
            obj_dict['n_vertices'] = n_vertices = self.parse_int()
            obj_dict['lowest_vertex_index'] = self.parse_int()
        except:
            raise COMSOLParserError('OBJ HEADER: Improper format', self)
        

        # Read mesh vertex coordinates
        vertices = np.empty((n_vertices, sdim), dtype=float)
        for i in range(n_vertices):
            line = self.next_line()
            pt = list(map(float, line.split()))
            if len(pt) != sdim:
                raise COMSOLParserError('VERTICES : Improper format', self)
            vertices[i, :] = pt
        
        obj_dict['vertices'] = vertices

        # Read number of element types
        n_types = self.parse_int()
        types = []
        
        for i in range(n_types):
            type_dict = self.parse_object_type()
            types.append(type_dict)

        obj_dict['types'] = types
        
        return obj_dict
    
    def parse_object_type(self):
        """Return dict for type definition in object from COMSOL file"""
        type_dict = dict()

        # Type header
        try:
            # Read type name
            line = self.next_line()
            type_dict['type_name'] = re.match(r'\d+ (\w+) #.*', line).group(1)
            type_dict['n_vertices_per_element'] = nvp_element = self.parse_int()
            type_dict['n_elements'] = n_elements = self.parse_int()
        except:
            raise COMSOLParserError('TYPE HEADER: Improper format', self)

        # Type element vertex indices
        try:
            element_indices = np.empty((n_elements, nvp_element), int)
            for i in range(n_elements):
                line = self.next_line()
                idxs = list(map(int, line.split()))
                if len(idxs) != nvp_element:
                    raise COMSOLParserError(
                        'TYPE ELEMENT INDICES: Incorrect number of indices', self
                    )
                element_indices[i, :] = idxs

            type_dict['element_indices'] = element_indices
        except:
            raise COMSOLParserError('TYPE ELEMENT INDICES: Improper format', self)
    
        # Parse geometric entity indices
        try:
            n_geom_entity_idxs = self.parse_int()
            geom_entity_idxs = []
            for i in range(n_geom_entity_idxs):
                geom_entity_idxs.append(self.parse_int())
            type_dict['geom_entry_idxs'] = geom_entity_idxs

        except:
            raise COMSOLParserError('TYPE ELEMENT: Geometry entity indices', self)
        
        return type_dict

    def next_line(self):
        """Return next non-blank, non-comment line"""
        while True:
            line = self.stream.readline()
            self.line_index += 1

            if line.startswith('#') or line == '\n':
                continue
            elif line == '':
                raise EOFError()
            else:
                line = line.strip()
                self.last_line = line
                return line


class COMSOLParserError(Exception):
    """Error raised for invalid COMSOL file format"""
    def __init__(self, comment, parser=None):
        if parser is not None:
            message = f'{comment}\n@ line {parser.line_index}: \'{parser.last_line}\''
        else:
            message = comment

        super().__init__(message)


class COMSOLFieldsParser:
    """Parser for COMSOL fields CSV exports"""

    def __init__(self):
        pass

    @classmethod
    def parse(cls, path):
        header_lines = cls._parse_header_lines(path)
        
        last_header_line = parse_header_lines(path)[-1]
        field_names = [
            entry.strip()
            for entry in last_header_line.split(',')
        ]

        field_dataframe = pd.read_csv(
            path, 
            skiprows=len(header_lines), 
            header=None, 
            names=field_names
        )
        return field_dataframe

    @staticmethod
    def _parse_header_lines(path):
        """Return the header lines for a COMSOL CSV file without prefix"""
        header_lines = []

        with open(path) as f:
            for line in f.readlines():
                if line.startswith('%'):
                    header_lines.append(line[2:])
                        
                else:
                    break
            return header_lines
    

@dataclass
class Eigenmodes:
    """Container for eigen-analysis modes

    The eigenmodes are defined on an array of `N` points with dimension `pt_dim`,
    where each mode has dimension `mode_dim`.

    Parameters
    ----------
    pts : (N, pt_dim) ndarray
        points which the eigenmodes are provided on
    modes : (N, mode_dim) ndarray
        values of the eigenmode on the collection of points
    """
    pts:np.ndarray
    modes:np.ndarray

    @property
    def n_pts(self):
        return self.pts.shape[0]
    
    @property
    def n_modes(self):
        return self.modes.shape[1]
    
    @property
    def mode_dim(self):
        return self.modes.shape[2]


class EigenmodesParser(COMSOLFieldsParser):
    pass
    
    
def parse_eigenanalysis_csv(path, mode_dim=3):
    """Return Eigenmodes from COMSOL eigen-analysis CSV file
    
    Parameters
    ----------
    path : str | pathlib.Path
        path to CSV file
    mode_dim : int
        dimension of modes in eigen-analysis. For example the deflection of a
        structure in a 3D analysis is 3-dimensional.
    
    Returns
    -------
    :Eigenmodes
        modes of eigen-analysis
    """
    df = _parse_csv(path)
    
    pt_dim = 3
    n_fields = len(df.columns)
    n_pts = len(df)
    
    n_modes = (n_fields - pt_dim) // mode_dim
    if n_modes * mode_dim + pt_dim != n_fields:
        raise ValueError('Field number mismatch!')
        
    pts = np.array(df.iloc[:, 0:pt_dim])
    modes = np.empty((n_pts, n_modes, mode_dim))
    
    for i in range(n_modes):
        m_idx = pt_dim + mode_dim * i
        modes[:, i, :] = np.array(df.iloc[:, m_idx:m_idx + mode_dim])
    
    result = Eigenmodes(pts, modes)
    return result
