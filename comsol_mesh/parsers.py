"""
Methods for parsing output from COMSOL files
"""
import re
import numpy as np
import pandas as pd

from textwrap import indent
from dataclasses import dataclass


class COMSOLObjects:
    """Collection of COMSOL mesh objects
    
    Attributes
    ----------
    header : dict
        dictionary of mesh information
    objects : list[dict]
        list of dictionaries described objects within the comsol mesh
    """
    def __init__(self, header, objects=[]):
        self.header = header
        self.objects = []

    def append_object(self, obj):
        self.objects.append(obj)

    @property
    def n_declared_objects(self):
        """Return number of objects declared in header"""
        return len(self.header['types'])

    @classmethod
    def from_file(cls, path):
        return COMSOLFileParser.parse(path)

    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, key):
        return self.objects[key]

    @staticmethod
    def obj_repr(obj_dict):
        n_vertices = obj_dict['n_vertices']
        type_reprs = [
            'COMSOLType(type_name={type_name}, n_elements={n_elements})'.format(**t)
            for t in obj_dict['types']
        ]

        obj_repr = (
            f'COMSOLObj(\n    n_vertices={n_vertices},\n    types=[\n'
            + ' ' * 8 + ('\n' + ' ' * 8).join(type_reprs) + '    \n    ]\n)\n'
        )
        return obj_repr

    def __repr__(self):
        obj_reprs = indent(',\n'.join(map(self.obj_repr, self.objects)), ' ' * 4)
        return f'{self.__class__.__name__}(\n{obj_reprs})'


class COMSOLFileParser:
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
            mesh = COMSOLObjects(header)
            
            for i in range(mesh.n_declared_objects):
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


@dataclass
class COMSOLField:
    """Container for fields on a collection of points

    A field is a vector or tensor quantity with shape `field_shape` defined on a
    collection of `N` points with dimension `dim_point`.

    Attributes
    ----------
    points : (N, dim_point) ndarray
        points which the eigenmodes are provided on
    values : (N, *field_shape) ndarray
        values of the eigenmode on the collection of points
    """
    points:np.ndarray
    values:np.ndarray

    @property
    def n_points(self):
        return self.points.shape[0]
    
    @classmethod
    def from_file(cls, path, *args, **kwargs):
        return COMSOLFieldParser.parse(path, *args, **kwargs)


class COMSOLEigenmodes(COMSOLField):
    @classmethod
    def from_file(cls, path, point_dim=3, mode_dim=3):
        """Return Eigenmodes from COMSOL eigen-analysis CSV file
        
        Parameters
        ----------
        path : str | pathlib.Path
            path to CSV file
        point_dim : int
            dimension of points the field is defined on
        mode_dim : int
            dimension of modes in eigen-analysis. For example the deflection of 
            a structure in a 3D analysis is 3-dimensional.
        
        Returns
        -------
        :COMSOLField
            modes of eigen-analysis
        """
        field = super().from_file(path, point_dim)
        field.values = field.values.reshape(field.n_points, -1, mode_dim)
        return field


class COMSOLFieldParser:
    """Parser for COMSOL field CSV exports"""

    def __init__(self):
        pass

    @classmethod
    def parse(cls, path, point_dim=3):
        """Parse COMSOL field CSV file
        
        Parameters
        ----------
        path : str | pathlib.Path
            path to CSV file
        point_dim : int
            dimension of points the field is defined on (default 3)

        Returns
        -------
        :COMSOLField
            field defined on a collection of points
        """
        field_df = cls._parse_CSV_to_dataframe(cls, path)
        points = np.array(field_df.iloc[:, 0:point_dim])
        values = np.array(field_df.iloc[:, point_dim:])
        return COMSOLField(points, values)
    
    @classmethod
    def _parse_CSV_to_dataframe(cls, path):
        """Return COMSOL CSV as dataframe"""
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

    @classmethod
    def _parse_header_lines(cls, path):
        """Return the header lines for a COMSOL CSV file without prefix"""
        header_lines = []

        with open(path) as f:
            for line in f.readlines():
                if line.startswith('%'):
                    header_lines.append(line[2:])
                        
                else:
                    break
            return header_lines
        