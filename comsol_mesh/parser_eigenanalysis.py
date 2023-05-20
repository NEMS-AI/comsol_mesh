"""
Scripts for parsing eigen-analysis outputs from COMSOL
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass


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
    
    @classmethod
    def from_csv(cls, path, mode_dim):
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
        return parse_eigenanalysis_csv(path, mode_dim)


def count_header_lines(path):
    """Return the number of header lines for COMSOL CSV file"""
    return len(parse_header_lines(path))


def parse_header_lines(path):
    """Return the header lines for a COMSOL CSV file without prefix"""
    header_lines = []
    
    with open(path) as f:
        for line in f.readlines():
            if line.startswith('%'):
                header_lines.append(line[2:])
                    
            else:
                break
        return header_lines


def parse_field_names(path):
    """Return field names for a COMSOL .csv file"""
    last_header_line = parse_header_lines(path)[-1]
    field_names = [
        entry.strip()
        for entry in last_header_line.split(',')
    ]
    return field_names


def _parse_csv(path):
    n_headers = count_header_lines(path)
    field_names = parse_field_names(path)
    
    df = pd.read_csv(path, skiprows=n_headers, header=None, names=field_names)
    return df
    
    
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
