# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 21:06:04 2022

@author: Raj
"""

import numpy as np

# ODEs
def dn_dt(t, n, k1, k2, k3=0):
    '''
    Recombination equation, assuming no auger
    
    Parameters
    ----------
    t : float
        Time (s)
    n : float
        carrier density (cm^-3).
    k1 : float
        recombination, first-order (s^-1)
    k2 : float
        recombination, bimolecular (cm^3/s)
    k3 : float
        recombination, Auger (cm^6/s). Default is 0

    Returns
    -------
    n_dot : float
        differential carrier concentration w.r.t. time.

    '''
    n_dot = -k1 * n - k2 * n**2 - k3 * n**3
    
    return n_dot

def dn_dt_g(t, n, k1, k2, pulse, dt):
    '''
    Recombination+Generation equation, assuming no auger
    
    Calculates the generation at a particular time given a light pulse array
    
    Parameters
    ----------
    t : float
        Time (s)
    n : float
        carrier density (cm^-3).
    k1 : float
        recombination, first-order (s^-1)
    k2 : float
        recombination, bimolecular (cm^3/s)
    pulse : ndarray
        The array describing the pulse being applied to the sample.
    dt : float
        time spacing (s).

    Returns
    -------
    n_dot : float
        differential carrier concentration w.r.t. time.

    '''
    tidx = min(int(np.floor(t / dt), len(pulse)-1))
    g = pulse[tidx]
    
    n_dot = g - k1 * n - k2 * n**2

    return n_dot