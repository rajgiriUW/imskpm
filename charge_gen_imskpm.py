# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:26:28 2020

@author: Raj
"""

import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp

from matplotlib import pyplot as plt

'''
From David

Simulation of trEFM and  IM-SKPM responses:

(1) define generation rate (light intensity * absorption cross section)  G / cm^3/s

(2) define recombination kinetics (could be simple - k * N where k ~ 10^6 s-1, 
                                   or could add bimolecular terms - k2 * N**2 
                                   (but take k2 from our PL papers, e.g. Dane's Science)

(3) solve for N(t) given the G(t) function from the trEFM or IM-SKPM pulses.  
Here G(t) has a rise and fall slope that is given by the IRF of the source -- rather than being an ideal step.

(4) convert N(t) to  V(t) and d2C/dz^2  (V ~ log (N),  d2C/dz^2 as appropriate with charge density)

(5) convert V(t) and d2C/dz^2(t) to  signal(t) 
-- then apply ff-trEFM filter/processing, or lockin/FT and first harmonic extraction 
to simulate IM-SKPM and examine how IM-SKPM signal vs freq is affected by laser pulse rise time, 
and trEFM signal vs. laser pulse rise time affect things…compare to ideal square wave excitation
'''

_I = 100e-3 # 100 mw/cm^2
_wl = 455e-9
_area = 1e4 * np.pi * (0.61 * _wl / 1.2)**2 # Rayleigh criterion in cm^2

# NOTE: solve_ivp requires f(t, y, args) while odeint requires f(y,t,args)
# sol = solve_ivp(dn_dt2, [0, 10], [5], args=(0.5,0.9), t_eval=tx, method='LSODA')

def dn_dt(t, n, k1, k2):
    '''
    n : concentration
    k1 : monomolecular rate constant
    k2 : bimolecular rate constant
    '''
    
    ndot = -k1 * n - k2 * n**2
    return ndot

def gen_t(t):
    '''
    generation
    '''
    
    return 5*np.expm1(t)
