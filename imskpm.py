# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:26:28 2020

@author: Raj
"""

import numpy as np
from scipy.integrate import solve_ivp
from calc_utils import gen_t
from calc_utils import calc_n_pulse, calc_n_pulse_train, calc_gauss_volt, calc_omega0
from pulses import pulse
from odes import dn_dt_g

from matplotlib import pyplot as plt
import warnings

class IMSKPM:
    
    def __init__(self, 
                 intensity = 0.1,
                 k1 = 1e6,
                 k2 = 1e-10,
                 k3 = 0,
                 thickness = 500e-7):
    
        # Active layer parameters
        self.kinetics(k1, k2, k3, absorbance=1)
        self.thickness = thickness
        
        # Excitation parameters
        self.intensity = intensity #Incident intensity (W/cm^2, 0.1 = 1 Sun)
        self.exc_source(intensity, wl=455e-9, NA=0.6)
        self.use_pulse_train = False
        self.make_pulse()

        return
    
    def exc_source(self, intensity = None, wl = None, NA = None):
        '''Defines the incident excitation source light area+wavelength'''
        if intensity is not None:
            self.intensity = intensity
        if wl is not None:
            self.wavelength = wl
        if NA is not None:
            self.NA = NA
            
        self.area = 1e4 * np.pi * 0.5 * (0.61 * wl / NA)**2 # Rayleigh criterion in cm^2
        
        return
    
    def real_pulse(self, rise = 1e-4, fall = 1e-4):
        '''Adds a rise/fall time to excitation pulse'''
        self.square = False
        self.rise = rise
        self.fall = fall
        
        return
    
    def square_pulse(self):
        '''Sets pulse parameters to be ideal'''
        self.square = True
        self.rise = 0
        self.fall = 0
        
        return
    
    def make_pulse(self, rise = 0, fall = 0, pulse_time = 10e-3, 
                   start_time = 2.5e-3, pulse_width = 5e-3):
        '''Creates a single pulse'''
        if rise == 0 and fall == 0:
            
            self.rise = 0
            self.fall = 0
            self.square = True
        
        else:
            
            self.square = False
            self.rise = max(rise, 1e-14)
            self.fall = max(fall, 1e-14)
            
        self.pulse_time = pulse_time
        self.start_time = start_time
        self.pulse_width = pulse_width
        self.frequency = 1/pulse_time
        
        if self.start_time + self.pulse_width > self.pulse_time:
            
            warnings.warn('Pulse exceeds total_time, cropping width to match')
            self.pulse_width = self.pulse_time - self.start_time
        
        dt, tx = self._dt_tx(self.pulse_width, self.pulse_time)
        self.pulse = pulse(tx, self.start_time, self.pulse_width, 
                           self.intensity, self.rise, self.fall)
            
        return
    
    @staticmethod
    def _dt_tx(pulse_width, total_time):
        if pulse_width < 1e-6:
            dt = 1e-8
        else:
            dt = 1e-7 # save computation time
        
        tx = np.arange(0, total_time, dt)
        
        return dt, tx
    
    def pulse_train(self, total_time = None, max_cycles = None):
        '''
        
        Parameters
        ----------
        total_time : TYPE, optional
            DESCRIPTION. The default is None.
        max_cycles : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        AttributeError
            If missing both parameters
        '''
        if total_time is None and max_cycles is None:
            raise AttributeError('Must specify either total_time or max_cycles')
        elif total_time is not None and max_cycles is not None:
           warnings.warn('Specifying both defaults to using total_time')
        
        if total_time is None:
            self.pulse = np.tile(self.pulse, max_cycles)
        else:
            cycles = total_time // self.pulse_time
            self.pulse = np.tile(self.pulse, cycles)
        
        return
    
    def kinetics(self, k1, k2, k3, absorbance=None):
        '''
        k1 : float
            recombination, first-order (s^-1). The default is 1e6.
        k2 : float
            recombination, bimolecular (cm^3/s). The default is 1e-10.
        k3 : float
            recombination, Auger (cm^6/s). Default is 0
        absorbance : float, optional
            user-specified absorbance of the layer (a.u.)
        '''
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        if 'NoneType' not in str(type(absorbance)):
            self.absorbance = absorbance
        
        return

    def calc_n_dot(self):
        '''
        Calculating the integrated charge density given an input pulse
        
        Returns
        -------
        n_dens : float
            Char density in the film due to ODE (Generation - Recombination) (#/cm^3).
        sol :  `OdeSolution`
            (From Scipy) Found solution as `OdeSolution` instance
        gen : float
            Carrier concentration GENERATED (#/cm^3).
    
        '''
        
        gen = gen_t(self.absorbance, self.pulse, self.thickness) # electrons / cm^3 / s generated
        
        # Used for computational accuracy
        scale = 1e-4 #1 = /cm^3, 1e-4 = /um^3, 1e2 = /m^2
        gen = gen * scale**3 #(from /cm^3) 
        k1 = self.k1
        k2 = self.k2 / scale**3 #(from cm^3/s)
        k3 = self.k3 / scale**6 #(from cm^6/s)
        
        dt, tx = self._dt_tx(self.pulse_width, self.pulse_time)


        sol = solve_ivp(dn_dt_g, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                        args = (k1, k2, k3, gen, tx[1]-tx[0]))
        
        if not any(np.where(sol.y.flatten() > 0)[0]):                
    
            sol = solve_ivp(dn_dt_g, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                            args = (k1, k2, k3, gen, tx[1]-tx[0]), max_step=1e-6)
        
        n_dens = sol.y.flatten()
        
        gen = gen / scale**3
        n_dens = n_dens / scale**3
        
        return n_dens, sol, gen
    
    def simulate(self):
        
        n_dens, sol, gen = self.calc_n_dot()

        self.n_dens = n_dens
        self.sol = sol
        self.gen = gen
        
        self.voltage = calc_gauss_volt(self.n_dens)
        self.omega0 = calc_omega0(self.voltage)
    
        return
    
    def plot(self):
        
        fig, ax = plt.subplots(nrows=1,figsize=(6,4),facecolor='white')
        ax.plot(self.sol.t, self.voltage, 'g')
        ax.set_ylabel('Voltage (V)')
        ax.set_xlabel('Time (s)')
        vmean = self.voltage.mean() * np.ones(len(self.sol.t))
        ax.plot(self.sol.t, vmean, 'r--')
        ax.set_title(str(self.frequency) + ' Hz')
        plt.tight_layout()
            
        return