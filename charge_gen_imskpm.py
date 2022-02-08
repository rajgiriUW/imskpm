# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:26:28 2020

@author: Raj
"""

import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from calc_utils import calc_n_pulse, calc_n_pulse_train, calc_gauss_volt, calc_omega0

from matplotlib import pyplot as plt
import warnings

class IMSKPM:
    
    def __init__(self, 
                 intensity = 0.1,
                 k1 = 1e6,
                 k2 = 1e-10,
                 k3 = 0):
    
        # Active layer parameters
        self.kinetics(k1, k2, k3, absorbance=1)
        
        # Excitation parameters
        self.intensity = intensity #Incident intensity (W/cm^2, 0.1 = 1 Sun)
        self.exc_source(intensity, wl=455e-9, NA=0.6)
        self.use_pulse_train = False
        self.make_pulse()

        return
    
    def exc_source(self, intensity = None, wl = None, NA = None):
        '''Defines the incident excitation source light area+wavelength'''
        if 'NoneType' not in str(type(intensity)):
            self.intensity = intensity
        if 'NoneType' not in str(type(wl)):
            self.wavelength = wl
        if 'NoneType' not in str(type(NA)):
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
        
        self.square = True
        self.rise = 0
        self.fall = 0
        
        return
    
    def make_pulse(self, rise = 0, fall = 0, total_time = 10e-3, 
                   start = 2.5e-3, pulse_width = 5e-3):
        
        if rise == 0 and fall == 0:
            
            self.rise = 0
            self.fall = 0
            self.square = True
        
        else:
            
            self.square = False
            self.rise = max(rise, 1e-14)
            self.fall = max(fall, 1e-14)
            
        self.total_time = total_time
        self.start = start
        self.pulse_width = pulse_width
        self.frequency = 1/total_time
        
        if self.start + self.pulse_width > self.total_time:
            warnings.warn('Pulse exceeds total_time, cropping width to match')
            self.pulse_width = self.total_time - self.start
        
        return
    
    def pulse_train(self, pulse_time, width, maxcycles = 30):
        
        self.use_pulse_train = True
        
        return
    
    def kinetics(self, k1, k2, k3, absorbance=None):
        
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        if 'NoneType' not in str(type(absorbance)):
            self.absorbance = absorbance
        
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
    
    def simulate(self):

        if not self.use_pulse_train:
            
            # n_dens, sol, gen, impulse = calc_n(intensity, 
            #                                    k1=k1v, 
            #                                    k2=k2v, 
            #                                    rise=r,
            #                                    fall=r, 
            #                                    square=square, 
            #                                    total_time=total_time, 
            #                                    pulse_time=pulse_time, 
            #                                    width=w, 
            #                                    start=st, 
            #                                    imskpm=True, 
            #                                    maxcycles=30)
            n_dens, sol, gen, impulse = calc_n_pulse(self.intensity,    
                                                     self.absorbance,
                                                     self.k1,
                                                     self.k2,
                                                     self.k3,
                                                     self.rise,
                                                     self.fall,
                                                     self.pulse_width,
                                                     self.total_time,
                                                     self.square)
                                                     

            self.n_dens = n_dens
            self.sol = sol
            self.gen = gen
            
            self.voltage = calc_gauss_volt(self.n_dens)
            self.omega0 = calc_omega0(self.voltage)
    
        return