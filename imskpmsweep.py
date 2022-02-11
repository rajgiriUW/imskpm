# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:02:07 2022

@author: Raj
"""

from .imskpmpoint import IMSKPMPoint
import numpy as np
import matplotlib.pyplot as plt

class IMSKPMSweep(IMSKPMPoint):
    '''
    Generates a simulated IMSKPM sweep
    '''
    def __init__(self, 
                 intensity = 0.1,
                 k1 = 1e6,
                 k2 = 1e-10,
                 k3 = 0,
                 thickness = 500e-7):
        
        super().__init__()
        self.frequencies()
        
        return
    
    def frequencies(self, arr = None):
        
        if arr is None:
            #self.frequency_list = np.array([4e-8, 1e-7, 4e-7, 1e-06, 4e-6, 1e-05, 4e-5, 1e-04, 4e-4, 1e-03, 4e-3])
            self.frequency_list = np.array([10, 20, 40, 70, 100, 200, 400, 700, 1000,
                                            2000, 4000, 7000, 1e4, 2e4, 4e4, 7e4, 1e5, 2e5, 4e5, 7e5,
                                            1e6, 2e6, 4e6, 7e6, 1e7, 2e7, 4e7, 7e7])
        elif isinstance(arr, np.ndarray):
            self.frequency_list = arr
        else:
            raise ValueError('Must supply a valid NumPy array')
            
        return
    
    def simulate_sweep(self, verbose=False):
        '''
        Simulates an IMSKPM sweep over many frequencies
        '''
        self.cpd_means = []
        self.omega0_means = []
        self.n_dens_means = []
        
        for f in self.frequency_list:
            if verbose:
                print('Frequency: ', f)
            if f > 1e6:
                self.dt = 1e-8
            if f < 100:
                self.interpolation = 4
            else:
                self.interpolation = 1
            self.pulse_time = 1/f
            self.pulse_width = 1/(2*f)
            self.start_time = 1/(4*f)
            self.make_pulse(self.rise, self.fall, self.pulse_time, 
                            self.start_time, self.pulse_width)
            self.pulse_train(total_time=2, max_cycles = 20)
            
            self.simulate()
            self.cpd_means.append(self.voltage.mean())
            self.omega0_means.append(self.omega0.mean())
            self.n_dens_means.append(self.n_dens.mean())
            
            if verbose:
                print('Voltage mean = ', self.voltage.mean())
                print('Frequency shift mean = ', self.omega0.mean())
                print('Carrier density mean = ', self.n_dens.mean())
            
        return
    
    def plot(self):
        '''
        Plots the average voltage vs frequency on semi-log plot
        '''
        fig, ax = plt.subplots(nrows=1,figsize=(6,4),facecolor='white')
        ax.semilogx(self.frequency_list, self.cpd_means, 'bs', markersize=6)
        ax.set_ylabel('Voltage (V)')
        ax.set_xlabel(r'Frequency (Hz)')
        ax.set_title(r'IMSKPM, intensity=' + str(self.intensity*1000) + r' $mW/cm^2$')
        plt.tight_layout()
        
        fig, ax = plt.subplots(nrows=1,figsize=(6,4),facecolor='white')
        ax.semilogx(self.frequency_list, self.n_dens_means, 'rt', markersize=6)
        ax.set_ylabel(r'Carrier Density ($cm^{-3}$)')
        ax.set_xlabel(r'Frequency (Hz)')
        ax.set_title(r'IMSKPM, intensity=' + str(self.intensity*1000) + r' $mW/cm^2$')
        plt.tight_layout()
        
        return ax