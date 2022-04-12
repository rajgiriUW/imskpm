# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:02:07 2022

@author: Raj
"""

from .imskpmpoint import IMSKPMPoint
import numpy as np
import matplotlib.pyplot as plt
from .odes import dn_dt_g

class IMSKPMSweep(IMSKPMPoint):
    '''
    Generates a simulated IMSKPM sweep across a range of frequencies
    
    Usage
    
    >> import imskpm
    >> from imskpm.imskpmsweep import IMSKPMSweep
    >> devicesweep = IMSKPMSweep()
    >> devicesweep.simulate_sweep()
    
    * Change the frequencies simulated
    >> devicesweep.frequencies([5,10,20...]) 
    
    * See the outputs during the simulations
    >> devicesweep.simulate_sweep(verbose=True)

    * Plot the result
    >> devicesweep.plot()
    
    Attributes
    ----------
    See IMSKPMPoint for inherited attributes
    cpd_means : list
        The Average CPD at each frequency calculated using Gauss's law
    omega0_means : list
        The average frequency shift at each frequency
    n_dens_means : list
        The average carrier density at each frequency 
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
        '''
        arr : ndArray, optional
            The list of frequencies to use
        '''
        if arr is None:
            #self.frequency_list = np.array([4e-8, 1e-7, 4e-7, 1e-06, 4e-6, 1e-05, 4e-5, 1e-04, 4e-4, 1e-03, 4e-3])
            self.frequency_list = np.array([100, 200, 400, 700, 1000,
                                            2000, 4000, 7000, 1e4, 2e4, 4e4, 7e4, 1e5, 2e5, 4e5, 7e5,
                                            1e6, 2e6, 4e6, 7e6, 1.5e7, 2e7, 4.8e7, 8e7])
        elif isinstance(arr, np.ndarray) or isinstance(arr, list):
            self.frequency_list = np.array(arr)
        else:
            raise ValueError('Must supply a valid NumPy array')
            
        return
    
    def simulate_sweep(self, verbose=False, 
                       total_time = 1.6, max_cycles = 20):
        '''
        Simulates an IMSKPM sweep over many frequencies
        
        self.func is the ODE equation used. Change self. func to any valid function
        
        totaL_time : float
            pulse time at each frequency, 1.6 s = Asylum IMSKPM time
        max_cycles : int
            max cycles per frequency step (number of pulses)
        verbose : bool, optional
            Console display of values at each frequency step
        
        '''
        self.cpd_means = []
        self.omega0_means = []
        self.n_dens_means = []
        
        for f in self.frequency_list:
            if verbose:
                print('Frequency: ', f)
            
            # Step size
            self.dt = 10**(-(np.log10(f)+2)) #100 points per
            self.dt = min(self.dt, 1e-5)
                
            # Create generation pulse sequence
            self.pulse_time = 1/f
            self.pulse_width = 1/(2*f)
            self.start_time = 1/(4*f)
            self.make_pulse(self.rise, self.fall, self.pulse_time, 
                            self.start_time, self.pulse_width)
            self.pulse_train(total_time, max_cycles)
            
            self.simulate()
            
            # Need better step size
            if self._error:
                print('Rerun')
                self.dt /= 10
                self.pulse_time = 1/f
                self.pulse_width = 1/(2*f)
                self.start_time = 1/(4*f)
                self.make_pulse(self.rise, self.fall, self.pulse_time, 
                                self.start_time, self.pulse_width)
                self.pulse_train(total_time, max_cycles)
            
            # Collect results
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
        ax.semilogx(self.frequency_list, self.n_dens_means, 'r^', markersize=6)
        ax.set_ylabel(r'Carrier Density ($cm^{-3}$)')
        ax.set_xlabel(r'Frequency (Hz)')
        ax.set_title(r'IMSKPM, intensity=' + str(self.intensity*1000) + r' $mW/cm^2$')
        plt.tight_layout()
        
        return ax