# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:02:07 2022

@author: Raj
"""

from imskpm import IMSKPM
import numpy as np
import matplotlib.pyplot as plt

class IMSKPMSweep(IMSKPM):
    '''
    Generates a simulated IMSKPM sweep
    '''
    def __init__(self, 
                 intensity = 0.1,
                 k1 = 1e6,
                 k2 = 1e-10,
                 k3 = 0,
                 thickness = 500e-7):
        
        self.frequencies()
        
        return
    
    def frequencies(self, arr = None):
        
        if arr is None:
            self.frequency_list = np.array([4e-8, 1e-7, 4e-7, 1e-06, 4e-6, 1e-05, 4e-5, 1e-04, 4e-4, 1e-03, 4e-3])
        elif isinstance(arr, np.ndarray):
            self.frequency_list = arr
        else:
            raise ValueError('Must supply a valid NumPy array')
            
        return
    
    def simulate_sweep(self):
        '''
        Simulates an IMSKPM sweep over many frequencies
        '''
        self.cpd_means = []
        self.omega0_means = []
        
        for f in self.frequency_list:
            
            self.pulse_time = 1/f
            self.pulse_width = 1/(2*f)
            self.start_time = 1e-9
            self.make_pulse(self.rise, self.fall, self.pulse_time, 
                            self.start_time, self.pulse_width)
            self.pulse_train(total_time = 20*self.pulse_time)
            
            self.simulate()
            self.cpd_means.append(self.voltage.mean())
            self.omega0_means.append(self.omega0.mean())
            
        return
    
    def plot(self):
        
        tx = self.sol.t
        fig, ax = plt.subplots(nrows=1,figsize=(6,4),facecolor='white')
        ax.plot(tx*1e6, self.voltage, 'g')
        ax.set_ylabel('Voltage (V)')
        ax.set_xlabel(r'Time ($\mu$s)')
        vmean = self.voltage.mean() * np.ones(len(self.sol.t))
        ax.plot(tx, vmean, 'r--')
        ax.set_title(str(self.frequency) + ' Hz')
        plt.tight_layout()
        
        return