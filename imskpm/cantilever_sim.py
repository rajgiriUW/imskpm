# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 11:13:00 2022

@author: raj
"""

'''
These commands allows you to simulate a trEFM response based on the input

Once you have simulated a cantilever, you could call the functions below

    >> device = IMSKPM
    
    >> device = IMSKPMPoint()
    >> device.simulate()
    
    >> Z, can_params = calc_cantsim(device.tx, device.omega0)
    >> pix = calc_tfp(Z, can_params)
    
For additional information, please see the FFTA package:
    FFTA: https://github.com/rajgiriUW/ffta
'''

import ffta
from ffta.simulation import mechanical_drive_simple as cw
import numpy as np

def calc_cantsim(tx, omega0, total_time=10e-3, resf=350e3):
    '''
    Simulates a cantilever using the supplied omega0 shift at each time.
    This method does not include a simulation for the electrostatic force directly.
    
    Parameters
    ----------
    tx : ndArray
        Time axis for generating a cantilever.
    omega0 : float
        Resonance frequency shift of the cantilever (Hz).
    total_time : float
        Total time of the simulation (s). The default is 10e-3.
    resf : float, optional
        Resonance frequency of the cantilever (Hz). The default is 350e3.

    Returns
    -------
    Z : ndArray
        The simulated cantilever position over time (m).
    can_params : dict
        Cantilever simulation dictionary
    '''
    
    sampling_rate = np.round(1/ (tx[1] - tx[0]), 0)
    
    can_params = ffta.simulation.utils.load.simulation_configuration('C:/Users/Raj/OneDrive/UW Work/Documents, Proposals, Papers/Grants and Proposals/2020_DOE_Perovskite_BES/Simulation/example_sim_params.cfg')

    k = 24 # N/m 
    start = tx[0]
    can_params[0]['res_freq'] = resf
    can_params[0]['drive_freq'] = resf
    can_params[0]['k'] = k
    can_params[2]['total_time'] = total_time
    can_params[2]['trigger'] = start
    can_params[2]['sampling_rate'] = sampling_rate
    
    cant = cw.MechanicalDrive_Simple(*can_params, w_array=omega0*2*np.pi)
    Z, _ = cant.simulate()
    
    return Z, can_params
    
def calc_tfp(Z, can_params, method='stft', **kwargs):
    '''
    Generates the Pixel class response to see the FFtrEFM response given input deflection Z
    
    Parameters
    ----------
    Z : ndArray
        The simulated cantilever position over time  (m)
    can_params : dict
        Cantilever simulation dictionary.
    method : str
        Pixel simulation method. One of 'hilbert' 'stft' 'wavelet' 'nfmd'. The default is 'stft'.
    **kwargs : 
        See ffta.pixel for acceptable parameters.

    Returns
    -------
    pix: ffta.pixel.Pixel
        Pixel class object of the processed cantilever motion
    '''
    parameters = {}
    
    parameters['n_taps'] = 1199
    parameters['total_time'] = can_params[2]['total_time']
    parameters['trigger'] = max(can_params[2]['trigger'], 0)
    parameters['roi'] = 0.5* (parameters['total_time'] - parameters['trigger'])
    parameters['sampling_rate'] = can_params[2]['sampling_rate']

    pix = ffta.pixel.Pixel(Z, parameters, filter_amplitude=True, method = method, fit=False, **kwargs)
    pix.fft_time_res = 20e-6
    
    try:
        pix.analyze()
    except:
        print('error in tfp')
        pix = ffta.pixel.Pixel(Z, parameters, filter_amplitude=True, method = method, 
                               fit=False, trigger = can_params[2]['total_time']/2)
        pix.analyze()

    return pix