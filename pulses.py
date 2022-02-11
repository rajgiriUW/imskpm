'''
Contains defining ODE equations and utilites for simulations
'''

import numpy as np
from scipy.integrate import solve_ivp
from .odes import dn_dt_g

def pulse(t, start, width, amp, rise, fall):
    '''
    A square pulse over timescale t with rise/fall time constants
    
    t : ndArray
        time axis for generating the pulse
    start : float
        Start time (s)
    width : float
        Width of the pulse (s)
    amp : float
        Amplitude of the pulse (a.u.)
    rise : float
        Rise time of the pulse (s)
    fall : float
        Fall time of the pulse (s)
    
    Returns:
    -------
    pulse: ndArray
        numpy array defining a specific pulse
        
    e.g. pulse(np.arange(0,1e-2,1e-5), 1e-3, 5e-3, 1e-4, 1e-5)
    is a pulse 5 ms wide starting at 1 ms, with rise time 100 us, fall
    time 10 us, with the entire time trace being 10 ms long at 10 us sampling
    
    '''
    
    # Perfect/ideal square pulses
    if rise == 0 and fall == 0:
        
        return step(t, start, width, amp)

    pulse = np.zeros(len(t))
    dt = t[1] - t[0]
    s = int(start / dt)
    w = int(width / dt)
    
    # Realistic pulses with an exponential time
    pulse[s : s + w] = amp * -np.expm1(-(t[:w]/rise))
    pulse[s + w -1 :] = pulse[s + w -1 ] * \
                        np.exp(-(t[s + w - 1:] - t[s + w -1])/fall)
        
    return pulse

def step(t, start, width, amp):
    '''
    Generates a single step pulse with no rise time
    
    Parameters
    ----------
    t : ndArray 
        Time axis for the pulse (s).
    start : float
        Time index for the pulse to start (s)
    width : float
        Temporal width of the pulse (s).
    amp : float
        Amplitude (a.u.)

    Returns
    -------
    pulse : ndArray
        Array defining the light pulse (a.u., x-scaling in s)

    '''
    
    pulse = np.zeros(len(t))
    dt = t[1] - t[0]
    s = int(start / dt)
    w = int(width / dt)
    
    pulse[s : s + w] = amp
        
    return pulse    

def pulse_train(width, amp=1, rise=1e-4, dt=1e-7, pulse_time = 1e-3, 
                total_time=50e-3, maxcycles=20, square=False):
    '''
    A train of pulses:
        each pulse of width = "width"
        each pulses lasts for time "pulse_time" 
        total series of pulses lasting time "total_time"
        
    width : float 
        The width of a single pulse (s)
    amp : float
        The amplitude of each pulse (a.u.)
    dt : float 
        time step (s). The default is 1e-7.
    pulse_time : float
        Time per individual pulse (s)
    total_time: float
        Total time for the pulse train (s)
    maxcycles : int
        Maximum number of pulses to generate, to avoid long computations
    square: bool, optional
        Returns "ideal square steps" instead of using a rise time
    '''
    
    # Create square wave train
    cycles = min(int(np.ceil(total_time/pulse_time)), maxcycles)
    
    pulse = np.zeros(int(pulse_time/dt))
    start = len(pulse) // 4
    end = start + int(width /dt)
    
    # pulse[len(pulse)//4:len(pulse//4)] = amp
    pulse[start:end] = amp
    impulse = np.tile(pulse, cycles)
    tx = np.arange(len(impulse))*dt
    
    if not square:
        #integrate this square wave train, assumes symmetric rise/fall
        sol = solve_ivp(dn_dt_g, [tx[0], tx[-1]], [impulse[0]], t_eval = tx,
                        args = (1/rise, 0, impulse, tx[1]-tx[0]), max_step=dt)
    
        pulses  = sol.y.flatten()
        #rescale since integration won't have right magnitude
        pulses = amp * (pulses - np.min(pulses))/(np.max(pulses) - np.min(pulses))
    
    else:
        
        pulses = impulse
    
    return pulses, tx


def pulse_at(t, start, width, amp, rise, fall):
    '''
    Find value of the pulse at a specific time
    
    t : float
        Specific time index to find (s)
    start : float
        Start time (s)
    width : float
        Width of the pulse (s)
    amp : float
        Amplitude of the pulse (a.u.)
    rise : float
        Rise time of the pulse (s)
    fall : float
        Fall time of the pulse (s)
    '''
    tx = np.arange(0, 10e-3, 0.00001) # assuming never exceeds 10 ms
    _pulse = pulse(tx, start, width, amp, rise, fall)
    
    tidx = np.searchsorted(tx, t)
    if tidx == len(tx):
        return _pulse[-1]
    return _pulse[tidx]