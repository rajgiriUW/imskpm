'''
Contains defining ODE equations and utilites for simulations
'''

import numpy as np

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
    
    pulse = np.zeros(len(t))
    dt = t[1] - t[0]
    s = int(start / dt)
    w = int(width / dt)
    pulse[s : s + w] = amp * -np.expm1(-(t[:w]/rise))
    pulse[s + w -1 :] = pulse[s + w -1 ] * \
                        np.exp(-(t[s + w - 1:] - t[s + w -1])/fall)
        
    return pulse

def pulse_train(width, amp=1, rise=1e-4, dt=1e-7, pulse_time = 1e-3, 
                total_time=50e-3, maxcycles=20, square=False):
    '''
    
    pulses of width, where each pulses lasts for pulse_time, with total
    series of pulses lasting total_time
    
    pulsewidth : The width of a single pulse
    dt : time step. The default is 1e-7.
    pulsetime : Time per individual pulse
    total_time: Total time for the pulse train
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


def gen_t(absorb, pulse, thickness):
    '''
    Calculates number of carriers based on pulse height and absorbance
    cross-section, evaluated at time t, in electrons/second
    
    pulse: ndArray
        incident light intensity in W/cm^2
    
    absorb : float
        absorbance (a.u.)
    
    thickness : float
        film thickness (cm)
    
    Returns:
    --------
    electrons_area : float
        Calculated electrons per unit area (#/cm^3)
    
    from Jian:
    1sun roughly corresponds to the order 1E16~1E17 cm^-3 conc, 
    60 W/m^2 (0.06 sun) is reasonable if approximated to 3e-15. 

    photons absorbed = incident light intensity in W/m^2
     Power incident * absorption = Power absorbed (J/s*m^2)
     Power absorbed / 1.6e-19 = (J/s*m^2) / (J/electron) = # electrons /s*m^2
     # electrons / thickness = # electrons / s*m^3
    '''
    
    photons = absorb * pulse # W absorbed (Joules/s absorbed) / cm^2
    electrons = photons / 1.6e-19  # electrons/cm^2/s generated
    electrons_area = electrons / thickness # electrons/cm^3/s
      
    return electrons_area

def unity_norm(arr):
    
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


