import numpy as np

from .pulses import pulse_train, pulse
from .odes import dn_dt_g
from scipy.integrate import solve_ivp

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

def calc_n_pulse_train(intensity, 
                       absorbance=1, 
                       k1 = 1e6, 
                       k2 = 1e-10, 
                       k3 = 0,
                       rise = 1e-4, 
                       fall = 1e-4,
                       start = 0.5e-3 , 
                       width = 1e-3, 
                       dt = None,
                       pulse_time=5e-3, 
                       total_time = 2e-3, 
                       maxcycles=20,
                       square = False                       
                       ):
    '''
    IM-SKPM method
    
    Calculating the integrated charge density by using a train of pulses

    Parameters
    ----------
    intensity : float
        intensity in W/cm^2. 0.1 = 1 Sun, 100 mW / cm^2
    absorbance : float, optional
        Light absorbed by the sample, for wavelength-dependence. The default is 1.
    k1 : float
        recombination, first-order (s^-1). The default is 1e6.
    k2 : float
        recombination, bimolecular (cm^3/s). The default is 1e-10.
    k3 : float
        recombination, Auger (cm^6/s). Default is 0
    rise : float
        Rise time of the pulse (s). The default is 1e-4.
    fall : float
        Fall time of the pulse (s). The default is 1e-4.
    start : float
        Start time (s). The default is 0.5e-3.
    width : float
        Width of the pulse (s). The default is 1e-3.
    dt : float 
        time step (s). The default is None.
    pulse_time : float
        Time per individual pulse (s). The default is 5e-3.
    total_time: float
        Total time for the pulse train (s). The default is 2e-3.
    maxcycles : int
        Maximum number of pulses to generate, to avoid long computations. The default is 20
    square : bool, optional
        Uses ideal square pulses. The default is False.


    Returns
    -------
    n_dens : float
        Char density in the film due to ODE (Generation - Recombination) (#/cm^3).
    sol :  `OdeSolution`
        (From Scipy) Found solution as `OdeSolution` instance
    gen : float
        Carrier concentration GENERATED (#/cm^3).
    impulse : ndArray
        The pulse train generated.

    '''
    if not dt:
    
        if width >= 1e-6:
            dt = 1e-7
        else:
            dt = width/10

    thickness = 500e-7 # 500 nm, in cm

    ty = np.arange(0, total_time, dt)
    
    impulse, ty = pulse_train(width, amp=intensity, rise=rise, dt=dt, 
                              pulse_time=pulse_time, total_time=total_time,
                              square=square, maxcycles=maxcycles)
            
    gen = gen_t(absorbance, impulse, thickness) # electrons / cm^3 / s generated
    
    scale = 1e-4 #1 = /cm^3, 1e-4 = /um^3, 1e2 = /m^2
    gen = gen * scale**3 #(from /cm^3) 
    k2 = k2 / scale**3 #(from cm^3/s)

    sol = solve_ivp(dn_dt_g, (ty[0], ty[-1]), (gen[0],), t_eval = ty,
                    args = (k1, k2, k3, gen, ty[1]-ty[0]), max_step = dt)

    n_dens = sol.y.flatten()  # charge density in the film due to recombination + generation

    gen = gen / scale**3
    n_dens = n_dens / scale**3
    
    return n_dens, sol, gen, impulse
    
def calc_n_pulse(intensity, 
                 absorbance = 1, 
                 k1 = 1e6, 
                 k2 = 1e-10, 
                 k3 = 0,
                 rise = 1e-4, 
                 fall = 1e-4,
                 start = 2e-3, 
                 width = 4e-3, 
                 total_time = 10e-3, 
                 square=False):
    '''
    Calculating the integrated charge density by using a single pulse
    
    Parameters
    ----------
    intensity : float
        intensity in W/cm^2. 0.1 = 1 Sun, 100 mW / cm^2
    absorbance : float, optional
        Light absorbed by the sample, for wavelength-dependence. The default is 1.
    k1 : float
        recombination, first-order (s^-1). The default is 1e6.
    k2 : float
        recombination, bimolecular (cm^3/s). The default is 1e-10.
    k3 : float
        recombination, Auger (cm^6/s). Default is 0
    rise : float
        Rise time of the pulse (s). The default is 1e-4.
    fall : float
        Fall time of the pulse (s). The default is 1e-4.
    start : float
        Start time (s). The default is 0.5e-3.
    width : float
        Width of the pulse (s). The default is 1e-3.
    square : bool, optional
        Uses ideal square pulses. The default is False.

    Returns
    -------
    n_dens : float
        Char density in the film due to ODE (Generation - Recombination) (#/cm^3).
    sol :  `OdeSolution`
        (From Scipy) Found solution as `OdeSolution` instance
    gen : float
        Carrier concentration GENERATED (#/cm^3).

    '''
    
    if width < 1e-6:
        dt = 1e-8
    else:
        dt = 1e-7
        
    #intensity = 0.1 #AM1.5 in W/m^2
    thickness = 500e-7 # 100 nm, in cm

    tx = np.arange(0, total_time, dt)
    
    pp = pulse(tx, start, width, intensity, rise, fall)
    
    gen = gen_t(absorbance, pp, thickness) # electrons / cm^3 / s generated
    
    scale = 1e-4 #1 = /cm^3, 1e-4 = /um^3, 1e2 = /m^2
    gen = gen * scale**3 #(from /cm^3) 
    k2 = k2 / scale**3 #(from cm^3/s)
    
    sol = solve_ivp(dn_dt_g, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                    args = (k1, k2, k3, gen, tx[1]-tx[0]))
    
    if not any(np.where(sol.y.flatten() > 0)[0]):                

        sol = solve_ivp(dn_dt_g, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                        args = (k1, k2, k3, gen, tx[1]-tx[0]), max_step=1e-6)
    
    n_dens = sol.y.flatten()
    
    gen = gen / scale**3
    n_dens = n_dens / scale**3
    
    return n_dens, sol, gen

def calc_gauss_volt(n_dens, lift = 20e-9, thickness = 500e-7):
    '''
    Calculates the voltage using Gauss's law integration at a certain lift height
    
    Parameters
    ----------
    n_dens : flaot
        Carrier density (electrons/cm^3)
    lift : float
        The lift height of the tip above the surface (m). The default is 20e-9.
    thickness : float
        Film thickness (m). The default is 500e-7.

    Returns
    -------
    voltage : float
        The calcualted voltage via Gauss's law (V)
    '''
    
    # Electrostatics
    # 1/k * d2C/dz2 * V**2 --> Hz
    eps0 = 8.8e-12 # F/m
    epsr = 25 # walsh paper
    n_area = n_dens * thickness * 1e4 # into meters
    _wl = 455e-9
    _radius = 0.5 *(0.61 * _wl / 0.6)
    _area = np.pi * _radius**2
    _lift = lift #lift height, meters
    
    # From formula for potential from disc of charge
    voltage = 1.6e-19 * 0.5 * n_area / (eps0*epsr) * (np.sqrt(_lift**2 + _radius**2) - _lift)
    #voltage = 1.6e-19 * n_area * _area / eps0
    
    return voltage

def calc_omega0(voltage, resf = 350e3):
    '''
    Calculates the frequency shift assuming an input voltage

    Parameters
    ----------
    voltage : float
        The voltage between the tip and sample (V).
    resf : float, optional
        The cantilever resonance frequency (Hz). The default is 350e3.

    Returns
    -------
    omega0 : float
        Resonance frequency shift of the cantilever (Hz)
    '''
    
    k = 24 # N/m 
    fit_from_parms = 0.88
    d2cdz2 = fit_from_parms * 4 *k / resf
    
    #delta_f = 0.25 * resf / (4*k) * d2cdz2 * voltage**2  # Marohn and others
    delta_f =  resf / (4*k) * d2cdz2 * voltage**2  # Marohn and others
    omega0 = resf * np.ones(len(voltage)) - delta_f
    
    return omega0

