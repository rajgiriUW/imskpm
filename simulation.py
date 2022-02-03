# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 17:09:23 2020

@author: raj

Simulation of trEFM and  IM-SKPM responses:

(1) define generation rate (light intensity * absorption cross section)  G / cm^3/s

(2) define recombination kinetics (could be simple - k * N
                                   where k ~ 10^6 s-1, 
                                   or could add bimolecular terms - k2 * N**2 
                                   (but take k2 from our PL papers, e.g. Dane's Science)

(3) solve for N(t) given the G(t) function from the trEFM or IM-SKPM pulses.  
Here G(t) has a rise and fall slope that is given by the IRF of the source 
-- rather than being an ideal step.

(4) convert N(t) to  V(t) and d2C/dz^2  (V ~ log (N), 
                                         d2C/dz^2 as appropriate with charge density)

(5) convert V(t) and d2C/dz^2(t) to  signal(t) -- 
then apply ff-trEFM filter/processing,
 or lockin/FT and first harmonic extraction to simulate IM-SKPM and 
 examine how IM-SKPM signal vs freq is affected by laser pulse rise time, 
 and trEFM signal vs. laser pulse rise time affect things…compare to ideal square wave excitation
"""



#%%

from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np

import ffta
#%%

def pulse(t, start, width, amp, rise, fall):
    '''
    A square pulse over timescale t with rise/fall time constants
    
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
    
    pulse: array, (W/cm^2)
        incident light intensity in W/cm^2
    
    absorb : float, a.u.
        absorbance
    
    thickness : float, cm
        film thickness
    
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

#%%

def dn_dt(t, n, k1, k2, k3=0):
    '''
    Recombination equation, assuming no auger
    
    Parameters
    ----------
    t : float
        Time (s)
    n : float
        carrier density (cm^-3).
    k1 : float
        recombination, first-order (s^-1)
    k2 : float
        recombination, bimolecular (cm^3/s)
    k3 : float
        recombination, Auger (cm^6/s). Default is 0

    Returns
    -------
    n_dot : float
        differential carrier concentration w.r.t. time.

    '''
    n_dot = -k1 * n - k2 * n**2 - k3 * n**3
    
    return n_dot

def dn_dt_g(t, n, k1, k2, pulse, dt):
    '''
    Recombination+Generation equation, assuming no auger
    
    Calculates the generation at a particular time given a light pulse array
    
    Parameters
    ----------
    t : float
        Time (s)
    n : float
        carrier density (cm^-3).
    k1 : float
        recombination, first-order (s^-1)
    k2 : float
        recombination, bimolecular (cm^3/s)
    pulse : ndarray
        The array describing the pulse being applied to the sample.
    dt : float
        time spacing (s).

    Returns
    -------
    n_dot : float
        differential carrier concentration w.r.t. time.

    '''
    tidx = int(np.floor(t / dt))
    if tidx == len(tx):
        g = pulse[-1]
    else:
        g = pulse[tidx]
    
    n_dot = g - k1 * n - k2 * n**2

    return n_dot

#%%
# 1 ms start, 5 ms long, 200 us rise time, 200 us fall time
# Intensity = 0.1 W/cm^2 = 1000 W / m^2 = 1 Sun
start = 1e-3 
width = 4e-3
rise = 1e-4 
fall = 1e-4
total_time = 10e-3
dt = 1e-7
intensity = 0.1 #AM1.5 in W/m^2
thickness = 500e-7 # 100 nm, in cm
absorbance = 1
tx = np.arange(0, total_time, dt)

intensity=0.1 #1 = 1 W/ cm ^2 = 10 suns
pp = pulse(tx, start, width, intensity, rise, fall)
k1 = 1e6
k2 = 1e-10

gen = gen_t(absorbance, pp, thickness) # electrons / cm^3 / s generated
sol = solve_ivp(dn_dt_g, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                args = (k1, k2, gen, tx[1]-tx[0]), atol=1)

n_dens = sol.y.flatten() # charge density in the film due to recombination + generation

fig, ax = plt.subplots(facecolor='white')
#ax.semilogy(sol.t, sol.y.flatten(), 'r')
ax.plot(sol.t, n_dens, 'r')
ax2 = plt.twinx(ax)
#ax2.semilogy(sol.t, gen, 'b')
ax2.plot(sol.t, gen, 'b')
ax.set_xlabel('Time (s)')
ax.set_ylabel('charge density N(t) (/$cm^3$)')
ax2.set_ylabel('charge generated (/$cm^3 /s$)')
ax.set_title('Charge generated with intensity of ' + str(intensity*1000) + ' $mW/cm^2$')

#%%
# 1 ms start, 5 ms long, 200 us rise time, 200 us fall time
# Intensity = 0.1 W/cm^2 = 1000 W / m^2 = 1 Sun
start = 2e-3 
width = 4e-3
rise = 1e-4
fall = 1e-4 
total_time = 10e-3
dt = 1e-7
intensity = 0.1 #AM1.5 in W/m^2
thickness = 500e-7 # 100 nm, in cm
absorbance = 1
tx = np.arange(0, total_time, dt)

intensity=0.1#0.1 #1 = 1 W/ cm ^2 = 10 suns
pp = pulse(tx, start, width, intensity, rise, fall)
k1 = 1e6
k2 = 1e-10

gen = gen_t(absorbance, pp, thickness) # electrons / cm^3 / s generated
sol = solve_ivp(dn_dt_g, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                args = (k1, k2, gen, tx[1]-tx[0]), atol=1)

n_dens = sol.y.flatten() # charge density in the film due to recombination + generation

fig, ax = plt.subplots(facecolor='white')
#ax.semilogy(sol.t, sol.y.flatten(), 'r')
ax.plot(sol.t, n_dens, 'r')
ax2 = plt.twinx(ax)
#ax2.semilogy(sol.t, gen, 'b')
ax2.plot(sol.t, gen, 'b')
ax.set_xlabel('Time (s)')
ax.set_ylabel('charge density N(t) (/$cm^3$)')
ax.yaxis.label.set_color('red')
ax2.yaxis.label.set_color('blue')
ax2.set_ylabel('charge generated (/$cm^3 /s$)')
ax.set_title('Charge generated with intensity of ' + str(intensity*1000) + ' $mW/cm^2$')
#%%
# Convert to V(t) by assuming V(t) proportional to log(N) like in Voc equation
use_gauss_law = True

if use_gauss_law:
    # Electrostatics
    # 1/k * d2C/dz2 * V**2 --> Hz
    eps0 = 8.8e-12 # F/m
    epsr = 25 # walsh paper
    n_area = n_dens * thickness * 1e4 # into meters
    _wl = 455e-9
    _radius = 0.5 *(0.61 * _wl / 0.6)
    _area = np.pi * _radius**2
    _lift = 20e-9 #lift height, meters
    # From formula for potential from disc of charge
    voltage = 1.6e-19 * 0.5 * n_area / (eps0*epsr) * (np.sqrt(_lift**2 + _radius**2) - _lift)
    #voltage = 1.6e-19 * n_area * _area / eps0

else:
    # PV equation
    voltage = 0.025 * np.log(_area * (thickness*1e-2) * n_dens * 1e6 + 1)  #+1 to avoid log errors ,0.025=kT/q, 1e6 for cm^3 to m^3
    for n, v in enumerate(voltage): # remove 0'sfrom 1e-14
        if np.isnan(v):
            voltage[n] = 0

dcdz = 1e-10
d2cdz2 = 1e-6 # F/m^2 , based on approximations from parabola fit, using 8/24 RRa data as example
# calculation is assuming the fit is -1, and -1 = resF/4k * C''
# e.g. 
k = 24 # N/m 
resf = 350000 #resonance frequency, Hz
fit_from_parms = 0.88
d2cdz2 = fit_from_parms * 4 *k / resf

#delta_f = 0.25 * resf / (4*k) * d2cdz2 * voltage**2  # Marohn and others
delta_f =  resf / (4*k) * d2cdz2 * voltage**2  # Marohn and others
omega0 = resf * np.ones(len(voltage)) - delta_f
fig, ax = plt.subplots(nrows=2,figsize=(8,8),facecolor='white')
#ax.plot(sol.t, delta_f, 'g')
ax[0].plot(sol.t, omega0 - resf, 'g')
ax[1].plot(sol.t, voltage, 'r')
ax[0].set_ylabel('Frequency shift (Hz)')
ax[1].set_ylabel('Voltage (V)')
ax[0].set_xlabel('Time (s)')
ax[1].set_xlabel('Time (s)')
ax[0].set_title('Frequency Shift (Hz)')
if use_gauss_law:
    ax[1].set_title('Voltage (V) with Gauss Law at lift height ' + str(_lift) + ' nm')
else:
    ax[1].set_title('Voltage (V) with PV assumption')
plt.tight_layout()
#%% Convert the delta_f function into a signal
# tx = sol.t
# sampling_rate = 1/ (sol.t[1] - sol.t[0])
# fx = np.linspace(-sampling_rate/2, sampling_rate/2, len(sol.t))

# signal = np.sin(2*np.pi* omega0 * sol.t)
# SIGNAL = np.abs(np.fft.fftshift(np.fft.fft(signal)))

# plt.figure()
# plt.plot(signal)

# #plt.figure(), plt.plot(fx, SIGNAL)

#%%

tx = sol.t
sampling_rate = 1/ (sol.t[1] - sol.t[0])
fx = np.linspace(-sampling_rate/2, sampling_rate/2, len(sol.t))

from ffta.simulation import cantilever_with_w0 as cw
can_params = ffta.simulation.load.simulation_configuration('C:/Users/Raj/OneDrive/UW Work/Documents, Proposals, Papers/Grants and Proposals/2020_DOE_Perovskite_BES/example_sim_params.cfg')

can_params[0]['res_freq'] = resf
can_params[0]['drive_freq'] = resf
can_params[0]['k'] = k
can_params[2]['total_time'] = total_time
can_params[2]['trigger'] = start
can_params[2]['sampling_rate'] = sampling_rate
print(can_params)
decay = np.ones(len(omega0)) * np.exp(-tx/1e-3)
# plt.figure(), plt.plot(decay)

cant = cw.MechanicalDrive_Simple(*can_params, w_array=omega0*2*np.pi)
#cant = cw.MechanicalDrive_Simple(*can_params, w_array=resf * 2 * np.pi * decay)
Z, _ = cant.simulate()
#%%

_, parameters = ffta.pixel_utils.load.configuration('C:/Users/Raj/OneDrive/UW Work/Documents, Proposals, Papers/Grants and Proposals/2020_DOE_Perovskite_BES/parameters.cfg')
print(parameters)
parameters['roi'] = 1e-3
parameters['n_taps'] = 499
method = 'stft'
pix = ffta.pixel.Pixel(Z, parameters, filter_amplitude=True, total_time = total_time, trigger = start-1e-5,
                       sampling_rate = sampling_rate,
                       method = method)
pix.fft_params['time_res'] = 20e-6
pix.analyze()
pix.plot()
print('tFP=',pix.tfp)

pix.fit = False
pix.analyze()
print('Shift=',pix.shift)
#%%% 
# Now loop through many rise times and see if that affects the tFP

def calc_n(intensity, k1 = 1e6, k2 = 1e-10, rise = 1e-4, fall = 1e-4 ):
    start = 2e-3 
    width = 4e-3

    total_time = 10e-3
    dt = 1e-7
    #intensity = 0.1 #AM1.5 in W/m^2
    thickness = 500e-7 # 100 nm, in cm
    absorbance = 1
    tx = np.arange(0, total_time, dt)
    
    intensity=0.1#0.1 #1 = 1 W/ cm ^2 = 10 suns
    pp = pulse(tx, start, width, intensity, rise, fall)
    
    gen = gen_t(absorbance, pp, thickness) # electrons / cm^3 / s generated
    sol = solve_ivp(dn_dt_g, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                    args = (k1, k2, gen, tx[1]-tx[0]), atol=1)
    
    n_dens = sol.y.flatten()

    return n_dens, sol

def calc_gauss_volt(n_dens):
    # Electrostatics
    # 1/k * d2C/dz2 * V**2 --> Hz
    eps0 = 8.8e-12 # F/m
    epsr = 25 # walsh paper
    n_area = n_dens * thickness * 1e4 # into meters
    _wl = 455e-9
    _radius = 0.5 *(0.61 * _wl / 0.6)
    _area = np.pi * _radius**2
    _lift = 20e-9 #lift height, meters
    # From formula for potential from disc of charge
    voltage = 1.6e-19 * 0.5 * n_area / (eps0*epsr) * (np.sqrt(_lift**2 + _radius**2) - _lift)
    #voltage = 1.6e-19 * n_area * _area / eps0
    
    return voltage

def calc_omega0(voltage):
    k = 24 # N/m 
    resf = 350000 #resonance frequency, Hz
    fit_from_parms = 0.88
    d2cdz2 = fit_from_parms * 4 *k / resf
    
    #delta_f = 0.25 * resf / (4*k) * d2cdz2 * voltage**2  # Marohn and others
    delta_f =  resf / (4*k) * d2cdz2 * voltage**2  # Marohn and others
    omega0 = resf * np.ones(len(voltage)) - delta_f
    
    return omega0

def calc_cantsim(tx, omega0):
    sampling_rate = 1/ (tx[1] - tx[0])
    
    can_params = ffta.simulation.load.simulation_configuration('C:/Users/Raj/OneDrive/UW Work/Documents, Proposals, Papers/Grants and Proposals/2020_DOE_Perovskite_BES/example_sim_params.cfg')
    
    can_params[0]['res_freq'] = resf
    can_params[0]['drive_freq'] = resf
    can_params[0]['k'] = k
    can_params[2]['total_time'] = total_time
    can_params[2]['trigger'] = start
    can_params[2]['sampling_rate'] = sampling_rate
    
    cant = cw.MechanicalDrive_Simple(*can_params, w_array=omega0*2*np.pi)
    Z, _ = cant.simulate()
    
    return Z
    
def calc_tfp(Z):
    _, parameters = ffta.pixel_utils.load.configuration('C:/Users/Raj/OneDrive/UW Work/Documents, Proposals, Papers/Grants and Proposals/2020_DOE_Perovskite_BES/parameters.cfg')
    parameters['roi'] = 1e-3
    parameters['n_taps'] = 499
    method = 'stft'
    pix = ffta.pixel.Pixel(Z, parameters, filter_amplitude=True, total_time = total_time, trigger = start-1e-5,
                           sampling_rate = sampling_rate,
                           method = method)
    pix.fft_params['time_res'] = 20e-6
    pix.analyze()

    return pix

#%% As function of intensity
intensity = 0.1
n_dens, sol = calc_n(intensity)
voltage = calc_gauss_volt(n_dens)
omega0 = calc_omega0(voltage)
Z = calc_cantsim(sol.t, omega0)
pix = calc_tfp(Z)

intensities= np.logspace(-3, 3, 7)
tfps = np.zeros(len(intensities))
shifts = np.zeros(len(intensities))
for n, i in enumerate(intensities):
    print(n, i)
    n_dens, sol = calc_n(i)
    voltage = calc_gauss_volt(n_dens)
    omega0 = calc_omega0(voltage)
    Z = calc_cantsim(sol.t, omega0)
    pix = calc_tfp(Z)
    
    print('tFP=',pix.tfp)
    tfps[n] = pix.tfp
    
    pix.fit = False
    pix.analyze()
    print('Shift=',pix.shift)
    shifts[n] = pix.shift
#%%

k1low = 1e4
k2low = 0
intensity = 0.1

risetimes = np.logspace(-6, -3, 4)

for n, r in enumerate(risetimes):
    print(n, r)
    n_dens, sol = calc_n(intensity, k1=k1low, k2=k2low, rise=r)
    voltage = calc_gauss_volt(n_dens)
    omega0 = calc_omega0(voltage)
    Z = calc_cantsim(sol.t, omega0)
    pix = calc_tfp(Z)
    
    print('tFP=',pix.tfp)
    tfps[n] = pix.tfp
    
    pix.fit = False
    pix.analyze()
    print('Shift=',pix.shift)
    shifts[n] = pix.shift