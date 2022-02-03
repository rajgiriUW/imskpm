# -*- coding: utf-8 -*-
"""
Simulation code for checking trEFM and IM-SKPM pulses

Raj Giridharagopal
"""


#%%

from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import pickle
import ffta
from ffta.simulation import mechanical_drive_simple as cw
from scipy.optimize import minimize
from scipy.interpolate import interp1d
#%%

def unity_norm(arr):
    
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def pulse(t, start, width, amp, rise, fall):
    '''
    A square pulse over timescale t with rise/fall time constants
    
    start : float, start time
    width : float, width of the pulse
    amp : float, amplitude of the pulse
    rise : float, rise time of the pulse
    fall : float, fall time of the pulse
    
    e.g. pulse = gen_t(np.arange(0,1e-2,1e-5), 1e-3, 5e-3, 1e-4, 1e-5)
    is a pulse 5 ms wide starting at 1 ms, with rise time 100 us, fall
    time 10 us, with the entire time trace being 10 ms long at 10 us sampling
    
    '''
    
    pulse = np.zeros(len(t))
    dt = t[1] - t[0]
    s = int(start / dt)
    w = int(width / dt)
    
    if rise == 0:
        pulse[s : s + w] = amp
        return pulse 
    
    pulse[s : s + w] = amp * -np.expm1(-(t[:w]/rise))
    pulse[s + w -1 :] = pulse[s + w -1 ] * \
                        np.exp(-(t[s + w - 1:] - t[s + w -1])/fall)
        
    return pulse

def step(t, start, width, amp):

    pulse = np.zeros(len(t))
    dt = t[1] - t[0]
    s = int(start / dt)
    w = int(width / dt)
    pulse[s : s + w] = amp
        
    return pulse    

def gen_t(absorb, pulse, thickness):
    '''
    Calculates number of carriers based on pulse height and absorbance
    cross-section, evaluated at time t, in electrons/second
    
    pulse: array, (W/cm^2)
        incident light intensity in W/cm^2
    
    absorb : float, a.u.
        percentage absorbance (not actually absorbance)
    
    thickness : float, cm
        film thickness
    
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

def im_pulse2(width, amp=1, rise=1e-4, dt=1e-7, pulse_time = 1e-3, 
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

def nkt_pulse():
    
    return
#%% ODE equations; dn_dt_g = N' + generation

def dn_dt_g(t, n, k1, k2, pulse, dt):
    '''
    Simple ODE for assuming monomoleuclar (k1) and bimolecular (k2) rates,
    with generation rate G defined by a given input pulse
    
    Pulse : ndarray, the excitation pulse that defines generation rate at each time
    dt : float, time step
    
    returns: n_dot = dN/dt, the charge density per unit time
    '''
    tidx = int(np.floor(t / dt))
    if tidx >= len(pulse):
        g = pulse[-1]
    else:
        g = pulse[tidx]
        
    n_dot = g - k1 * n - k2 * n**2

    return n_dot

#%% Function wrappers for calculations

# 
def calc_n(intensity, absorbance=1, k1 = 1e6, k2 = 1e-10, rise = 1e-4, fall = 1e-4,
           start = 0.5e-3 , width = 1e-3, total_time = 2e-3, pulse_time=5e-3, 
           square=False, imskpm=False, maxcycles=20, dt = None):
    '''
    IM-SKPM method
    
    Calculating the integrated charge density by using a train of pulses
    '''
    if not dt:
    
        if width >= 1e-6:
            dt = 1e-7
        else:
            dt = width/10

    thickness = 500e-7 # 100 nm, in cm

    ty = np.arange(0, total_time, dt)
    
    if imskpm:
        impulse, ty = im_pulse2(width, amp=intensity, rise=rise, dt=dt, 
                                pulse_time=pulse_time, total_time=total_time,
                                square=square, maxcycles=maxcycles)
    else:
        impulse = pulse(ty, start, width, intensity, rise, fall)
        if square:
            # pp = np.zeros(len(pp))
            # pp[int(start / dt):int(start+width / dt)] = intensity
            dt = 1e-8
            ty = np.arange(0, total_time, dt)
            impulse = pulse(ty, start, width, intensity, 0, 0)
            
    gen = gen_t(absorbance, impulse, thickness) # electrons / cm^3 / s generated
    
    scale = 1e-4 #1 = /cm^3, 1e-4 = /um^3, 1e2 = /m^2
    gen = gen * scale**3 #(from /cm^3) 
    k2 = k2 / scale**3 #(from cm^3/s)

    sol = solve_ivp(dn_dt_g, (ty[0], ty[-1]), (gen[0],), t_eval = ty,
                args = (k1, k2, gen, ty[1]-ty[0]), max_step = dt)

    n_dens = sol.y.flatten()  # charge density in the film due to recombination + generation

    gen = gen / scale**3
    n_dens = n_dens / scale**3
    
    return n_dens, sol, gen, impulse
    
def calc_n_old(intensity, k1 = 1e6, k2 = 1e-10, rise = 1e-4, fall = 1e-4,
           start = 2e-3 , width = 4e-3, total_time = 10e-3, square=False):
    '''
    FF-trEFM method
    
    Calculating the integrated charge density by using a single pulse
    '''
    if width < 1e-6:
        dt = 1e-8
    else:
        dt = 1e-7
        
    #intensity = 0.1 #AM1.5 in W/m^2
    thickness = 500e-7 # 100 nm, in cm
    absorbance = 1
    tx = np.arange(0, total_time, dt)
    
    pp = pulse(tx, start, width, intensity, rise, fall)
    if square:
        # pp = np.zeros(len(pp))
        # pp[int(start / dt):int(start+width / dt)] = intensity
        pp = pulse(tx, start, width, intensity, 1e-16, 1e-16)
    
    gen = gen_t(absorbance, pp, thickness) # electrons / cm^3 / s generated
    
    scale = 1e-4 #1 = /cm^3, 1e-4 = /um^3, 1e2 = /m^2
    gen = gen * scale**3 #(from /cm^3) 
    k2 = k2 / scale**3 #(from cm^3/s)
    
    sol = solve_ivp(dn_dt_g, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                    args = (k1, k2, gen, tx[1]-tx[0]))
    
    if not any(np.where(sol.y.flatten() > 0)[0]):                

        sol = solve_ivp(dn_dt_g, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                        args = (k1, k2, gen, tx[1]-tx[0]), max_step=1e-6)
    
    n_dens = sol.y.flatten()
    
    gen = gen / scale**3
    n_dens = n_dens / scale**3
    
    return n_dens, sol, gen

def calc_gauss_volt(n_dens, lift = 20e-9):
    '''
    Calculates the voltage using Gauss's law integration at a 20 nm lift height
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
    
    resf: resonance frequency of the cantilever
    '''
    k = 24 # N/m 
    fit_from_parms = 0.88
    d2cdz2 = fit_from_parms * 4 *k / resf
    
    #delta_f = 0.25 * resf / (4*k) * d2cdz2 * voltage**2  # Marohn and others
    delta_f =  resf / (4*k) * d2cdz2 * voltage**2  # Marohn and others
    omega0 = resf * np.ones(len(voltage)) - delta_f
    
    return omega0

def calc_cantsim(tx, omega0, total_time=10e-3, resf=350e3):
    '''
    Simulates a cantilever using the supplied omega0 shift at each time.
    This method does not include a simulation for the electrostatic force directly.
    '''
    sampling_rate = np.round(1/ (tx[1] - tx[0]), 0)
    
    can_params = ffta.simulation.utils.load.simulation_configuration('C:/Users/Raj/OneDrive/UW Work/Documents, Proposals, Papers/Grants and Proposals/2020_DOE_Perovskite_BES/Simulation/example_sim_params.cfg')

    k = 24 # N/m 
    
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
    '''
 
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

#%% Plotting wrapper

def plot():
    # Plots
    fig, ax = plt.subplots(facecolor='white')
    ax.plot(sol.t*1e3, n_dens, 'r')
    ax2 = plt.twinx(ax)
    ax2.plot(sol.t*1e3, gen, 'b')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('charge density N(t) (/$cm^3$)')
    ax.yaxis.label.set_color('red')
    ax2.yaxis.label.set_color('blue')
    ax2.set_ylabel('charge generated (/$cm^3 /s$)')
    ax.set_title('Charge generated with intensity of ' + str(intensity*1000) + ' $mW/cm^2$')
    
    fig, ax = plt.subplots(nrows=2,figsize=(8,8),facecolor='white')
    #ax.plot(sol.t, delta_f, 'g')
    ax[0].plot(sol.t*1e3, omega0 - resf, 'g')
    ax[1].plot(sol.t*1e3, voltage, 'r')
    ax[0].set_ylabel('Frequency shift (Hz)')
    ax[1].set_ylabel('Voltage (V)')
    ax[0].set_xlabel('Time (ms)')
    ax[1].set_xlabel('Time (ms)')
    ax[0].set_title('Frequency Shift (Hz)')
    ax[1].set_title('Voltage (V) at lift height ' + str(lift) + ' nm')
    plt.tight_layout()
    
    fig, ax = plt.subplots(facecolor='white', figsize=(10,6))
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    st = int(pix.trigger * pix.sampling_rate)
    ax.plot(1e3 *sol.t[st-500:-500], pix.inst_freq[st-500:-500], linewidth=2, color='k')
    ax.legend()
    ax.set_title('k1,' + str(k1) + ' vs rise time,' + str(intensity) + 'mW/cm^2')

    return
#%% Single pulse example

thickness = 500e-7 # in centimeters
resf = 350e3
lift = 20e-9

intensity = 10#10 # 0.1 = 1 Sun, 100 mW / cm^2
k1 = 1e5
k2 = 1e-11

start = 5e-4
width = 1e-3
pulse_time = 3e-3
total_time = 2.8e-3
rise = 5e-7

k1 = 1e6
k2 = 1e-10
rise = 5e-7
width = 2e-3
pulse_time = 4e-3
total_time = 5e-3

if rise ==0:
    square = True
else:
    square = False
    
n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start)
dt = np.round(sol.t[-1] - sol.t[-2], 9)
voltage = calc_gauss_volt(n_dens, lift=lift)
omega0 = calc_omega0(voltage, resf = resf)
Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
# can_params[2]['trigger'] = start
# can_params[2]['total_time'] = len(sol.t) / can_params[2]['sampling_rate']  
pix = calc_tfp(Z, can_params, method='hilbert')

print('tFP=',pix.tfp)
print('Shift=',pix.shift)

plot()

if pix.sampling_rate / 1e7 > 1:
    print('up')
    pix.n_taps = 2499
pix.analyze()
pix.plot()
#%% Pulse train/ IMSKPM example

thickness = 500e-7 # in centimeters
resf = 350e3
lift = 20e-9

intensity = 1 # in W/cm^2. 0.1 = 1 Sun, 100 mW / cm^2
k1 = 1e5
k2 = 1e-11

start = 1e-9
width = 0.1e-9
pulse_time = 13e-9
total_time = 20e-3
rise = 0#1e-5    
if rise ==0:
    square = True
else:
    square = False
    
n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, 
                                   square=square, total_time=total_time, pulse_time=pulse_time, 
                                   width=width, start=start, imskpm=True, maxcycles=100)
dt = np.round(sol.t[-1] - sol.t[-2], 10)
if not dt:
    dt = np.round(sol.t[-1] - sol.t[-2], 12)
voltage = calc_gauss_volt(n_dens, lift=lift)
omega0 = calc_omega0(voltage, resf = resf)
try:
    Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
except:
    Z, can_params = calc_cantsim(sol.t, omega0, total_time=sol.t[-1] + dt)
can_params[2]['trigger'] = start
can_params[2]['total_time'] = len(sol.t) / can_params[2]['sampling_rate']  
pix = calc_tfp(Z, can_params, method='hilbert')

plot()


#%%
# Pulse Picker example, from 2 MHz to 78 MHz = 500 ns to 13 ns

pulse_times = np.array([13, 20, 50, 100, 200]) * 1e-9
scales = pulse_times / 13e-9
thickness = 500e-7 # in centimeters
resf = 350e3
lift = 20e-9

intensity = 1 # 0.1 = 1 Sun, 100 mW / cm^2
k1 = 1e6
k2 = 1e-10

start = 1e-9
width = 0.1e-9
pulse_time = 13e-9
total_time = 20e-3
rise = 0#1e-5    
if rise ==0:
    square = True
else:
    square = False
    
for pulse_time, scale in zip(pulse_times, scales):
    
    print(pulse_time, scale)
    n_dens, sol, gen, impulse = calc_n(intensity*scale, k1=k1, k2=k2, rise=rise,fall=rise, 
                                   square=square, total_time=total_time, pulse_time=pulse_time, 
                                   width=width, start=start, imskpm=True, maxcycles=50)
    print('Averge generation = ', gen.mean())
    dt = np.round(sol.t[-1] - sol.t[-2], 10)
    if not dt:
        dt = np.round(sol.t[-1] - sol.t[-2], 12)
    voltage = calc_gauss_volt(n_dens, lift=lift)
    omega0 = calc_omega0(voltage, resf = resf)
    print(voltage.mean())
    # try:
    #     Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
    # except:
    #     Z, can_params = calc_cantsim(sol.t, omega0, total_time=sol.t[-1] + dt)
    # can_params[2]['trigger'] = start
    # can_params[2]['total_time'] = len(sol.t) / can_params[2]['sampling_rate']  
    # pix = calc_tfp(Z, can_params, method='hilbert')
    
    fig, ax = plt.subplots(facecolor='white')
    ax.plot(sol.t*1e3, n_dens, 'r')
    ax2 = plt.twinx(ax)
    ax2.plot(sol.t*1e3, gen, 'b')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('charge density N(t) (/$cm^3$)')
    ax.yaxis.label.set_color('red')
    ax2.yaxis.label.set_color('blue')
    ax2.set_ylabel('charge generated (/$cm^3 /s$)')
    ax.set_title('Charge generated with intensity of ' + str(intensity*scale*1000) + ' $mW/cm^2$, pulse width = ' + str(pulse_time * 1e9) + ' ns')
    plt.tight_layout()
    plt.savefig('Charge generated with intensity of ' + str(intensity*scale*1000) + ' mW_cm^2, pulse width = ' + str(pulse_time * 1e9) + ' ns'+'.jpg')
    
    fig, ax = plt.subplots(nrows=2,figsize=(8,8),facecolor='white')
    #ax.plot(sol.t, delta_f, 'g')
    ax[0].plot(sol.t*1e3, omega0 - resf, 'g')
    ax[1].plot(sol.t*1e3, voltage, 'r')
    ax[0].set_ylabel('Frequency shift (Hz)')
    ax[1].set_ylabel('Voltage (V)')
    ax[0].set_xlabel('Time (ms)')
    ax[1].set_xlabel('Time (ms)')
    ax[0].set_title('Frequency Shift (Hz)')
    ax[1].set_title('Voltage (V) at lift height ' + str(lift) + ' nm, k1=' + str(k1) + ' /s' + str(pulse_time * 1e9) + ' ns')
    plt.tight_layout()
    plt.savefig('Charge generated with intensity of ' + str(intensity*scale*1000) + ' mW_cm^2, pulse width = ' + str(pulse_time * 1e9) + ' ns'+'.jpg')
    
    with open('sol_pulsedep_' + str(intensity*10) +'Sun_k1,' + str(k1) + '_pulsetime,_' + \
              str(pulse_time*1e9) + 'ns.pkl', 'wb') as handle:
        pickle.dump(sol, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%% Simple function of rise times for tFP calcs
# Weird bug where fast pulses don't seem to integrate if placed at the center

k1low = 1e4
k2low = 0
intensity = 10

tfps = []
shifts = []

k1_vals = [1e5, 1e6, 1e7]
k2_vals = [1e-11,1e-10, 1e-9]
rise_vals = [1e-7, 1e-4]
# rise_vals = [5e-7]
k1 = k1_vals[2]
k2 = k2_vals[2]

k1 = 1e6
k2 = 1e-10

start = 5e-3
width = 1e-7
start = 1e-9
width = 0.1e-9
pulse_time = 13e-9
total_time = 20e-3

fig, ax = plt.subplots(facecolor='white', figsize=(10,6))
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Frequency (Hz)')
for n, r in enumerate(rise_vals):
    
    print(n, r)
    if r ==0:
        square = True
    else:
        square = False
    
    n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, 
                                   square=square, total_time=total_time, pulse_time=pulse_time, 
                                   width=width, start=start, imskpm=True, maxcycles=300)
    
    voltage = calc_gauss_volt(n_dens)
    omega0 = calc_omega0(voltage)
    Z, can_params = calc_cantsim(sol.t, omega0)
    can_params[2]['trigger'] = start
    can_params[2]['total_time'] = len(sol.t) / can_params[2]['sampling_rate']  
    pix = calc_tfp(Z, can_params)
    
    print('tFP=',pix.tfp)
    print('Shift=',pix.shift)

    _idx0 = int(start/2 * pix.sampling_rate)
    _idxf = int((start + width+1e-3) * pix.sampling_rate)
    _tidx = int(start * pix.sampling_rate)
    if r < 1e-5:
        _tidx -= 200
    ax.plot(1e3 *sol.t[_idx0:_idxf], pix.inst_freq[_idx0:_idxf]-pix.inst_freq[_tidx], linewidth=2, label=r)
    # ax.plot(pix.inst_freq[_idx0:_idxf]-pix.inst_freq[_tidx], linewidth=2, label=r)
ax.legend()
ax.set_title('k1,' + str(k1) + ' vs rise time,' + str(intensity) + 'mW/cm^2')
#%% Example IM-SKPM curve

#widths = np.array([4e-7, 1e-06, 4e-6, 1e-05, 4e-5, 1e-04, 4e-4, 1e-03, 4e-3, 1e-02, 4e-2, 1e-1, 4e-1])
widths = np.array([4e-8, 1e-7, 4e-7, 1e-06, 4e-6, 1e-05, 4e-5, 1e-04, 4e-4, 1e-03, 4e-3])
widths = np.array([4e-8, 1e-7, 4e-7, 1e-06, 4e-6, 1e-05, 4e-5, 1e-04, 4e-4])

thickness = 500e-7 # in centimeters
resf = 350e3
lift = 20e-9

intensity = 10 # 0.1 = 1 Sun, 100 mW / cm^2
k1 = 1e8
k2 = 1e-8

start = 1e-4
total_time = 20e-3
rise = 5e-7    

if rise == 0:
    square = True
else:
    square = False

v_means = []
omega0_means = []
total_dens = {}
for w in widths:
            
    print(w)
    
    st = w/4
    pulse_time = 2*w
    n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, 
                                   square=square, total_time=total_time, pulse_time=pulse_time, 
                                   width=w, start=st, imskpm=True, maxcycles=500)
    
    voltage = calc_gauss_volt(n_dens)
    omega0 = calc_omega0(voltage)
    v_means.append(voltage.mean())
    omega0_means.append(omega0.mean())
    total_dens[w] = sol
    print(voltage.mean())

    if square:
        name = str(w**-1) + ' Hz, k1,' + str(k1) + '_k2,' + str(k2) + '_risefall,000_' + str(intensity*1000) +'.jpg'# + 'mW.pkl'
    else:
        name = str(w**-1) + ' Hz, k1,' + str(k1) + '_k2,' + str(k2) + '_risefall,' + str(rise) + '_' + str(intensity*1000) +'.jpg'# + 'mW.pkl'
    total_dens['v_means'] = np.copy(v_means)
    total_dens['omega0_means'] = np.copy(omega0_means)
    
    fig, ax = plt.subplots(nrows=1,figsize=(6,4),facecolor='white')
    ax.plot(sol.t, voltage, 'g')
    ax.set_ylabel('Voltage (V)')
    ax.set_xlabel('Time (s)')
    vmean = voltage.mean() * np.ones(len(sol.t))
    ax.plot(sol.t, vmean, 'r--')
    ax.set_title(str(w**-1) + ' Hz')
    plt.tight_layout()
    plt.savefig(name)

total_dens['widths'] = widths
fig, ax = plt.subplots(facecolor='white')
ax.plot(sol.t, n_dens, 'r')
ax2 = plt.twinx(ax)
ax2.plot(sol.t, gen[:len(sol.t)], 'b')
ax.set_xlabel('Time (s)')
ax.set_ylabel('charge density N(t) (/$cm^3$)')
ax.yaxis.label.set_color('red')
ax2.yaxis.label.set_color('blue')
ax2.set_ylabel('charge generated (/$cm^3 /s$)')
ax.set_title('Charge generated with intensity of ' + str(intensity*1000) + ' $mW/cm^2$')

print('v_means', v_means)
print('w0_means', omega0_means)
        
fig, ax = plt.subplots(facecolor='white')
ax.semilogx(total_dens['widths']**-1, total_dens['v_means'], 'bs', markersize=8)
ax.set_xlabel('Pulse Frequency (Hz)')
ax.set_ylabel('Average Voltage (V)') 
ax.set_title('k1='+str(k1) +' /s, k2=' + str(k2) + 'cm^3/s, with ' + str(rise) + ' s rise time')
plt.tight_layout()
   

#%% Processing IMSKPM into trEFM
base = r'E:\Raj DOE SImulation Pickles\pkl\k1_dep_imskpm\\'
with open(base + r'total_dens_1e4100Sun_k1,1e4.pkl', 'rb') as handle:
    total_dens_1e4 = pickle.load(handle)
    
with open(base + r'total_dens_1e6100Sun_k1,1e6.pkl', 'rb') as handle:
    total_dens_1e6 = pickle.load(handle)

with open(base + r'total_dens_1e8100Sun_k1,1e8.pkl', 'rb') as handle:
    total_dens_1e8 = pickle.load(handle)
widths = np.array([4e-8, 1e-7, 4e-7, 1e-06, 4e-6, 1e-05, 4e-5, 1e-04, 4e-4])
scale = 1e-4
thickness = 500e-7 # in centimeters
resf = 350e3
lift = 20e-9
start = 1e-4
total_time = 20e-3
rise = 5e-7    

labels = [1e4, 1e6, 1e8]
_, parameters = ffta.pixel_utils.load.configuration('C:/Users/Raj/OneDrive/UW Work/Documents, Proposals, Papers/Grants and Proposals/2020_DOE_Perovskite_BES/Simulation/parameters.cfg')

for tdens in  [total_dens_1e4, total_dens_1e6, total_dens_1e8]:
    
    f_means = []
    for w in widths:
        
        sol = tdens[w]
        n_dens = tdens[w].y.flatten()
        n_dens = n_dens / scale**3 # to get back into /cm^3 units from /um^3
        voltage = calc_gauss_volt(n_dens, lift=lift)
        omega0 = calc_omega0(voltage, resf = resf)
        dt = np.round(sol.t[-1] - sol.t[-2], 10)
        if not dt:
            dt = np.round(sol.t[-1] - sol.t[-2], 12)
        total_time = len(sol.t) * dt
        Z, can_params = calc_cantsim(tdens[w].t, omega0, total_time=total_time)
        # can_params[2]['trigger'] = start
        # can_params[2]['total_time'] = len(sol.t) / can_params[2]['sampling_rate']  
        parameters['n_taps'] = 99
        parameters['trigger'] = max(can_params[2]['trigger'], 0)
        parameters['roi'] = 0.5* (parameters['total_time'] - parameters['trigger'])
        parameters['sampling_rate'] = can_params[2]['sampling_rate']
        pix = ffta.pixel.Pixel(Z, parameters, filter_amplitude=True, method = 'hilbert', 
                               fit=False, roi=total_time, total_time=total_time)
        # pix = calc_tfp(Z, can_params, method='hilbert')
        pix.n_taps = 99
        pix.analyze()
        f_means.append(pix.inst_freq[int(pix.n_points/2):-1000].mean())
        
    tdens['f_means'] = f_means
