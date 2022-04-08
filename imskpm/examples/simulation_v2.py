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


#%% Function wrappers for calculations

# 


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
