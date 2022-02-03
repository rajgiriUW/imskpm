# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 20:53:42 2020

@author: Raj
"""


import numpy as np
import ffta
from matplotlib import pyplot as plt
sim_params = ({'amp_invols': 1.001e-07, 'def_invols': 9.24e-08, 'soft_amp': 0.3,  
               'drive_freq': 348261.0,  'res_freq': 348261.0,  'k': 24.9,  'q_factor': 384.0},
		 {'es_force': 3.2e-09, 'delta_freq': -73.0, 'tau': 1e-08}, 
         {'trigger': 0.001, 'total_time': 0.01, 'sampling_rate': 10000000.0})
cant = ffta.simulation.mechanical_drive.MechanicalDrive(*sim_params)
cant.func_args = [1e-4]
y_t, _ = cant.simulate()
cant.create_parameters()
parameters = cant.parameters
pix = ffta.pixel.Pixel(y_t, parameters, filter_amplitude=True, method='hilbert')
tx = np.arange(0, pix.total_time, 1 / pix.sampling_rate)
pix.n_taps = 499
_, _, inst_freq = pix.analyze()

# construct single exponential
exp_out = np.ones(len(tx)) * cant.res_freq
t0 = cant.trigger
for n, t in enumerate(tx):

    if t >= t0:
        exp_out[n] = cant.res_freq + cant.delta_freq * cant.func(t - t0, *cant.func_args)
    
plt.rcParams.update({'font.size': 19})
fig, ax  = plt.subplots(facecolor='white')
ax.plot(tx*1e3, pix.inst_freq + cant.res_freq, 'r', label='Inst. Freq.', linewidth=4) 
# ax2 = ax.twinx()
ax.plot(tx*1e3, exp_out, 'b', label='Res. Freq.', linewidth=4)
ax.set_xlim(0.9, 2.5)
# ax2.set_xlim(0.9, 2.5)
ax.set_ylim(347850, 348280) # 10 us rise time
# ax2.set_ylim(348180, 348280)
ax.set_title(str(cant.func_args[0]* 1e3) + ' ms rise time')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Frequency (Hz)')
# ax2.set_ylabel('Resonance Frequency (Hz)')
ax.legend()
plt.tight_layout()

taus = np.logspace(-7, -3, 10)
tfps = np.zeros(len(taus))
for n, t in enumerate(taus):
    
    cant.func_args = [t]
    y_t, _ = cant.simulate()
    if t < 1e-4:
        roi = 3e-4
    else:
        roi = 2e-3
    pix = ffta.pixel.Pixel(y_t, parameters, filter_amplitude=True, method='hilbert', roi=roi)
    pix.n_taps = 499
    _, _, inst_freq = pix.analyze()
    tfps[n] = pix.tfp
    
fig, ax  = plt.subplots(facecolor='white')
ax.loglog(taus*1e6, tfps*1e6, 'rs', linestyle='--')
ax.set_xlabel(r'$\tau $ ($\mu$s)')
ax.set_ylabel('t$_{fp}$ ($\mu$s)')
plt.tight_layout()
