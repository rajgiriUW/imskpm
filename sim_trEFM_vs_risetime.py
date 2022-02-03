# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 11:47:24 2020

@author: Raj
"""

'''
This checks the trEFM response as a function of risetime to validate the technique

'''

def unity_norm(arr):
    
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

#%%   Comparing the zoom in pixel thing for David

lift = 20e-9 
thickness = 500e-7
resf = 350000

k1 = 1e5
k2 = 1e-11
rise = 5e-7

start = 5e-4
width = 2e-3
pulse_time = 4e-3
total_time = 4e-3

if rise ==0:
    square = True
else:
    square = False
    
n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start)
voltage = calc_gauss_volt(n_dens, lift=lift)
omega0 = calc_omega0(voltage, resf = resf)
Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
pix = calc_tfp(Z, can_params, method='hilbert')

pix5nsk11e5 = pix
sol5nsk11e5 = sol

k1 = 1e6
k2 = 1e-10

n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start)
voltage = calc_gauss_volt(n_dens, lift=lift)
omega0 = calc_omega0(voltage, resf = resf)
Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
pix = calc_tfp(Z, can_params, method='hilbert')

pix5nsk11e6 = pix
sol5nsk11e6 = sol

## Rise time change
k1 = 1e5
k2 = 1e-11
rise = 1e-4

n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start)
voltage = calc_gauss_volt(n_dens, lift=lift)
omega0 = calc_omega0(voltage, resf = resf)
Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
pix = calc_tfp(Z, can_params, method='hilbert')

pix100usk11e5 = pix
sol100usk11e5 = sol

k1 = 1e6
k2 = 1e-10
rise = 1e-4

n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start)
voltage = calc_gauss_volt(n_dens, lift=lift)
omega0 = calc_omega0(voltage, resf = resf)
Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
pix = calc_tfp(Z, can_params, method='hilbert')

pix100usk11e6 = pix
sol100usk11e6 = sol

#%%   Comparing the zoom in pixel thing for David, vs fall time

lift = 20e-9 
thickness = 500e-7
resf = 350000
intensity=10
scale=1

k1 = 1e5
k2 = 1e-11
rise = 5e-7

start = 5e-4
width = 2e-3
pulse_time = 4e-3
total_time = 5e-3

if rise ==0:
    square = True
else:
    square = False
    
n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start)
voltage = calc_gauss_volt(n_dens, lift=lift)
omega0 = calc_omega0(voltage, resf = resf)
Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
pix = calc_tfp(Z, can_params, method='hilbert', trigger=start+width, recombination=True, roi=1e-3)

pix5nsk11e5 = pix
sol5nsk11e5 = sol

k1 = 1e6
k2 = 1e-10

n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start)
voltage = calc_gauss_volt(n_dens, lift=lift)
omega0 = calc_omega0(voltage, resf = resf)
# Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
Z, can_params = calc_cantsim(sol.t, omega0 + (omega0-resf)*scale, total_time=total_time) # To avoid some numerical issues
pix = calc_tfp(Z, can_params, method='hilbert', trigger=start+width, recombination=True, roi=1e-3)

pix5nsk11e6 = pix
sol5nsk11e6 = sol

k1 = 1e7
k2 = 1e-9

n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start, dt=1e-7)
voltage = calc_gauss_volt(n_dens, lift=lift)
omega0 = calc_omega0(voltage, resf = resf)
# Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
Z, can_params = calc_cantsim(sol.t, omega0 + (omega0-resf)*scale**2, total_time=total_time) # To avoid some numerical issues
pix = calc_tfp(Z, can_params, method='hilbert', trigger=start+width, recombination=True, roi=1e-3)

pix5nsk11e7 = pix
sol5nsk11e7 = sol

## Rise time change
k1 = 1e5
k2 = 1e-11
rise = 1e-4

n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start, dt=1e-7)
voltage = calc_gauss_volt(n_dens, lift=lift)
omega0 = calc_omega0(voltage, resf = resf)
Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
pix = calc_tfp(Z, can_params, method='hilbert', trigger=start+width, recombination=True, roi=1e-3)

pix100usk11e5 = pix
sol100usk11e5 = sol

k1 = 1e6
k2 = 1e-10

n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start, dt=1e-7)
voltage = calc_gauss_volt(n_dens, lift=lift)
omega0 = calc_omega0(voltage, resf = resf)
# Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
Z, can_params = calc_cantsim(sol.t, omega0 + (omega0-resf)*scale, total_time=total_time) # To avoid some numerical issues
pix = calc_tfp(Z, can_params, method='hilbert', trigger=start+width, recombination=True, roi=1e-3)

pix100usk11e6 = pix
sol100usk11e6 = sol

k1 = 1e7
k2 = 1e-9

n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start, dt=1e-7)
voltage = calc_gauss_volt(n_dens, lift=lift)
omega0 = calc_omega0(voltage, resf = resf)
# Z, can_params = calc_cantsim(sol.t, omega0, total_time=total_time)
Z, can_params = calc_cantsim(sol.t, omega0 + (omega0-resf)*scale**2, total_time=total_time) # To avoid some numerical issues
pix = calc_tfp(Z, can_params, method='hilbert', trigger=start+width, recombination=True, roi=1e-3)

pix100usk11e7 = pix
sol100usk11e7 = sol


#%%
fig, ax = plt.subplots(nrows=2, facecolor='white', figsize = (8, 6))
plt.rcParams.update({'font.size': 15})

st = 4500
sp = 20000

kwargs= {'n_taps': 99, 'roi':1e-3, 'method':'hilbert', 'recombination':True, 'trigger':500e-6}

for k, v in kwargs.items():
    print(k, v)
    setattr(pix100usk11e5, k, v)
    setattr(pix100usk11e6, k, v)
    setattr(pix5nsk11e5, k, v)
    setattr(pix5nsk11e6, k, v)
    setattr(pix5nsk11e7, k, v)
    setattr(pix100usk11e7, k, v)

pix100usk11e5.analyze()
pix100usk11e6.analyze()
pix5nsk11e5.analyze()
pix5nsk11e6.analyze()
pix100usk11e7.analyze()
pix5nsk11e7.analyze()

ax[0].plot(sol100usk11e6.t[st:sp]*1e6, unity_norm(pix100usk11e6.inst_freq[st:sp]), 'b', linewidth=2)
ax[0].plot(sol100usk11e5.t[st:sp]*1e6, unity_norm(pix100usk11e5.inst_freq[st:sp]), 'xkcd:orange', linewidth=2)
#ax[0].plot(sol100usk11e7.t[st:sp]*1e6, unity_norm(pix100usk11e7.inst_freq[st:sp]), 'xkcd:pea green', linewidth=2)

st = 24500
sp = 30000

kwargs= {'trigger': 2500e-6}

for k, v in kwargs.items():
    print(k, v)
    setattr(pix100usk11e5, k, v)
    setattr(pix100usk11e6, k, v)
    setattr(pix5nsk11e5, k, v)
    setattr(pix5nsk11e6, k, v)
    setattr(pix5nsk11e7, k, v)
    setattr(pix100usk11e7, k, v)

pix100usk11e5.analyze()
pix100usk11e6.analyze()
pix5nsk11e5.analyze()
pix5nsk11e6.analyze()
pix100usk11e7.analyze()
pix5nsk11e7.analyze()

ax[1].plot(sol100usk11e6.t[st:sp]*1e6, unity_norm(pix100usk11e6.inst_freq[st:sp]), 'b', linewidth=2)
ax[1].plot(sol100usk11e5.t[st:sp]*1e6, unity_norm(pix100usk11e5.inst_freq[st:sp]), 'xkcd:orange', linewidth=2)
#ax[1].plot(sol100usk11e7.t[st:sp]*1e6, unity_norm(pix100usk11e7.inst_freq[st:sp]), 'xkcd:pea green', linewidth=2)

ax[0].legend(labels=['k1 = 10$^6$ /s', 'k1 = 10$^5$ /s'])
ax[0].set_ylabel('Norm. Frequency (a.u)')
ax[0].set_xlabel('Time ($\mu$s)')
ax[1].set_ylabel('Norm. Frequency (a.u.)')
ax[1].set_xlabel('Time ($\mu$s)')
ax[0].set_title('Comparing Rise Time Response, Intensity='+str(intensity*1000)+'mW/cm$^2$, k1='+str(k1))
plt.tight_layout()
#%%
fig, ax = plt.subplots(nrows=2, facecolor='white', figsize = (8, 6))
plt.rcParams.update({'font.size': 15})

st = 4500
sp = 20000
kwargs= {'n_taps': 99, 'roi':1e-3, 'method':'hilbert', 'recombination':True, 'trigger':500e-6}

for k, v in kwargs.items():
    print(k, v)
    setattr(pix100usk11e5, k, v)
    setattr(pix100usk11e6, k, v)
    setattr(pix5nsk11e5, k, v)
    setattr(pix5nsk11e6, k, v)
    setattr(pix5nsk11e7, k, v)
    setattr(pix100usk11e7, k, v)

pix100usk11e5.analyze()
pix100usk11e6.analyze()
pix5nsk11e5.analyze()
pix5nsk11e6.analyze()
pix100usk11e7.analyze()
pix5nsk11e7.analyze()

ax[0].plot(sol5nsk11e6.t[st:sp]*1e6, unity_norm(pix5nsk11e6.inst_freq[st:sp]), 'b', linewidth=2)
ax[0].plot(sol5nsk11e5.t[st:sp]*1e6, unity_norm(pix5nsk11e5.inst_freq[st:sp]), 'xkcd:orange', linewidth=2)
# ax[0].plot(sol5nsk11e7.t[st:sp]*1e6, unity_norm(pix5nsk11e7.inst_freq[st:sp]), 'xkcd:pea green', linewidth=2)

st = 24500
sp = 30000

kwargs= {'trigger': 2500e-6}

for k, v in kwargs.items():
    print(k, v)
    setattr(pix100usk11e5, k, v)
    setattr(pix100usk11e6, k, v)
    setattr(pix5nsk11e5, k, v)
    setattr(pix5nsk11e6, k, v)
    setattr(pix5nsk11e7, k, v)
    setattr(pix100usk11e7, k, v)
    
ax[1].plot(sol5nsk11e6.t[st:sp]*1e6, unity_norm(pix5nsk11e6.inst_freq[st:sp]), 'b', linewidth=2)
ax[1].plot(sol5nsk11e5.t[st:sp]*1e6, unity_norm(pix5nsk11e5.inst_freq[st:sp]), 'xkcd:orange', linewidth=2)
# ax[1].plot(sol5nsk11e7.t[st:sp]*1e6, unity_norm(pix5nsk11e7.inst_freq[st:sp]), 'xkcd:pea green', linewidth=2)
# 
ax[0].legend(labels=['k1 = 10$^6$ /s', 'k1 = 10$^5$ /s'])
ax[0].set_ylabel('Norm. Frequency (a.u)')
ax[0].set_xlabel('Time ($\mu$s)')
ax[1].set_ylabel('Norm. Frequency (a.u.)')
ax[1].set_xlabel('Time ($\mu$s)')
ax[0].set_title('Comparing Rise Time Response, Intensity='+str(intensity*1000)+'mW/cm$^2$, k1='+str(k1))
plt.tight_layout()