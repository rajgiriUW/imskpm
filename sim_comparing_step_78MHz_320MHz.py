# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:01:41 2020

@author: Raj
"""
#%% Scratchpad for comparing tFPs with pulses

# Pixel step file
with open('C:/Users/Raj/PixelSTEP100 Suns.pkl', 'rb') as handle:
    pixSTEP = pickle.load(handle)
with open('C:/Users/Raj/n_densFIlong.pkl', 'rb') as handle:
    n_densFIlong = pickle.load(handle)


#%% Comparing Step and 78 MHz and 320 MHz
'''
This generates the voltage and frequency plots for a given intensity for the 
3 different modulation schemes

Note that the 78 MHz = 0.7% duty cycle, and 320 MHz = 3.2% duty cycle
Step = CW (i.e. no duty cycle)

So a better comparison is scale the intensity

'''

thickness = 500e-7 # in centimeters
resf = 350e3
lift = 20e-9

intensity = 10 # 0.1 = 1 Sun, 100 mW / cm^2
k1 = 1e6
k2 = 1e-10

start = 1e-9
width = 0.1e-9
total_time = 20e-3
rise = 0#1e-5    
if rise ==0:
    square = True
else:
    square = False

# 78 MHz
pulse_time = 13e-9
n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, 
                                   square=square, total_time=total_time, pulse_time=pulse_time, 
                                   width=width, start=start, imskpm=True, maxcycles=1000)
dt = np.round(sol.t[-1] - sol.t[-2], 10)
if not dt:
    dt = np.round(sol.t[-1] - sol.t[-2], 12)
voltageFI = calc_gauss_volt(n_dens, lift=lift)
omega0FI = calc_omega0(voltageFI, resf = resf)
n_densFI = n_dens
solFI = sol
genFI = gen
print('Average charge generation = ', gen.mean())

# 320 MHz
pulse_time = 3e-9
n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, 
                                   square=square, total_time=total_time, pulse_time=pulse_time, 
                                   width=width, start=start, imskpm=True, maxcycles=1000)
dt = np.round(sol.t[-1] - sol.t[-2], 10)
if not dt:
    dt = np.round(sol.t[-1] - sol.t[-2], 12)
print('dt', dt)
voltageOCT = calc_gauss_volt(n_dens, lift=lift)
omega0OCT = calc_omega0(voltageOCT, resf = resf)
n_densOCT = n_dens
solOCT = sol
genOCT = gen 

# Step
start = 1e-6
width = 4e-6
pulse_time = 1e-5
total_time = 1e-5
rise = 0 
if rise ==0:
    square = True
else:
    square = False
    
n_dens, sol, gen, impulse = calc_n(intensity, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start)
dt = np.round(sol.t[-1] - sol.t[-2], 9)
voltageSTEP = calc_gauss_volt(n_dens, lift=lift)
omega0STEP = calc_omega0(voltageSTEP, resf = resf)
solSTEP = sol
n_densSTEP = n_dens
genSTEP = gen

fig, ax = plt.subplots(nrows=2,figsize=(8,8),facecolor='white')
#ax.plot(sol.t, delta_f, 'g')
ax[0].plot(solSTEP.t*1e6 - 1, omega0STEP - resf, 'r', label='Ideal')
ax[1].plot(solSTEP.t*1e6 - 1, voltageSTEP, 'r', label='Ideal')
ax[0].set_ylabel('Frequency shift (Hz)')
ax[1].set_ylabel('Voltage (V)')
ax[0].set_xlabel(r'Time ($\mu$s)')
ax[1].set_xlabel('Time ($\mu$s)')
ax[0].set_title('Frequency Shift (Hz), intensity=' + str(intensity*1000) + ' $mW/cm^2$, k1=' + str(k1) + '/s')
ax[1].set_title('Voltage (V) at lift height ' + str(lift) + ' nm')
plt.tight_layout()
ax[0].plot(solOCT.t*1e6, omega0OCT - resf, 'g', label='320 MHz')
ax[1].plot(solOCT.t*1e6, voltageOCT, 'g', label='320 MHz') 
ax[0].plot(solFI.t*1e6, omega0FI - resf, 'b', label='78 MHz')
ax[1].plot(solFI.t*1e6, voltageFI, 'b', label='78 MHz')
ax[0].legend()
ax[1].legend()
ax[0].set_xlim(0, 4)
ax[1].set_xlim(0, 4)
plt.tight_layout()

def unity_norm(arr):
    
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

fig, ax = plt.subplots(nrows=2,figsize=(8,8),facecolor='white')
#ax.plot(sol.t, delta_f, 'g')
ax[0].plot(solSTEP.t[1000:5000]*1e6 - 1, unity_norm(omega0STEP[1000:5000]), 'r', label='Ideal')
ax[1].plot(solSTEP.t[1000:5000]*1e6 - 1, unity_norm(voltageSTEP[1000:5000]), 'r', label='Ideal')
ax[0].set_ylabel('Norm. Freq. Shift (Hz)')
ax[1].set_ylabel('Norm. Voltage (V)')
ax[0].set_xlabel(r'Time ($\mu$s)')
ax[1].set_xlabel('Time ($\mu$s)')
ax[0].set_title('Norm. Freq. Shift (Hz), intensity=' + str(intensity*1000) + ' $mW/cm^2$, k1=' + str(k1) + '/s')
ax[1].set_title('Voltage (V) at lift height ' + str(lift) + ' nm')
plt.tight_layout()
ax[0].plot(solOCT.t*1e6, unity_norm(omega0OCT), 'g', label='320 MHz')
ax[1].plot(solOCT.t*1e6, unity_norm(voltageOCT), 'g', label='320 MHz') 
ax[0].plot(solFI.t*1e6, unity_norm(omega0FI), 'b', label='78 MHz')
ax[1].plot(solFI.t*1e6, unity_norm(voltageFI), 'b', label='78 MHz')
ax[0].legend()
ax[1].legend()
# ax[0].set_xlim(0, 1)
# ax[1].set_xlim(0, 1)
plt.tight_layout()

# Densities
fig, ax = plt.subplots(facecolor='white')
ax.plot(solSTEP.t*1e6-1, n_densSTEP, 'r', label='Ideal')
ax.set_xlabel('Time ($\mu$s)')
ax.set_ylabel('charge density N(t) (/$cm^3$)')
ax.yaxis.label.set_color('red')
ax.set_title('Charge generated with intensity of ' + str(intensity*1000) + ' $mW/cm^2$')
ax.plot(solOCT.t*1e6, n_densOCT, 'g', label='320 MHz')
ax.set_xlabel('Time ($\mu$s)')
ax.set_ylabel('charge density N(t) (/$cm^3$)')
ax.yaxis.label.set_color('red')
ax.set_title('Charge generated with intensity of ' + str(intensity*1000) + ' $mW/cm^2$')

ax.plot(solFI.t*1e6, n_densFI, 'b', label='78 MHz')
ax.set_xlabel('Time ($\mu$s)')
ax.set_ylabel('charge density N(t) (/$cm^3$)')
ax.yaxis.label.set_color('red')
ax.set_title('Charge generated with intensity of ' + str(intensity*1000) + ' $mW/cm^2$')

#%% Comparing trEFM response

'''
This generates the tFP plots using a bunch of corrections and additions to make sure the generated data are
the right size/shape/etc

Note that the 78 MHz = 0.7% duty cycle, and 320 MHz = 3.2% duty cycle
Step = CW (i.e. no duty cycle)

So a better comparison is scale the intensity


'''

thickness = 500e-7 # in centimeters
resf = 350e3
lift = 20e-9

scale78 = 1/0.007
scale320 = 1/0.032

intensity = 0.1 # 0.1 = 1 Sun, 100 mW / cm^2
k1 = 1e5
k2 = 1e-11

start = 1e-9
width = 0.1e-9
total_time = 20e-3
rise = 0#1e-5    
if rise ==0:
    square = True
else:
    square = False

# 78 MHz
pulse_time = 13e-9
scale = scale78
n_dens, sol, gen, impulse = calc_n(intensity*scale, k1=k1, k2=k2, rise=rise,fall=rise, 
                                   square=square, total_time=total_time, pulse_time=pulse_time, 
                                   width=width, start=start, imskpm=True, maxcycles=1000)
print('Average charge generation = ', gen.mean())
solFI = sol
n_densFIlong = n_dens
n_densFIlong = np.append(np.zeros(int(1e-4 / 1e-11)),  n_densFIlong)
n_densFIlong = np.append(n_densFIlong,  np.tile(n_dens[-10000:],5000))
plt.figure(), plt.plot(n_densFIlong)
voltageFIlong = calc_gauss_volt(n_densFIlong, lift=lift)
omega0FIlong = calc_omega0(voltageFIlong, resf = resf)
omega0FIlong_rs = omega0FIlong[::1000]
# This step needed because of small shift effects
if gen.mean() < 1e23:
    print('Scaling frequency shift...')
    Z, can_params = calc_cantsim([1e-8, 2e-8], omega0FIlong_rs + (omega0FIlong_rs-resf)*10, total_time=.000613)
else:
    Z, can_params = calc_cantsim([1e-8, 2e-8], omega0FIlong_rs, total_time=.000613)

_, parameters = ffta.pixel_utils.load.configuration('C:/Users/Raj/OneDrive/UW Work/Documents, Proposals, Papers/Grants and Proposals/2020_DOE_Perovskite_BES/Simulation/parameters.cfg')
parameters['n_taps'] = 1199
parameters['total_time'] = can_params[2]['total_time']
parameters['trigger'] = can_params[2]['trigger']
parameters['roi'] = 0.0002#parameters['total_time'] - parameters['trigger']
parameters['sampling_rate'] = can_params[2]['sampling_rate']
method = 'stft'
pixFI = ffta.pixel.Pixel(Z, parameters, filter_amplitude=True, method = method, 
                       fit=False, trigger=1e-4, total_time = 0.000613)
pixFI.analyze()

# 320 MHz
pulse_time = 3e-9
scale = scale320
n_dens, sol, gen, impulse = calc_n(intensity*scale, k1=k1, k2=k2, rise=rise,fall=rise, 
                                   square=square, total_time=total_time, pulse_time=pulse_time, 
                                   width=width, start=start, imskpm=True, maxcycles=1000)
print('Average charge generation = ', gen.mean())
solOCT = sol
n_densOCTlong = n_dens
n_densOCTlong = np.append(np.zeros(int(1e-4 / 1e-11)),  n_densOCTlong)
n_densOCTlong = np.append(n_densOCTlong,  np.tile(n_dens[-10000:],5000))
plt.figure(), plt.plot(n_densOCTlong)
voltageOCTlong = calc_gauss_volt(n_densOCTlong, lift=lift)
omega0OCTlong = calc_omega0(voltageOCTlong, resf = resf)
omega0OCTlong_rs = omega0OCTlong[::1000]
if gen.mean() < 1e23:
    print('Scaling frequency shift...')
    Z, can_params = calc_cantsim([1e-8, 2e-8], omega0OCTlong_rs + (omega0OCTlong_rs-resf)*10, total_time=.00060299)
else:
    Z, can_params = calc_cantsim([1e-8, 2e-8], omega0OCTlong_rs, total_time=.00060299)
    
_, parameters = ffta.pixel_utils.load.configuration('C:/Users/Raj/OneDrive/UW Work/Documents, Proposals, Papers/Grants and Proposals/2020_DOE_Perovskite_BES/Simulation/parameters.cfg')
parameters['n_taps'] = 1199
parameters['total_time'] = can_params[2]['total_time']
parameters['trigger'] = can_params[2]['trigger']
parameters['roi'] = parameters['total_time'] - parameters['trigger']
parameters['sampling_rate'] = can_params[2]['sampling_rate']
method = 'stft'
pixOCT = ffta.pixel.Pixel(Z, parameters, filter_amplitude=True, method = method, 
                       fit=False, trigger=1e-4, total_time = 0.00060299)
pixOCT.analyze()

# Step
start = 1e-4
width = 4e-4
pulse_time = 0.000613
total_time = 0.000613
rise = 0 
if rise ==0:
    square = True
else:
    square = False
n_dens, sol, gen, impulse = calc_n(intensity*1.5, k1=k1, k2=k2, rise=rise,fall=rise, square=square, 
                                   total_time=total_time, pulse_time = pulse_time, width=width, start=start)
print('Average charge generation = ', gen.mean())
solSTEP = sol
n_densSTEP = n_dens
voltageSTEP = calc_gauss_volt(n_densSTEP, lift=lift)
omega0STEP = calc_omega0(voltageSTEP, resf = resf)
if gen.mean() < 1e23:
    print('Scaling frequency shift...')
    Z, can_params = calc_cantsim([1e-8, 2e-8], omega0STEP + (omega0STEP-resf)*10, total_time=.000613)
else:
    Z, can_params = calc_cantsim([1e-8, 2e-8], omega0STEP, total_time=.000613)
_, parameters = ffta.pixel_utils.load.configuration('C:/Users/Raj/OneDrive/UW Work/Documents, Proposals, Papers/Grants and Proposals/2020_DOE_Perovskite_BES/Simulation/parameters.cfg')
parameters['n_taps'] = 1199
parameters['total_time'] = can_params[2]['total_time']
parameters['trigger'] = can_params[2]['trigger']
parameters['roi'] = parameters['total_time'] - parameters['trigger']
parameters['sampling_rate'] = can_params[2]['sampling_rate']
method = 'stft'
pixSTEP = ffta.pixel.Pixel(Z, parameters, filter_amplitude=True, method = method, 
                           fit=False, trigger=1e-4, total_time = 0.000613)
pixSTEP.analyze()

with open('pixSTEP_' + str(intensity*10) +'Sun_k1,' + str(k1) + '.pkl', 'wb') as handle:
    pickle.dump(pixSTEP, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('pix320MHZ_' + str(intensity*10) +'Sun_k1,' + str(k1) + '.pkl', 'wb') as handle:
    pickle.dump(pixOCT, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('pix78MHZ_' + str(intensity*10) +'Sun_k1,' + str(k1) + '.pkl', 'wb') as handle:
    pickle.dump(pixFI, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('solSTEP_' + str(intensity*10) +'Sun_k1,' + str(k1) + '.pkl', 'wb') as handle:
    pickle.dump(solSTEP, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('solOCT' + str(intensity*10) +'Sun_k1,' + str(k1) + '.pkl', 'wb') as handle:
    pickle.dump(solOCT, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('solFT_' + str(intensity*10) +'Sun_k1,' + str(k1) + '.pkl', 'wb') as handle:
    pickle.dump(solFI, handle, protocol=pickle.HIGHEST_PROTOCOL)    

# Plot
tx = np.arange(0, 0.000613, 1e-8)
fig, ax = plt.subplots(facecolor='white')
ax.plot(tx*1e6, pixSTEP.inst_freq, 'r', label='Ideal')
ax.plot(tx*1e6, pixFI.inst_freq, 'b', label='78 MHz')
ax.plot(tx[:60299]*1e6, pixOCT.inst_freq, 'g', label='320 MHz')
ax.set_xlabel('Time ($\mu$s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title(str(intensity*1000) + ' $mW/cm^2$, Comparing FFtrEFM')
ax.legend()
ax.set_xlim(80, 400)
ax.set_ylim(-.1, .1)
plt.tight_layout()

fig, ax = plt.subplots(facecolor='white')
ax.plot(tx*1e6, omega0STEP, 'r', label='Ideal')
ax.plot(tx*1e6, omega0FIlong_rs, 'b', label='78 MHz')
ax.plot(tx[:60299]*1e6, omega0OCTlong_rs, 'g', label='320 MHz')
ax.set_xlabel('Time ($\mu$s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title(str(intensity*1000) + ' $mW/cm^2$, Comparing FFtrEFM')
ax.legend()
ax.set_xlim(80, 400)

#%%
soltemp = pickle.load(open('E:/Raj DOE SImulation Pickles/pkl/tFP, step vs train/solFT_100Sun_k1,1000000.0.pkl', 'rb'))
