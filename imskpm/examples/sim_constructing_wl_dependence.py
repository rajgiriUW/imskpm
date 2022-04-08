# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 16:23:14 2020

@author: Raj
"""


import pandas as pd
mapi = pd.read_csv('C:/Users/Raj/Dropbox/UW Work/Grants/DOE_SPM_Zhu_2020/Figures/Simulation/MAPI.csv')
mapi = mapi.set_index('Wavelength (nm)')

passes = 4
for p in range(passes):
    mapi['Absorbance (AU)'] = sps.fftconvolve(mapi['Absorbance (AU)'], np.ones(4)/4, mode='same')
    
#%%
wl_lo = 400
wl_hi = 1000

'''
At each wavelength, use the absorbance value to calculate charge generated
Then, calculate the omega0 and voltage

How will this not simply look like an absorbance curve?


'''

thickness = 500e-7 # in centimeters
resf = 350e3
lift = 20e-9

intensity = 1 # in W/cm^2. 0.1 = 1 Sun, 100 mW / cm^2
k1 = 1e6
k2 = 1e-10

start = 1e-9
width = 0.1e-9
pulse_time = 13e-9
total_time = 20e-3
rise = 1e-7#1e-5    
if rise ==0:
    square = True
else:
    square = False

#%%

wl_spectra = np.arange(wl_lo,wl_hi,20)

v_spectra = np.zeros(len(wl_spectra))
w0_spectra = np.zeros(len(wl_spectra))

widths = np.array([4e-7, 1e-06, 4e-6, 1e-05, 4e-5, 1e-04, 4e-4, 1e-03, 4e-3])

v_spec_df = pd.DataFrame(index = wl_spectra)
w0_spec_df = pd.DataFrame(index = wl_spectra)

w = 1e-4

for w in widths:
   
    v_spectra = np.zeros(len(wl_spectra))
    w0_spectra = np.zeros(len(wl_spectra))
    
    for n, wl in enumerate(wl_spectra):
            
        st = w/4
        pulse_time = 2*w
        absorbance = mapi.loc[wl]['Absorbance (AU)']
        absorbance = (1 - 10**(-absorbance)) * intensity # W/cm^2 absorbed
        n_dens, sol, gen, impulse = calc_n(intensity, absorbance = absorbance, k1=k1, k2=k2, rise=rise, fall=rise, 
                                           square=square, total_time=total_time, pulse_time=pulse_time, 
                                           width=w, start=st, imskpm=True, maxcycles=30)
        
        voltage = calc_gauss_volt(n_dens)
        omega0 = calc_omega0(voltage)
        v_spectra[n] = voltage.mean()
        w0_spectra[n] = omega0.mean()
        # print(mapi.loc[wl], voltage.mean())
        
    v_spec_df[int(w**-1)] = v_spectra
    w0_spec_df[int(w**-1)] = w0_spectra
    print ('Frequency', w**-1, ';', v_spectra.mean())
    
    # plot()
#%%
cm = np.linspace(0.02,0.95,len(w0_spec_df.index.values))
fig, ax = plt.subplots() 
for v, c in zip(w0_spec_df.index.values, cm):
    ax.semilogx(widths, unity_norm(w0_spec_df.loc[v]-350e3), linewidth=2, linestyle='--', marker='s', color=plt.cm.jet(c))
    
#%% Wl-dependent trEFM analysis

