# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 12:43:16 2020

@author: Raj
"""


''' 
IM-SKPM simulation and resultant curve-fitting

'''

# If loading a pickle
 # with open(total_dens, 'rb') as handle:
    # total_dens = pickle.load(handle)

# total_dens = pickle.load(open('E:/Raj DOE SImulation Pickles/pkl/Total_Dens_100Sun.pkl', 'rb'))
#%% Loop through rise times
'''
This simulates many IM-SKPM runs

Widths = 1/frequency of the pulse (e.g. 4e-8 = 40 ns)
Intensity = W/cm^2 intensity; 0.1 W/cm^2 = 1 Sun
K1, k2 = rate constants in /s and cm^3/s
rise, fall = rise and fall times for the light source
k1_vals and k2_vals are arrays of k1 and k2 values to loop through
rise_vals is rise times to loop through.

The output is a giant (~0.5 GB ) dictionary total_dens, containing keys:
    widths
    v_means (average voltage)
    omega0 (frequency shift)
    (n_dens, sol.t, gen) (tuple of charge density calculated, time axis for ODE integration, and charge generated via light)
    
total_dens[5e-7][1e6] would be the total

'''
widths = np.array([4e-8, 1e-7, 4e-7, 1e-06, 4e-6, 1e-05, 4e-5, 1e-04, 4e-4, 1e-03, 4e-3])

intensity = 0.1
k1 = 1e5   #1e6, 1e5, 1e4, 1e7, 1e8
k2 = 1e-11  #1e-10, 1e-11, 1e-9, 1e-8
rise = 1e-4
fall = 1e-4

k1_vals = [1e5, 1e6, 1e7]
k2_vals = [1e-11,1e-10, 1e-9]
rise_vals = [0, 5e-7, 1e-5, 1e-4]

total_dens = {}
for r in rise_vals:

    total_dens[r] = {}    
    if r == 0:
        square = True # square pulse limit
    else:
        square = False
    for k1v, k2v in zip(k1_vals, k2_vals): 
        
        v_means = []
        omega0_means = []
        total_dens[r][k1v] = {}
        print('k1',k1v, 'k2', k2v, 'intensity', intensity)
        for w in widths:
            
            print(w)
            st = w/4
            pulse_time = 2*w
            n_dens, sol, gen, impulse = calc_n(intensity, k1=k1v, k2=k2v, rise=r,fall=r, 
                                               square=square, total_time=total_time, pulse_time=pulse_time, 
                                               width=w, start=st, imskpm=True, maxcycles=30)
            voltage = calc_gauss_volt(n_dens)
            omega0 = calc_omega0(voltage)
            v_means.append(voltage.mean())
            omega0_means.append(omega0.mean())
            total_dens[r][k1v][w] = (n_dens, sol.t, gen)
                
            if square:
                name = str(np.round(w**-1,0)) + ' Hz, k1,' + str(np.round(k1v,0)) + '_k2,' + str(k2v) + \
                    '_risefall,000_' + str(intensity*1000) +'.jpg'
            else:
                name = str(np.round(w**-1,0)) + ' Hz, k1,' + str(np.round(k1v,0)) + '_k2,' + str(k2v) + \
                    '_risefall,' + str(r) + '_' + str(intensity*1000) +'.jpg'

            fig, ax = plt.subplots(nrows=1,figsize=(6,4),facecolor='white')
            ax.plot(sol.t, voltage, 'g')
            ax.set_ylabel('Voltage (V)')
            ax.set_xlabel('Time (s)')
            vmean = voltage.mean() * np.ones(len(sol.t))
            ax.plot(sol.t, vmean, 'r--')
            ax.set_title(str(w**-1) + ' Hz')
            plt.tight_layout()
            plt.savefig(name)
            
            plt.close()
    
        total_dens[r][k1v]['v_means'] = np.copy(v_means)
        total_dens[r][k1v]['omega0_means'] = np.copy(omega0_means)
        total_dens[r][k1v]['widths'] = widths

for k1 in k1_vals:
    fig, ax = plt.subplots(facecolor='white', figsize=(10,6))
    ax.set_xlabel('Pulse Frequency (Hz)')
    ax.set_ylabel('Average Voltage (V)')
    ax.semilogx(total_dens[0][k1]['widths']**-1, total_dens[0][k1]['v_means'], 'k^', markersize=8, linestyle='--')
    ax.semilogx(total_dens[5e-7][k1]['widths']**-1, total_dens[5e-7][k1]['v_means'], 'bs', markersize=8, linestyle='--')
    ax.semilogx(total_dens[1e-5][k1]['widths']**-1, total_dens[1e-5][k1]['v_means'], 'ro', markersize=8, linestyle='--')
    ax.semilogx(total_dens[1e-4][k1]['widths']**-1, total_dens[1e-4][k1]['v_means'], 'gd', markersize=8, linestyle='--')
    ax.set_title('k1='+str(k1)+' /s vs rise time')
    ax.legend(labels=[0, 5e-7, 1e-5, 1e-4])
    plt.tight_layout()
    plt.savefig('Comparing_RiseTimes_at_k1,'+ str(k1) +'_intensity' +str(intensity)+'.tiff')
    # print('v_means', v_means)
        # print('w0_means', omega0_means)
    
with open('Total_Dens_' + str(intensity*100) + ' Suns.pkl', 'wb') as handle:
    pickle.dump(total_dens, handle, protocol=pickle.HIGHEST_PROTOCOL)   
    
#%% Simple IMSKPM Curve Fitting


    

# t = total_dens[0][k1]['widths']**-1
t = total_dens[0][1e7]['widths']
# y = total_dens[0][k1]['v_means']
y = total_dens[0][1e7]['v_means']
k1_vals = [k for k in total_dens[0].keys()]
rise_vals = [r for r in total_dens.keys()]

# popt = cost_fit(t, y, [0.6, 0.4, 1e-8])

for k1 in k1_vals:
    fig, ax = plt.subplots(facecolor='white', figsize=(10, 6))
    ax.set_xlabel('Pulse Frequency (Hz)')
    ax.set_ylabel('Average Voltage (V)')
    
    for r, c, m in zip(rise_vals, ['k', 'b', 'r', 'g'], ['^', 's', 'o', 'd']):
        
        if r > 1e-6:
            t = total_dens[r][k1]['widths'][4:]**-1
            y = total_dens[r][k1]['v_means'][4:]
            tau_g = 1e-4
        else:
            t = total_dens[r][k1]['widths'][3:]**-1
            y = total_dens[r][k1]['v_means'][3:]
            tau_g = 1e-6
        ax.semilogx(t, y, color=c, marker=m, markersize=8, linewidth=0)
        popt = cost_fit(t, y, [y[-1], np.max(y) - np.min(y), 1/k1])
        # popt = cost_fit(t, y, [np.max(y) - np.min(y), 1/k1])
        print(popt.x)
        _t = np.logspace(2, 7.3, 85)
        ax.semilogx(_t, expf(_t, *popt.x) , color=c, linewidth=2, linestyle='--', label=np.round(popt.x[2]*1e9,1))
        # ax.semilogx(_t, expf(_t, *[*popt.x[:-1], 1/k1]) , color=c, linewidth=1, linestyle='-.', label=np.round(popt.x[2]*1e9,1))
        
    ax.set_title('k1='+str(k1)+' /s vs rise time, ' + str(intensity*1000) + ' mW/cm$^2$')
    ax.legend()
    plt.tight_layout()
    plt.savefig('Comparison_' + 'k1='+str(k1)+' pers vs rise time, ' + str(intensity*1000) + ' mWcm$^2$.tiff')
#%%  Showing bad and good fits

r = 1e-4
k1 = 1e5
t = total_dens[r][k1]['widths']**-1
y = total_dens[r][k1]['v_means']

fig, ax = plt.subplots(facecolor='white', figsize=(10,6))
ax.set_xlabel('Pulse Frequency (Hz)')
ax.set_ylabel('Average Voltage (V)')
ax.semilogx(t, y, color='g', marker='^', markersize=9, linewidth=0)
popt = cost_fit(t[4:], y[4:], [y[-1], np.max(y[4:]) - np.min(y[4:]), 1/k1])
ax.semilogx(_t, expf(_t, *popt.x) , color='g', linewidth=3, linestyle='--', 
            label=np.round(popt.x[2]*1e9,1))
ax.semilogx(_t, expf(_t, *[*popt.x[:-1], 1/k1]) , color='g', 
            linewidth=1, linestyle='-.', label=np.round(popt.x[2]*1e9,1))
print(popt.x)
r = 1e-5
t = total_dens[r][k1]['widths']**-1
y = total_dens[r][k1]['v_means']

ax.semilogx(t, y, color='r', marker='^', markersize=9, linewidth=0)
popt = cost_fit(t[4:], y[4:], [y[-1], np.max(y[4:]) - np.min(y[4:]), 1/k1])

ax.semilogx(_t, expf(_t, *popt.x) , color='r', linewidth=3, linestyle='-.', 
            label=np.round(popt.x[2]*1e9,1))
ax.semilogx(_t, expf(_t, *[*popt.x[:-1], 1/k1]) , color='r', 
            linewidth=1, linestyle='-.', label=np.round(popt.x[2]*1e9,1))
ax.legend()
ax.set_title('k1='+str(k1)+' /s vs rise time, ' + str(intensity*1000) + ' mW/cm$^2$')
plt.tight_layout()
plt.savefig('Comparison_' + 'k1='+str(k1)+' pers vs rise time, ' + str(intensity*1000) + ' mWcm$^2$_OnlyBad.tiff')

r = 5e-7
t = total_dens[r][k1]['widths']**-1
y = total_dens[r][k1]['v_means']

fig, ax = plt.subplots(facecolor='white', figsize=(10,6))
ax.set_xlabel('Pulse Frequency (Hz)')
ax.set_ylabel('Average Voltage (V)')
ax.semilogx(t, y, color='b', marker='^', markersize=9, linewidth=0)
popt = cost_fit(t[3:], y[3:], [y[-1], np.max(y[3:]) - np.min(y[3:]), 1/k1])
ax.semilogx(_t, expf(_t, *popt.x) , color='b', linewidth=3, linestyle='--', 
            label=np.round(popt.x[2]*1e9,1))
ax.semilogx(_t, expf(_t, *[*popt.x[:-1], 1/k1]) , color='b', 
            linewidth=1, linestyle='-.', label=np.round(popt.x[2]*1e9,1))

r = 0
t = total_dens[r][k1]['widths']**-1
y = total_dens[r][k1]['v_means']

ax.semilogx(t, y, color='k', marker='^', markersize=9, linewidth=0)
popt = cost_fit(t[3:], y[3:], [y[-1], np.max(y[3:]) - np.min(y[3:]), 1/k1])
ax.semilogx(_t, expf(_t, *popt.x) , color='k', linewidth=3, linestyle='-.', 
            label=np.round(popt.x[2]*1e9,1))
ax.semilogx(_t, expf(_t, *[*popt.x[:-1], 1/k1]) , color='k', 
            linewidth=1, linestyle='-.', label=np.round(popt.x[2]*1e9,1))
ax.legend()
ax.set_title('k1='+str(k1)+' /s vs rise time, ' + str(intensity*1000) + ' mW/cm$^2$')
plt.tight_layout()
plt.savefig('Comparison_' + 'k1='+str(k1)+' pers vs rise time, ' + str(intensity*1000) + ' mWcm$^2$_OnlyGood.tiff')

#%% Comparing k1 times

for r in rise_vals:

    fig, ax = plt.subplots(facecolor='white', figsize=(10, 6))
    ax.set_xlabel('Pulse Frequency (Hz)')
    ax.set_ylabel('Average Voltage (V)')
    
    for k1, c, m in zip([1e6, 1e5],['k', 'b', 'r', 'g'], ['^', 's', 'o', 'd']):
        
        if r > 1e-6:
            t = total_dens[r][k1]['widths'][4:]**-1
            y = total_dens[r][k1]['v_means'][4:]
            tau_g = 1e-4
        else:
            t = total_dens[r][k1]['widths'][3:]**-1
            y = total_dens[r][k1]['v_means'][3:]
            tau_g = 1e-6
        ax.semilogx(t, y, color=c, marker=m, markersize=8, linewidth=0)
        popt = cost_fit(t, y, [y[-1], np.max(y) - np.min(y), 1/k1])
        # popt = cost_fit(t, y, [np.max(y) - np.min(y), 1/k1])
        print(popt.x)
        _t = np.logspace(2, 7.3, 85)
        #ax.semilogx(_t, expf(_t, *popt.x) , color=c, linewidth=2, linestyle='--', label=str(r) +'_' + str(np.round(popt.x[2]*1e9,1)))
        # ax.semilogx(_t, expf(_t, *[*popt.x[:-1], 1/k1]) , color=c, linewidth=1, linestyle='-.', label=np.round(popt.x[2]*1e9,1))
        
    ax.set_title('rise time=' +str(r) + '_' + str(intensity*1000) + ' mW/cm$^2$')
    ax.legend()
    plt.tight_layout()
    plt.savefig('Comparison_kvals,' + 'rise time=' +str(r) + '_ ' + str(intensity*1000) + ' mWcm$^2$.tiff')
    
    
def unity_norm(arr):
    
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

for r in rise_vals:

    fig, ax = plt.subplots(facecolor='white', figsize=(10, 6))
    ax.set_xlabel('Modulation Frequency (Hz)')
    ax.set_ylabel('Norm. Photovoltage (V)')
    
    for k1, c, m in zip([1e6, 1e5],['k', 'b', 'r', 'g'], ['^', 's', 'o', 'd']):
        xx= 0 
        if r > 1e-6:
            xx = 4
            t = total_dens[r][k1]['widths']**-1
            y = total_dens[r][k1]['v_means']
            tau_g = 1e-4
        else:
            xx=3
            t = total_dens[r][k1]['widths']**-1
            y = total_dens[r][k1]['v_means']
            
            t = total_dens[r][k1]['widths'][1:]**-1
            y = total_dens[r][k1]['v_means'][1:]
            
            tau_g = 1e-6
        ax.semilogx(t, unity_norm(y), color=c, marker=m, markersize=8, linewidth=2, linestyle='--', label=np.round(k1,0))
        popt = cost_fit(t[xx:], y[xx:], [y[-1], np.max(y[xx:]) - np.min(y[xx:]), 1/k1])
        # popt = cost_fit(t, y, [np.max(y) - np.min(y), 1/k1])
        print(popt.x)
        _t = np.logspace(2.5, 7, 5000)
        ff = interp1d(t, unity_norm(y))
        # ax.semilogx(_t, ff(_t), color=c, linewidth=2, linestyle='--')
        #ax.semilogx(_t, unity_norm(expf(_t, *popt.x)) , color=c, linewidth=2, linestyle='--', label=str(k1) +'_' + str(np.round(popt.x[2]*1e9,1)))
        #ax.semilogx(_t, unity_norm(expf(_t, *[*popt.x[:-1], 1/k1])) , color=c, linewidth=1, linestyle='-.', label=np.round(popt.x[2]*1e9,1))
        
    ax.set_title('rise time=' +str(r) + '_' + str(intensity*1000) + ' mW/cm$^2$')
    ax.legend()
    plt.tight_layout()
    plt.savefig('NOFIT_NormComparison_kvals,' + 'rise time=' +str(r) + '_ ' + str(intensity*1000) + ' mWcm$^2$.tiff')
#%% Lmfit version
from lmfit import Minimizer, Parameters, report_fit
params = Parameters()


def expf2(params, f, data):
    
    y0 = params['y0']
    a = params['a']
    tau = params['tau']
    
    model = y0 + a * np.exp(-1/(f*tau))
     
    return (model - data)**1

def expf1(params, f, data):
    
    y0 = params['y0']
    a = params['a']
    tau = params['tau']
    
    model = y0 + 0.5*a + tau * a * f * (1 - np.exp(-1/(2*f*tau)))
     
    return (model - data)**2


# t = total_dens[0][k1]['widths']**-1
t = total_dens[0][1e7]['widths']
# y = total_dens[0][k1]['v_means']
y = total_dens[0][1e7]['v_means']
k1_vals = [k for k in total_dens[0].keys()]
rise_vals = [r for r in total_dens.keys()]

# popt = cost_fit(t, y, [0.6, 0.4, 1e-8])

for k1 in k1_vals:
    fig, ax = plt.subplots(facecolor='white', figsize=(10,6))
    ax.set_xlabel('Pulse Frequency (Hz)')
    ax.set_ylabel('Average Voltage (V)')
    
    for r, c, m in zip(rise_vals, ['k', 'b', 'r', 'g'], ['^', 's', 'o', 'd']):
        
        if r > 1e-6:
            t = total_dens[r][k1]['widths'][4:]**-1
            y = total_dens[r][k1]['v_means'][4:]
            tau_g = 1e-4
        else:
            t = total_dens[r][k1]['widths'][3:]**-1
            y = total_dens[r][k1]['v_means'][3:]
            tau_g = 1e-6
        ax.semilogx(t, y, color=c, marker=m, markersize=8, linewidth=0)
        params = Parameters()
        params.add('a', value = np.max(y) - np.min(y))
        params.add('tau', value = 1/k1, min = 1e-8, max=1e-3)
        params.add('y0', value =  0)
        minner = Minimizer(expf2, params, fcn_args=(t, y))
        result = minner.minimize(method='tnc')
        
        popt = np.array([result.params[x].value for x in result.params])
        print(popt)
        # popt = cost_fit(t, y, [np.max(y) - np.min(y), 1/k1])
        _t = np.logspace(2, 7.3, 85)
        # ax.semilogx(_t, expf2(_t, *popt) , color=c, linewidth=2, linestyle='--', label=np.round(popt[2]*1e9,1))
        # ax.semilogx(_t, expf(_t, *[*popt.x[:-1], 1/k1]) , color=c, linewidth=1, linestyle='-.', label=np.round(popt.x[2]*1e9,1))
        
    ax.set_title('k1='+str(k1)+' /s vs rise time, ' + str(intensity*1000) + ' mW/cm$^2$')
    ax.legend()
    plt.tight_layout()
    
#%%

fig, ax = plt.subplots(facecolor='white', figsize=(10, 6))
ax.set_xlabel('Modulation Frequency (Hz)')
ax.set_ylabel('Average Voltage (V)')
ax.semilogx(total_dens[5e-7][1e5]['widths']**-1, total_dens[5e-7][1e5]['v_means'], 'r^')
ax.plot(total_dens[5e-7][1e5]['widths']**-1, total_dens[5e-7][1e6]['v_means'], 'go')

fig, ax = plt.subplots(facecolor='white', figsize=(10, 6))
ax.set_xlabel('Modulation Frequency (Hz)')
ax.set_ylabel('Average Voltage (V)')
ax.semilogx(total_dens[5e-7][1e5]['widths']**-1, unity_norm(total_dens[5e-7][1e5]['v_means']), 'r^', markersize=8)
ax.plot(total_dens[5e-7][1e5]['widths']**-1, unity_norm(total_dens[5e-7][1e7]['v_means']), 'bo', markersize=8)
    