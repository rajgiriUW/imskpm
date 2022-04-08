'''
Fitting Functions
'''

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Assumes frequency as x-axis
def expf(f, y0, a, tau):
    return y0 + a * np.exp(-1/(f*tau))

def expf_(f, y0, a, tau):
    
    return y0 + 0.5*a + tau * a * f * (1 - np.exp(-1/(2*f*tau)))

def cost_fit(f, data, init):
    '''
    f  = frequency access ("widths")
    data = data to fit ("v_means")
    init = initial guess, good example is [v_means[0], np.max(v_means) - np.min(v_means), 1e-7]
    '''
    cost = lambda p: np.sum((expf(f, *p) - data)**2)
#     popt = minimize(cost, init, options={'disp': False},
# 					bounds=[(-10000, -1.0),
# 							(5e-7, 0.1),
# 							(1e-5, 0.1)])

    popt = minimize(cost, init, options={'disp': False})
 					
    return popt

def expf2(params, f, data):
    
    y0 = params['y0']
    a = params['a']
    tau = params['tau']
    
    model = y0 + 0.5*a + tau * a * f * (1 - np.exp(-1/(2*f*tau)))
     
    return (model - data)**2