'''
Fitting Functions
'''

import numpy as np
from scipy.optimize import minimize


def expf_1tau(f, y0, a, tau):
    '''
    Single exponential fit as a function of modulation frequency
    
    Based on:
    Takihara, M.; Takahashi, T.; Ujihara, T. 
    Minority Carrier Lifetime in Polycrystalline Silicon Solar Cells Studied by Photoassisted Kelvin Probe Force Microscopy. 
    Appl. Phys. Lett. 2008, 93 (2), 2006–2009. https://doi.org/10.1063/1.2957468.
    
    Parameters
    ----------
    f : float
        frequency axis.
    y0 : float
        y-offset
    a : float
        amplitude
    tau : float
        time constant 

    Returns
    -------
    float

    '''    
    return y0 + 0.5*a + tau * a * f * (1 - np.exp(-1/(2*f*tau)))

def expf_2tau(f, y0, a, taub, taud):
    '''
    Exponential fit with two lifetimes. This fitting is somewhat more difficult, 
    having an extra variable to control
    
    Based on:
    Fernández Garrillo, P. A.; Borowik, Ł.; Caffy, F.; Demadrille, R.; Grévin, B. 
    Photo-Carrier Multi-Dynamical Imaging at the Nanometer Scale in Organic and Inorganic Solar Cells. 
    ACS Appl. Mater. Interfaces 2016, 8 (45), 31460–31468. https://doi.org/10.1021/acsami.6b11423. 

    Parameters
    ----------
    f : float
        frequency axis.
    y0 : float
        y-offset
    a : float
        amplitude
    taub : float
        time constant (build-up tau)
    taud : float
        time constant (decay tau)
        
    Returns
    -------
    float        
    '''
    
    exptaud = np.exp(-1/(2*f*taud))
    exptaub = np.exp(-1/(2*f*taub))
    
    return y0 + 0.5*a*(1-exptaub*exptaud) + a*f*(taud-taub)*(1-exptaud)*(1-exptaub)

def cost_fit(f, data, init):
    '''
    Uses cost-minimization fitting instead of standard L-Q fitting
    
    This approach requires significantly more debugging and is prone to fit errors
    However, it is generally faster
    
    To access, use IMSKPMSweep.fit(cfit=True)
    
    Parameters
    ----------
    f : float
        frequency axis.
    data : ndarray
        The input data to be minimized; typically IMSKPMSweep.cpd_means
    init : ndarray or list
        The initial guess in order (y0, amplitude, tau)
        e.g.
        >> mn = IMSKPMSweep.cpd_means
        >> init = [mn[0], mn.max() - mn.min(), 1e-7]
        
    Returns
    -------
    popt : array
        List of fit outputs
    '''
    cost = lambda p: np.sum((expf_1tau(f, *p) - data)**2)
    popt = minimize(cost, init, options={'disp': False},
 					bounds=[(-1000, 1000),
 							(-1000, 1000),
 							(1e-10, 0.1)])

    return popt

def expf_lm(params, f, data):
    '''
    Planning ahead if want to use lmfit or equivalent package.
    
    '''    
    y0 = params['y0']
    a = params['a']
    tau = params['tau']
    
    model = y0 + 0.5*a + tau * a * f * (1 - np.exp(-1/(2*f*tau)))
     
    return (model - data)**2