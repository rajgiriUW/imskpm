'''
Contains defining ODE equations and utilites for simulations
'''

import numpy as np

def gen_t(absorb, pulse, thickness):
    '''
    Calculates number of carriers based on pulse height and absorbance
    cross-section, evaluated at time t, in electrons/second
    
    pulse: ndArray
        incident light intensity in W/cm^2
    
    absorb : float
        absorbance (a.u.)
    
    thickness : float
        film thickness (cm)
    
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

def unity_norm(arr):
    '''
    Scales arr from 0 to 1
    '''
    
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))



