# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 15:26:28 2020

@author: Raj
"""

import numpy as np
from scipy.integrate import solve_ivp
from .calc_utils import gen_t
from .calc_utils import calc_gauss_volt, calc_omega0
from .pulses import pulse
from .odes import dn_dt_g

from matplotlib import pyplot as plt
import warnings

class IMSKPMPoint:
    '''
    Generates a single IMSKPM sweep at a particular frequency

    intensity : float
        incident light intensity in W/cm^2 (0.1 = 100 mW/cm^2 = 1 Sun)
    k1 : float
        recombination, first-order (s^-1). The default is 1e6.
    k2 : float
        recombination, bimolecular (cm^3/s). The default is 1e-10.
    k3 : float
        recombination, Auger (cm^6/s). Default is 0
    thickness : float, optional
        user-specified layer thickness (m)

    Usage
    >> import imskpm
    >> from imskpm.imskpmpoint import IMSKPMPoint
    >> device = IMSKPMPoint()
    >> frequency = 100 # 100 Hz
    >> device.make_pulse(0,0,1/frequency,1/(4*frequency),1/(2*frequency))
    >> device.simulate()
    >> device.plot()

    * Add multiple pulses in a row to simulate a function generator
    >> device.pulse_train(total_time=2e-3, max_cycles=20) # repeat pulse up to 2 ms in length
    >> import matplotlib.pyplot as plt
    >> plt.plot(device.tx, device.pulse, 'b') # visualize the pulse train
    >> device.simulate()
    >> device.plot()

    * Change recombination parameters
    >> device.k1 = 1e5 # change recombination parameters
    >> device.k2 = 1e-9
    >> device.simulate()
    >> device.plot()

    * Add a realistic rise and fall time to the laser
    >> device.make_pulse(rise=1e-7,fall=1e-7,1/frequency,1/(4*frequency),1/(2*frequency))
    >> device.pulse_train(total_time=2e-3, max_cycles=20)
    >> device.simulate()
    >> device.plot()

    * Change the simulation step-size
    >> device.dt = 1e-5 #default = 1e-7 = 100 ns

    * Change the simulation evaluation size (which time steps to actually evaluate)
    >> device.interpolation = 4 # every 4 time steps

    Attributes
    ----------
    voltage : ndArray
        The calculated voltage via Gauss's law (V)
    omega0 : ndArray
        Resonance frequency shift of the cantilever (Hz)
    n_dens : ndArray
        Charge density in the film due to ODE (Generation - Recombination) (#/cm^3).
    sol :  `OdeSolution`
        (From Scipy) Found solution as `OdeSolution` instance
    gen : ndArray
        Carrier concentration GENERATED (#/cm^3).

    '''
    def __init__(self,
                 intensity = 0.1,
                 k1 = 1e6,
                 k2 = 1e-10,
                 k3 = 0,
                 thickness = 500e-7):

        # Simulation parameters
        self.dt = 1e-7
        self.func = dn_dt_g # simulation ODE function

        # Active layer parameters
        self.kinetics(k1, k2, k3, absorbance=1)
        self.thickness = thickness
        self.lift_height = 20e-9

        # Excitation parameters
        self.intensity = intensity #Incident intensity (W/cm^2, 0.1 = 1 Sun)
        self.exc_source(intensity, wl=455e-9, NA=0.6)
        self.make_pulse()

        return

    def exc_source(self, intensity = None, wl = None, NA = None):
        '''Defines the incident excitation source light area+wavelength'''
        if intensity is not None:
            self.intensity = intensity
        if wl is not None:
            self.wavelength = wl
        if NA is not None:
            self.NA = NA

        self.area = 1e4 * np.pi * 0.5 * (0.61 * wl / NA)**2 # Rayleigh criterion in cm^2

        return

    def make_pulse(self, rise = 0, fall = 0, pulse_time = 10e-3,
                   start_time = 2.5e-3, pulse_width = 5e-3):
        '''
        Creates a single light pulse. The amplitude is defined by the attribute
        intensity. If you want to use a different incident intensity, then you need
        to re-call this function after setting self.intensity.

        For a given frequency F, you can use these as reasonable values:
            pulse_time = 1/F
            pulse_width = 1/(2*F) (i.e. half the pulse_time)
            start_time = 1/(4*F) (pulse starts about 25% in)

        >> device = IMSKPMPoint() # default intensity is 0.1 = 100 mW/cm^2 = 1 Sun
        >> device.intensity = 1e-4
        >> device.make_pulse(...) with your desired parameters

        You could also do:
        >> device.intensity = 1e-4
        >> device.pulse *= 10 * 1e-4  #factor of 10 because starts at 0.1 by default

        Parameters
        ----------
        rise : float
            Rise time of the pulse (s). The default is 0.
        fall : float
            Fall time of the pulse (s). The default is 0.
        pulse_time : TYPE, optional
            Total time of the pulse (s). The default is 10e-3.
        start_time : TYPE, optional
            Start time (s) of the pulse. The default is 2.5e-3.
        pulse_width : TYPE, optional
            Width of the pulse (s). The default is 5e-3.
        '''
        if rise == 0 and fall == 0:

            self.rise = 0
            self.fall = 0
            self.square = True

        else:

            self.square = False
            self.rise = max(rise, 1e-14)
            self.fall = max(fall, 1e-14)

        self.pulse_time = pulse_time
        self.start_time = start_time
        self.pulse_width = pulse_width
        self.frequency = 1/pulse_time

        if self.dt > self.pulse_time/100:
            self.dt = self.pulse_time*1e-2
        else:
            self.dt = 1e-7

        if self.start_time + self.pulse_width > self.pulse_time:

            warnings.warn('Pulse exceeds total_time, cropping width to match')
            self.pulse_width = self.pulse_time - self.start_time

        self.tx = np.arange(0, self.pulse_time, self.dt)
        self.pulse = pulse(self.tx, self.start_time, self.pulse_width,
                           self.intensity, self.rise, self.fall)

        return

    def pulse_train(self, total_time = None, max_cycles = None):
        '''
        Creates a sequence of pulses by repeating self.pulse
        Total_time sets the maximum time for the sequence.
        Max_cycles will instead create the sequence through tiling

        Parameters
        ----------
        total_time : float, optional
            Total time of the pulse sequence (s). The default is None.
        max_cycles : int, optional
            Total number of cycles. The default is None.

        Raises
        ------
        AttributeError
            If missing both parameters
        '''

        if total_time is None and max_cycles is None:
            raise AttributeError('Must specify either total_time or max_cycles')

        if total_time is None:
            self.pulse = np.tile(self.pulse, max_cycles)
        else:
            cycles = int(total_time // self.pulse_time)

            if max_cycles is not None:
                cycles = min(max_cycles, cycles)

            if cycles > 0:
                self.pulse = np.tile(self.pulse, cycles)

        self.tx = np.arange(0, len(self.pulse)*self.dt, self.dt)

        if len(self.tx) > len(self.pulse):
            self.tx = self.tx[:len(self.pulse)]

        return

    def kinetics(self, k1, k2, k3, absorbance=None):
        '''
        Set the k1, k2, k3, and absorbance via function call rather than explicitly

        Parameters
        ----------
        k1 : float
            recombination, first-order (s^-1). The default is 1e6.
        k2 : float
            recombination, bimolecular (cm^3/s). The default is 1e-10.
        k3 : float
            recombination, Auger (cm^6/s). Default is 0
        absorbance : float, optional
            user-specified absorbance of the layer (a.u.)
        '''
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        if 'NoneType' not in str(type(absorbance)):
            self.absorbance = absorbance

        return

    def calc_n_dot(self):
        '''
        Calculating the integrated charge density given an input pulse

        Simulation is processed with self.func function. See odes.py for more detail

        To use a different function, user must supply self.args and self.init
            (the arguments for the function and initial values for the function)

        Returns
        -------
        n_dens : float
            Char density in the film due to ODE (Generation - Recombination) (#/cm^3).
        sol :  `OdeSolution`
            (From Scipy) Found solution as `OdeSolution` instance
        gen : float
            Carrier concentration GENERATED (#/cm^3).
        '''

        gen = gen_t(self.absorbance, self.pulse, self.thickness) # electrons / cm^3 / s generated

        # Used for computational accuracy
        scale = 1e-4 #1 = /cm^3, 1e-4 = /um^3, 1e2 = /m^2
        gen = gen * scale**3 #(from /cm^3)
        k1 = self.k1
        k2 = self.k2 / scale**3 #(from cm^3/s)
        k3 = self.k3 / scale**6 #(from cm^6/s)

        tx = self.tx[::self.interpolation]
        func = self.func

        if hasattr(self, 'args'):
            sol = solve_ivp(func, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                            args = self.args)
        else:
            sol = solve_ivp(func, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                            args = (k1, k2, k3, gen, tx[1]-tx[0]))

        self._error = False

        # Decrease default step size if solution fails
        if not any(np.where(sol.y.flatten() > 0)[0]):

            #print('error in solve, changing max_step_size')
            self._error = True
            if hasattr(self, 'args'):
                sol = solve_ivp(func, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                                args = self.args, max_step=self.dt)
            else:
                sol = solve_ivp(func, [tx[0], tx[-1]], [gen[0]], t_eval = tx,
                                args = (k1, k2, k3, gen, tx[1]-tx[0]), max_step=self.dt)

        n_dens = sol.y.flatten()

        gen = gen / scale**3
        n_dens = n_dens / scale**3

        return n_dens, sol, gen

    def simulate(self, interpolation=1):
        '''
        Simulates the single pulse

        interpolation : int, optional
            Determines subsampling of the time values (to speed up integration)
            Default is 1

        Attributes
        ----------
        n_dens : float
            Char density in the film due to ODE (Generation - Recombination) (#/cm^3).
        sol :  `OdeSolution`
            (From Scipy) Found solution as `OdeSolution` instance
        gen : float
            Carrier concentration GENERATED (#/cm^3).
        voltage : float
             The calcualted voltage via Gauss's law (V)t
        omega0 : float
            Resonance frequency shift of the cantilever (Hz)
        '''

        self.interpolation = int(interpolation)
        n_dens, sol, gen = self.calc_n_dot()

        self.n_dens = n_dens
        self.sol = sol
        self.gen = gen

        self.voltage = calc_gauss_volt(self.n_dens, self.lift_height, self.thickness)
        self.omega0 = calc_omega0(self.voltage)

        return

    def plot(self, semilog=False, charge_only=True):
        '''
        Plots the calculated voltage
        '''
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        tx = self.sol.t
        fig_voltage, ax_voltage = plt.subplots(nrows=1,figsize=(6,4),facecolor='white')
        ax_voltage.plot(tx*1e6, self.voltage, 'g')
        ax_voltage.set_ylabel('Voltage (V)')
        ax_voltage.set_xlabel(r'Time ($\mu$s)')
        vmean = self.voltage.mean() * np.ones(len(self.sol.t))
        ax_voltage.plot(tx*1e6, vmean, 'r--', label='Voltage Mean: ' + "%.6f" % vmean[0])
        ax_voltage.legend(bbox_to_anchor=(1, 0), loc="lower right", bbox_transform=fig_voltage.transFigure)
        ax_voltage.set_title('Voltage at ' + str(np.round(self.frequency,2)) + ' Hz')
        plt.tight_layout()

        '''
        Plots carrier density
        '''
        fig_dndt, ax_dndt = plt.subplots(nrows=1,figsize=(6,4),facecolor='white')
        if semilog:
            ax_dndt.semilogy(tx*1e6, self.n_dens, 'r', label='Carrier density')
        else:
            ax_dndt.plot(tx*1e6, self.n_dens, 'r', label='Carrier density')
        ax_dndt.set_ylabel(r'Carrier Density ($cm^{-3}$)')
        if not charge_only:
            ax2 = ax_dndt.twinx()
            ax2.plot(tx*1e6, self.gen, 'b', label='Carrier Generated')

            ax2.set_ylabel(r'Carrier Generated ($cm^{-3}$)')
            ax_dndt.legend(bbox_to_anchor=(1, 0), loc="lower right", bbox_transform=fig_dndt.transFigure)
            ax2.legend(bbox_to_anchor=(1, -0.07), loc="lower right", bbox_transform=fig_dndt.transFigure)

        ax_dndt.set_xlabel(r'Time ($\mu$s)')
        ax_dndt.set_title(r'Carriers generated, intensity=' + str(self.intensity*1000) + ' $mW/cm^2$')
        plt.tight_layout()

        '''
        Plots carrier lifetime
        '''
        fig_lifetime, ax_lifetime = plt.subplots(nrows=1,figsize=(6,4),facecolor='white')
        idx = np.where(tx >= self.start_time + self.pulse_width)
        if semilog:
            ax_lifetime.semilogy(tx[idx]*1e6, self.n_dens[idx], 'r', label='Carrier Lifetime')
        else:
            ax_lifetime.plot(tx[idx]*1e6, self.n_dens[idx], 'r', label='Carrier Lifetime')
        ax_lifetime.set_ylabel(r'Carrier Density ($cm^{-3}$)')
        ax_lifetime.set_xlabel(r'Time ($\mu$s)')
        ax_lifetime.set_title(r'Carrier Lifetime, intensity=' + str(self.intensity*1000) + ' $mW/cm^2$')
        plt.tight_layout()

        '''
        Plots carrier lifetime
        '''
        fig_lifetime, ax_lifetime = plt.subplots(nrows=1,figsize=(6,4),facecolor='white')
        idx = np.where(tx >= self.start_time + self.pulse_width)
        if semilog:
            ax_lifetime.semilogy(tx[idx]*1e6, self.n_dens[idx], 'r', label='Carrier Lifetime')
        else:
            ax_lifetime.plot(tx[idx]*1e6, self.n_dens[idx], 'r', label='Carrier Lifetime')
        ax_lifetime.set_ylabel(r'Carrier Density ($cm^{-3}$)')
        ax_lifetime.set_xlabel(r'Time ($\mu$s)')
        ax_lifetime.set_title(r'Carrier Lifetime, intensity=' + str(self.intensity*1000) + ' $mW/cm^2$')
        plt.tight_layout()

        return fig_voltage, fig_dndt, fig_lifetime, ax_voltage, ax_dndt, ax_lifetime
