import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

import imskpm
from imskpm.imskpmpoint import IMSKPMPoint
from imskpm.imskpmsweep import IMSKPMSweep

from imskpm.fitting import cost_fit, expf_1tau, expf_2tau

##----------------------------------------------
## Heading and introduction

st.set_page_config(page_title='IM-SKPM')
st.title('Intensity-Modulated Scanning Kelvin Probe Microscopy (IM-SKPM)')
st.subheader('Rajiv Giridharagopal, Ph.D.')
st.subheader('University of Washington, rgiri@uw.edu')

with st.expander('Quick Guide'):
    st.write("Simulating IM-SKPM in photovoltaics based on conventional charge density recombination ODE.")
    st.write("This approach simulates equations of the form:")
    st.latex("\\frac {dn} {dt} = G-k_1n -k_2n^2-k_3n^3")
    st.write("where:")
    st.write("""
* $n$ = carrier density (#$/cm^3$)
* $\\frac {dn} {dt} $ = change in carrier density (#$/{cm^3s}$)
* $G$ = generation rate (#$/cm^3$)
* $k_1$ = monomoecular recombination rate ($/s$), trapping/nonradiative recombination
* $k_2$ = bimolecular recombination rate ($cm^3/s$), band-band/radiative recombination
* $k_3$ = third order recombination rate ($cm^6/s$), Auger recombination""")

st.markdown("""---""")


##----------------------------------------------
## Simulating a single IM-SKPM curve

st.header('Simulating an IM-SKPM curve')

# Determine recombination rates
st.sidebar.subheader('Changing Recombination Rates')
st.sidebar.write('$k_1 (units: /s)$')
k1_input = st.sidebar.number_input('The monomoecular recombination rate, trapping/nonradiative recombination', min_value=1e3, max_value=1e10, value=1e6, format='%e')
st.sidebar.write('$k_2 (units: cm^3/s)$')
k2_input = st.sidebar.number_input('The bimolecular recombination rate, band-band/radiative recombination', min_value=1e-13, max_value=1e-7, value=1e-10, format='%e')
st.sidebar.write('$k_3 (units: cm^6/s)$')
k3_input = st.sidebar.number_input('The third order recombination rate, Auger recombination')

# Determine the excitation
st.sidebar.markdown("""---""")
st.sidebar.subheader('Changing the Excitation')
rise_input = st.sidebar.number_input('Rise Time (s)', min_value=0e0, max_value=1e-3, value=0e0, format='%e')
fall_input = st.sidebar.number_input('Fall Time (s)', min_value=0e0, max_value=1e-3, value=0e0, format='%e')
intensity_input = st.sidebar.number_input('Intensity (0.1 = 100mW/cm^2 = 1 Sun)', min_value=0e0, max_value=1e8, value=1e-1, format='%e')
wl_input = st.sidebar.number_input('Wavelength (nm)', min_value=3e2, max_value=1e3, value=4.55e2, format='%e')
NA_input = st.sidebar.number_input('Numerical Aperture', min_value=0.1, max_value=3.5, value=0.6)

# Determine frequency
st.sidebar.markdown("""---""")
freq_input = st.sidebar.number_input('Frequency (Hz)', min_value=0e0, max_value=1e12, value=1e4, format='%e')
# frequency = 10000 # 10 kHz
frequency = freq_input

# Determine max cycles
st.sidebar.markdown("""---""")
cycles_input = st.sidebar.number_input('Max cycles', min_value=1, max_value=50, value=1)
st.sidebar.markdown("""---""")

with st.expander(label="See Current Values",expanded=False):
    st.write('k1 = ', k1_input, ' , k2 = ', k2_input, ', k3 = ', k3_input)
    st.write('Intensity = ', intensity_input, ', Wavelength = ', wl_input, ', Numerical Aperture', NA_input)
    st.write('Frequency = ', freq_input)
    st.write('Max Cycles = ', cycles_input)

# toggles
col1, col2 = st.columns(2, gap="large")
with col1:
    semilog_input = st.select_slider('Semilog', options=['off','on'], value='off')
with col2:
    charge_input = st.select_slider('Charge only', options=['off', 'on'], value='off')

if semilog_input=='on':
    semilog_input=True
else:
    semilog_input=False

if charge_input=='on':
    charge_input=True
else:
    charge_input=False

# buttons
st.caption("")
st.markdown("**Select the graph to display:**")
b1, b2, b3 = st.columns(3, gap="large")
with b1:
    carrier = st.button("Carriers Generated")
with b2:
    voltage = st.button("Voltage")
with b3:
    lifetime = st.button("Carrier Lifetime")

placeholder = st.empty()

# Plot the result
with st.spinner('Loading graphs...'):
    device = IMSKPMPoint(k1=k1_input, k2=k2_input, k3=k3_input)
    device.exc_source(intensity=intensity_input, wl=wl_input * 1e-9, NA=NA_input)
    device.make_pulse(rise=rise_input, fall=fall_input, pulse_time = 1/frequency, start_time = 1/(4*frequency), pulse_width = 1/(2*frequency))
    device.pulse_train(max_cycles=cycles_input) # inserted from cycles function
    device.simulate()
    fig_voltage, fig_dndt, fig_zoom, _, _, _ = device.plot(semilog=semilog_input, charge_only=charge_input)

    if carrier:
        st.pyplot(fig_dndt)
    if voltage:
        st.pyplot(fig_voltage)
    if lifetime:
        st.pyplot(fig_zoom)

##----------------------------------------------##############
## Changing the Simuation
st.markdown("""---""")
st.header('Changing the Simulation')

# define a new function
# you generally always want the first two lines (to find the input light pulse value)
def new_dn_dt(t, n, i, k1, k2, k3, pulse, dt):

    tidx = min(int(np.floor(t / dt)), len(pulse)-1)
    g = pulse[tidx]

    return eval(equation)

with st.expander(label="See Current Values",expanded=False):
    st.write('k1 = ', k1_input, ' , k2 = ', k2_input, ', k3 = ', k3_input)
    st.write('Intensity = ', intensity_input, ', Wavelength = ', wl_input, ', Numerical Aperture', NA_input)
    st.write('Frequency = ', freq_input)
    st.write('Max Cycles = ', cycles_input)


# frequency = freq_input
device = IMSKPMPoint()
device.kinetics(k1 = k1_input, k2=k2_input,k3=k3_input)
device.exc_source(intensity=intensity_input, wl=wl_input * 1e-9, NA=NA_input)
device.make_pulse(rise=rise_input, fall=fall_input, pulse_time = 1/frequency, start_time = 1/(4*frequency), pulse_width = 1/(2*frequency))
device.pulse_train(max_cycles=cycles_input)

i = 1e3 # 10^15/cm^3 into /um^3
gen = imskpm.calc_utils.gen_t(device.absorbance, device.pulse, device.thickness)
gen = gen * 1e-12 # to convert to /um^3 for computational accuracy


options = st.multiselect('What variables will you be using?', ['a', 'b', 'c', 'd', 'f', 'h',
                                                               'i', 'j', 'k', 'l', 'm', 'o',
                                                               'p', 'q', 'r', 's', 'u', 'v',
                                                               'w', 'x', 'y', 'z'],['i'])
for option in options:
    val = st.number_input(option + " = ", value = 0e0, format='%e')
    exec(option + "=" + str(val))

with st.form("Changing the Simulation"):
    device.args = (i, device.k1, device.k2, device.k3, gen, device.dt)
    equation = st.text_input('Input an Equation', 'g - k1*(n+i) + k2 * n**2 + k3 * n**3')
    submitted = st.form_submit_button("Generate Graph")
    if submitted:
        try:
            device.func = new_dn_dt
            with st.spinner('Loading graphs...'):
                device.simulate()
                fig_voltage, fig_dndt, fig_zoom, _, _, _ = device.plot()
                st.pyplot(fig_dndt)
                st.pyplot(fig_voltage)
                st.pyplot(fig_zoom)
        except:
            st.warning('Equation not valid. Please check for errors.')


##----------------------------------------------
## Sweep

st.markdown("""---""")
st.header('Sweep')

with st.expander(label="See Current Values",expanded=False):
    st.write('k1 = ', k1_input, ', k2 ', k2_input, ', k3 = ', k3_input)

with st.form("Sweep"):
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        lh_input = st.number_input('Lift Height', min_value=0e0, max_value=1e4, value=1e0, format='%e')
    with col2:
        intensity_input = st.number_input('Intensity (0.1 = 100mW/cm^2 = 1 Sun)', min_value=0e0, max_value=1e8, value=1e1, format='%e')

    st.markdown("""---""")
    st.write("Select a range of frequencies:")
    c1, c2, c3= st.columns(3, gap="medium")
    with c1:
        start_freq_input = st.number_input('Start', min_value=0e0, max_value=1e12, value=1e3, format='%e')
    with c2:
        stop_freq_input = st.number_input('Stop', min_value=0e0, max_value=1e12, value=1e6, format='%e')
    with c3:
        num_freq_input = st.number_input('Number of frequencies', min_value=2, max_value=50, value=10)
    valid_range = start_freq_input < stop_freq_input
    frequencies_input = np.logspace(np.log10(start_freq_input), np.log10(stop_freq_input), num_freq_input)

    st.markdown("""---""")
    submitted = st.form_submit_button("Simulate Graph")
    if submitted:
        if valid_range:
            with st.spinner('Loading graphs...'):
                devicesweep = IMSKPMSweep(k1=k1_input, k2=k2_input, k3=k3_input)
                devicesweep.lift_height = lh_input * 1e-9
                devicesweep.intensity=intensity_input
                devicesweep.frequency_list = frequencies_input
                devicesweep.simulate_sweep(verbose=False) #verbose displays outputs to the command window for feedback

                fig_voltage, fig_dndt, ax_voltage, _ = devicesweep.plot_sweep()

                popt = devicesweep.fit(cfit=False)
                ax_voltage.plot(devicesweep.frequency_list, expf_1tau(devicesweep.frequency_list, *popt), 'r--')

                st.write('$Y_0: $', popt[0], '$A: $', popt[1], '$\\tau: $', popt[2])
                st.pyplot(fig_voltage)
                st.pyplot(fig_dndt)
        else:
            st.warning("Make sure the start frequency is less than the stop frequency.")
