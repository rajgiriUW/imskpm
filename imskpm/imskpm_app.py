import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
import mpld3


##----------------------------------------------
## Heading and introduction

st.set_page_config(page_title='IM-SKPM')
st.title('Intensity-Modulated Scanning Kelvin Probe Microscopy (IM-SKPM)')
st.subheader('Rajiv Giridharagopal, Ph.D.')
st.subheader('University of Washington, rgiri@uw.edu')

# with st.expander('Quick Guide'):
#     st.write('''
#     Insert additional information''')

st.markdown("""---""")

st.sidebar.header('Adjust Graph Settings')


##----------------------------------------------
## Simulating a single IM-SKPM curve

# toggles
st.sidebar.markdown("""---""")

col1, col2 = st.sidebar.columns(2, gap="large")
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

st.header('Simulating a single IM-SKPM curve')

# Determine recombination rates
st.sidebar.markdown("""---""")
st.sidebar.write('$k_1 (units: /s)$')
k1_input = st.sidebar.number_input('The monomoecular recombination rate, trapping/nonradiative recombination', min_value=1e3, max_value=1e10, value=1e6, format='%e')
st.sidebar.markdown("""---""")
st.sidebar.write('$k_2 (units: cm^3/s)$')
k2_input = st.sidebar.number_input('The bimolecular recombination rate, band-band/radiative recombination', min_value=1e-13, max_value=1e-7, value=1e-10, format='%e')
st.sidebar.markdown("""---""")
st.sidebar.write('$k_3 (units: cm^6/s)$')
k3_input = st.sidebar.number_input('The third order recombination rate, Auger recombination')

# Determine frequency
st.sidebar.markdown("""---""")
freq_input = st.sidebar.number_input('Frequency', min_value=0e1, max_value=5e5, value=1e4)
# frequency = 10000 # 10 kHz
frequency = freq_input

# Determine max cycles
st.sidebar.markdown("""---""")
cycles_input = st.sidebar.number_input('Max cycles', min_value=1, max_value=50, value=1)
st.sidebar.markdown("""---""")

# # Description TODO: CREATE TEXT BOX
with st.expander(label="See Current Values",expanded=False):
    st.write('k1 = ', k1_input, ' , k2 = ', k2_input, ', k3 = ', k3_input)
    st.write('Frequency = ', freq_input)
    st.write('Max Cycles = ', cycles_input)
    st.write('Semilog on: ', semilog_input, ', Charge only: ', charge_input)

# Plot the result
with st.spinner('Loading graphs...'):
    device = IMSKPMPoint(k1=k1_input, k2=k2_input, k3=k3_input)
    device.make_pulse(0,0,pulse_time = 1/frequency, start_time = 1/(4*frequency), pulse_width = 1/(2*frequency))
    device.pulse_train(max_cycles=cycles_input) # inserted from cycles function
    device.simulate()
    fig_voltage, fig_dndt, _, _ = device.plot(semilog=semilog_input, charge_only=charge_input)
    st.pyplot(fig_voltage)
    st.pyplot(fig_dndt)

    # interactive graph
#     fig_html = mpld3.fig_to_html(fig_voltage)
#     components.html(fig_html, height=450)

#     fig_html = mpld3.fig_to_html(fig_dndt)
#     components.html(fig_html, height=450)


##----------------------------------------------
## Changing the intensity, wavelenth, and numerical aperture

st.markdown("""---""")
st.header('Changing the Excitation:')
st.subheader('Intensity, Wavelength, and Numerical Aperture')

with st.expander(label="See Current Values",expanded=False):
    st.write('k1 = ', k1_input, ' , k2 = ', k2_input, ', k3 = ', k3_input)
    st.write('Frequency = ', freq_input)
    st.write('Max Cycles = ', cycles_input)

with st.form("Excitation"):
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        intensity_input = st.number_input('Intensity (0.1= 100mW/cm^2= 1 Sun)', min_value=0e0, max_value=1e8, value=1e1, format='%e')
    with col2:
        wl_input = st.number_input('Wavelength', min_value=0e0, max_value=1e3, value=455e-9, format='%e')
    with col3:
        NA_input = st.number_input('Numerical Aperture', min_value=0.1, max_value=3.5, value=0.6)

    st.markdown("""---""")
    submitted = st.form_submit_button("Simulate Changes")
    if submitted:
        with st.spinner('Loading graphs...'):
            # device.exc_source(intensity=10, wl=455e-9, NA=0.6)
            device.exc_source(intensity=intensity_input, wl=wl_input, NA=NA_input)
            device.make_pulse() # if we change the excitation, we shoud update the pulse
            device.simulate()
            fig_voltage, fig_dndt, _, _ = device.plot() # default: semilog False, charge_only True
            st.pyplot(fig_voltage)
            st.pyplot(fig_dndt)

            # interactive graph
#             fig_html = mpld3.fig_to_html(fig_voltage)
#             components.html(fig_html, height=450)

#             fig_html = mpld3.fig_to_html(fig_dndt)
#             components.html(fig_html, height=450)

# # Description TODO: CREATE TEXT BOX
# st.write('Intensity: ', intensity_input)
# st.write('Wavelength: ', wl_input)
# st.write('Numerical aperture: ', NA_input)


##----------------------------------------------
## Sweep

st.markdown("""---""")
st.header('Sweep')

with st.expander(label="See Current Values",expanded=False):
    st.write('k1 = ', k1_input, ', k2 ', k2_input, ', k3 = ', k3_input)

with st.form("Sweep"):
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        lh_input = st.number_input('Lift Height', min_value=0e0, max_value=10e-6, value=1e-9, format='%e')
    with col2:
        intensity_input = st.number_input('Intensity (0.1 = 100mW/cm^2 = 1 Sun)', min_value=0e0, max_value=1e8, value=1e1, format='%e')
    st.markdown("""---""")

    st.write('Choose frequencies to plot')
    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        i1 = st.number_input('1', min_value=0e0, max_value=5e6, value=1e3, format='%e')
        i4 = st.number_input('4', min_value=0e0, max_value=5e6, value=5e4, format='%e')
        i7 = st.number_input('7', min_value=0e0, max_value=5e6, value=1e6, format='%e')
    with col2:
        i2 = st.number_input('2', min_value=0e0, max_value=5e6, value=5e3, format='%e')
        i5 = st.number_input('5', min_value=0e0, max_value=5e6, value=1e5, format='%e')
        i8 = st.number_input('8', min_value=0e0, max_value=5e6, value=5e5, format='%e')
    with col3:
        i3 = st.number_input('3', min_value=0e0, max_value=5e6, value=1e4, format='%e')
        i6 = st.number_input('6', min_value=0e0, max_value=5e6, value=5e5, format='%e')
        i9 = st.number_input('9', min_value=0e0, max_value=5e6, value=1e6, format='%e')

    st.markdown("""---""")
    submitted = st.form_submit_button("Simulate Graph")
    if submitted:
        with st.spinner('Loading graphs...'):
            # devicesweep = IMSKPMSweep(k1=1e6, k2=1e-10, k3=0)
            devicesweep = IMSKPMSweep(k1=k1_input, k2=k2_input, k3=k3_input)

            # devicesweep.lift_height = 1e-9
            devicesweep.lift_height = lh_input

            # devicesweep.intensity=10
            devicesweep.intensity=intensity_input

            # Let's update the default list (which is from 100 Hz to 80 MHz) to save time
            devicesweep.frequency_list = np.array([i1, i4, i7, i2, i5, i8, i3, i6, i9])
            devicesweep.frequency_list.sort()

            devicesweep.simulate_sweep(verbose=False) #verbose displays outputs to the command window for feedback

            fig_voltage, ax_voltage = plt.subplots()
            fig_dndt, ax_dndt = plt.subplots()

            fig_voltage, fig_dndt, ax_voltage, ax_dndt = devicesweep.plot()

            popt = devicesweep.fit(cfit=False)
            ax_voltage.plot(devicesweep.frequency_list, expf_1tau(devicesweep.frequency_list, *popt), 'r--')
            #             ax_dndt.plot(devicesweep.frequency_list, expf_1tau(devicesweep.frequency_list, *popt), 'b--')

            st.write('$Y_0: $', popt[0], '$A: $', popt[1], '$\\tau: $', popt[2])
            st.pyplot(fig_voltage)
            st.pyplot(fig_dndt)

            # interactive graph
#             fig_html = mpld3.fig_to_html(fig_voltage)
#             components.html(fig_html, height=450)

#             fig_html = mpld3.fig_to_html(fig_dndt)
#             components.html(fig_html, height=450)

