"""
Created on Mon Aug 11, 2022

@author: Helen Kuang
"""

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

def intro():
    st.title('Intensity-Modulated Scanning Kelvin Probe Microscopy (IM-SKPM)')
    st.subheader('The Ginger Lab, University of Washington')
    st.caption('Rajiv Giridharagopal, rgiri@uw.edu; Helen Kuang, helenkg2@uw.edu')

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
        st.write("")

    st.markdown("___")


def sidebar_input():
    '''
    Takes in user input via the sidebar

    Saves settings

    Returns
    -------
    sidebar_data : dict
        The dictionary of all sidebar settings

    '''
    with st.sidebar.expander('Quick Info'):
        st.write('''
            * All sidebar settings affect the charge density simulation
            * The sweep only uses $k_1, k_2, $ and $k_3$ from these settings''')

    # Determine recombination rates
    global k1_input
    global k2_input
    global k3_input
    st.sidebar.subheader('Change Recombination Rates')
    st.sidebar.write('$k_1 (units: /s)$')
    k1_input = st.sidebar.number_input('The monomoecular recombination rate, trapping/nonradiative recombination',
                                       min_value=1e3, max_value=1e10, value=1e6, format='%e')
    st.sidebar.write('$k_2 (units: cm^3/s)$')
    k2_input = st.sidebar.number_input('The bimolecular recombination rate, band-band/radiative recombination',
                                       min_value=1e-13, max_value=1e-7, value=1e-10, format='%e')
    st.sidebar.write('$k_3 (units: cm^6/s)$')
    k3_input = st.sidebar.number_input('The third order recombination rate, Auger recombination')

    # Determine the excitation
    global rise_input
    global fall_input
    global intensity_input
    global wl_input
    global NA_input
    st.sidebar.markdown("""---""")
    st.sidebar.subheader('Change the Excitation')

    col1, col2 = st.sidebar.columns(2)
    with col1:
        rise_input = st.number_input('Rise Time (s)', min_value=0e0, max_value=1e-3, value=0e0, format='%e')
    with col2:
        fall_input = st.number_input('Fall Time (s)', min_value=0e0, max_value=1e-3, value=0e0, format='%e')

    intensity_input = st.sidebar.number_input('Intensity (0.1 = 100mW/cm^2 = 1 Sun)',
                                              min_value=0e0, max_value=1e8, value=1e-1, format='%e')
    wl_input = st.sidebar.number_input('Wavelength (nm)', min_value=300, max_value=1000, value=455)
    NA_input = st.sidebar.number_input('Numerical Aperture', min_value=0.1, max_value=3.5, value=0.6)

    global lift_input
    lift_input = st.sidebar.number_input('Lift Height (nm)', min_value=0, max_value=10000, value=20)

    # Determine frequency
    global frequency
    st.sidebar.markdown("""---""")
    freq_input = st.sidebar.number_input('Frequency (Hz)', min_value=0e0, max_value=1e12, value=1e4, format='%e')
    frequency = freq_input

    # Determine max cycles
    global cycles_input
    cycles_input = st.sidebar.number_input('Max cycles', min_value=1, max_value=50, value=1)
    st.sidebar.markdown("""---""")

    sidebar_data = {}
    sidebar_data['k1'] = k1_input
    sidebar_data['k2'] = k2_input
    sidebar_data['k3'] = k3_input
    sidebar_data['Rise Time (s)'] = rise_input
    sidebar_data['Fall Time (s)'] = fall_input
    sidebar_data['Intensity'] = intensity_input
    sidebar_data['Wavelength (nm)'] = wl_input
    sidebar_data['Numerical Aperture'] = NA_input
    sidebar_data['Lift Height (nm)'] = lift_input
    sidebar_data['Frequency'] = freq_input
    sidebar_data['Max Cycles'] = cycles_input

    return sidebar_data

@st.cache
def convert_df(df):
    '''
    Convert a DataFrame to a csv

    Parameters
    ----------
    df : DataFrame
        The DataFrame to convert

    Returns
    -------
    csv : csv
        The input as a csv

    '''
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    csv = df.to_csv().encode('utf-8')
    return csv


def download_sidebar_data(sidebar_data):
    '''
    Downloads sidebar settings as a csv file

    Parameters
    ----------
    sidebar_data : dict
        A dictionary of the sidebar settings to download

    '''
    df = pd.DataFrame(sidebar_data, index=[0])
    csv = convert_df(df)

    st.sidebar.download_button(
        label="Download settings as CSV",
        data=csv,
        file_name='sidebar_data.csv',
        mime='text/csv',
    )
    st.sidebar.write('___')


def download_sweep_data(sweep_data):
    '''
    Downloads sweep data points as a csv file

    Parameters
    ----------
    sweep_data : ndarray
        An ndarray of the data points to download

    '''
    df = pd.DataFrame(sweep_data)
    #     st.write(df)
    csv = convert_df(df)

    st.download_button(
        label="Download points as CSV",
        data=csv,
        file_name='sweep_data.csv',
        mime='text/csv',
    )

def download_n_dens(n_dens):
    '''
    Downloads n_dens as a csv file

    Parameters
    ----------
    n_dens : ndarray
        An ndarray of n_dens to download

    '''
    df = pd.DataFrame(n_dens)
    csv = convert_df(df)
    st.download_button(
        label="Download Carrier Densities as CSV",
        data=csv,
        file_name='n_dens.csv',
        mime='text/csv',
    )


# define a new function
# you generally always want the first two lines (to find the input light pulse value)
def new_dn_dt(t, n, k1, k2, k3, pulse, dt, *var_val):

    tidx = min(int(np.floor(t / dt)), len(pulse)-1)
    g = pulse[tidx]

    for var in var_val:
        exec(var[0] + "=" + str(var[1])) # declares new variables

    return eval(equation)


def sim_charge_density():
    '''
    Simulates charge density

    '''

    st.header('Simulate Charge Density')

    with st.expander(label="See Current Values",expanded=False):
        st.write(sidebar_data)

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

    device = IMSKPMPoint(k1=k1_input, k2=k2_input, k3=k3_input)
    device.lift_height = lift_input * 1e-9
    device.exc_source(intensity=intensity_input, wl=wl_input * 1e-9, NA=NA_input)
    device.make_pulse(rise=rise_input, fall=fall_input, pulse_time = 1/frequency,
                      start_time = 1/(4*frequency), pulse_width = 1/(2*frequency))
    device.pulse_train(max_cycles=cycles_input) # inserted from cycles function

    #     i = 1e3 # 10^15/cm^3 into /um^3
    gen = imskpm.calc_utils.gen_t(device.absorbance, device.pulse, device.thickness)

    st.subheader('Change the Rate Equation')
    options = st.multiselect('What variables will you be using?', ['a', 'b', 'c', 'd', 'f', 'h',
                                                                   'i', 'j', 'k', 'l', 'm', 'o',
                                                                   'p', 'q', 'r', 's', 'u', 'v',
                                                                   'w', 'x', 'y', 'z'])
    var_val = {} # dict that maps variable name to value
    for option in options:
        val = st.number_input(option + " = ", value = 0e0, format='%e')
        var_val[option] = val

    tuple_list = list(var_val.items()) # makes a list of tuples: (var, val)

    scale = 1e-4 #1 = /cm^3, 1e-4 = /um^3, 1e2 = /m^2
    args_tuple = (device.k1, device.k2/scale**3, device.k3/scale**6, gen*scale**3, device.dt) + tuple(tuple_list)

    device.args = args_tuple
    global equation
    equation = st.text_input('Input an Equation', 'g - k1 * n - k2 * n**2 - k3 * n**3')

    try:
        with st.spinner('Loading graphs...'):
            device.func = new_dn_dt
            device.simulate()
            fig_voltage, fig_dndt, fig_lifetime, _, _, _ = device.plot(semilog=semilog_input, charge_only=charge_input, lifetime=True)

            tab1, tab2, tab3 = st.tabs(['Carriers Generated', 'Carrier Lifetime', 'Voltage'])
            with tab1:
                st.pyplot(fig_dndt)
            with tab2:
                st.pyplot(fig_lifetime)
            with tab3:
                st.pyplot(fig_voltage)

            # Allow n_dens download
            download_n_dens(device.n_dens)
    except:
        st.warning('Equation not valid. Please check for errors.')


def sweep():
    '''
    Simulates an IMSKPM sweep over many frequencies

    '''
    st.markdown("""---""")
    st.header('Sweep')

    with st.expander(label="See Current Values",expanded=False):
        st.write('k1 = ', k1_input, ', k2 ', k2_input, ', k3 = ', k3_input)
    st.write("")

    # Determine lift height and intensity
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        lh_input = st.number_input('Lift Height (nm)', min_value=0, max_value=10000, value=20, key=2)
    with col2:
        intensity_input = st.number_input('Intensity (0.1 = 100mW/cm^2 = 1 Sun)',
                                          min_value=0e0, max_value=1e8, value=1e1, format='%e')
    # Determine total time and max cycles
    col3, col4 = st.columns(2, gap="medium")
    with col3:
        total_t_input = st.number_input('Total time (s)', value=1.6)
    with col4:
        cycles_input = st.number_input('Max cycles', min_value=1, max_value=50, value=20)

    # Choose start, stop, and num of frequencies
    st.write("")
    st.write("*Select a range of frequencies:*")
    c1, c2, c3= st.columns(3, gap="medium")
    with c1:
        start_freq_input = st.number_input('Start (Hz)', min_value=0e0, max_value=1e12, value=1e3, format='%e')
    with c2:
        stop_freq_input = st.number_input('Stop (Hz)', min_value=0e0, max_value=1e12, value=1e6, format='%e')
    with c3:
        num_freq_input = st.number_input('Number of frequencies', min_value=2, max_value=50, value=7)

    st.write("")

    #####################
    st.subheader('Change the Rate Equation')
    options = st.multiselect('What variables will you be using for the sweep?', ['a', 'b', 'c', 'd', 'f', 'h',
                                                                                 'i', 'j', 'k', 'l', 'm', 'o',
                                                                                 'p', 'q', 'r', 's', 'u', 'v',
                                                                                 'w', 'x', 'y', 'z'])
    var_val = {}
    for option in options:
        val = st.number_input(option + " = ", value = 0e0, format='%e')
        var_val[option] = val

    tuple_list = list(var_val.items())

    global equation
    equation = st.text_input('Input an equation for the sweep', 'g - k1 * n - k2 * n**2 - k3 * n**3')
    #####################

    upload_error = False
    uploaded_file = st.file_uploader("Optional: Upload a CSV UTF-8 (Comma delimited) file of frequencies and voltage (CPD)")

    if uploaded_file is not None:
        try:
            sweep_input = pd.read_csv(uploaded_file, sep=',',skipfooter=1)
            sweep_input = sweep_input.to_numpy()
        except:
            upload_error = True
            st.error("Check the uploaded file for errors.")
    else:
        sweep_input = None

    # button
    start_sweep = st.button("Simulate Sweep")
    st.write("")

    # Determine if range is valid
    valid_range = start_freq_input < stop_freq_input

    if valid_range and not upload_error:
        frequencies_input = np.logspace(np.log10(start_freq_input), np.log10(stop_freq_input), num_freq_input)
        devicesweep = IMSKPMSweep(k1=k1_input, k2=k2_input, k3=k3_input)
        devicesweep.k1 = k1_input
        devicesweep.k2 = k2_input
        devicesweep.k3 = k3_input
        devicesweep.lift_height = lh_input * 1e-9
        devicesweep.intensity=intensity_input
        devicesweep.frequency_list = frequencies_input
        #################
        gen = imskpm.calc_utils.gen_t(devicesweep.absorbance, devicesweep.pulse, devicesweep.thickness)
        scale = 1e-4
        args_tuple = (devicesweep.k1, devicesweep.k2/scale**3, devicesweep.k3/scale**6, gen*scale**3,
                      devicesweep.dt) + tuple(tuple_list)
        devicesweep.args = args_tuple
        devicesweep.func = new_dn_dt
        #################

        if start_sweep:
            valid_equation = True
            try:
                with st.spinner('Loading graphs...'):
                    devicesweep.simulate_sweep(verbose=False, total_time=total_t_input, max_cycles=cycles_input)

            except:
                valid_equation = False
                st.warning("Equation not valid. Please check for errors.")

            if valid_equation:
                fig_voltage, fig_dndt, ax_voltage, _ = devicesweep.plot_sweep(sweep_input)
                popt = devicesweep.fit(cfit=False)
                ax_voltage.plot(devicesweep.frequency_list, expf_1tau(devicesweep.frequency_list, *popt), 'r--')

                st.write("**Fit Line:**")
                st.write('$Y_0 = $', popt[0], '$A = $', popt[1], '$\\tau = $', popt[2])

                st.pyplot(fig_voltage)
                st.pyplot(fig_dndt)

                with st.expander("See Data Points"):
                    with st.spinner('Loading data points...'):
                        freq_arr = frequencies_input.reshape(len(frequencies_input), 1)
                        voltages = np.array(devicesweep.cpd_means)
                        volt_arr = voltages.reshape(len(voltages), 1)
                        points = np.concatenate((freq_arr, volt_arr), axis=1)
                        download_sweep_data(points)
                        st.write(points)
    elif not valid_range:
        st.warning("Make sure the start frequency is less than the stop frequency.")


if __name__ == '__main__':
    intro()
    sidebar_data = sidebar_input()
    download_sidebar_data(sidebar_data)
    sim_charge_density()
    sweep()
