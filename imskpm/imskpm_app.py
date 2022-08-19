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
    global carrier_input
    global rise_input
    global fall_input
    global intensity_input
    global wl_input
    global NA_input
    st.sidebar.markdown("""---""")
    st.sidebar.subheader('Change the Excitation')

    carrier_input = st.sidebar.select_slider('Holes/Electrons', options=['positive','negative'], value='positive')

    if carrier_input == 'negative':
        carrier_input = -1
    else:
        carrier_input = 1

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
    sidebar_data['Holes/Electrons'] = carrier_input
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


def download_points(data_points, name):
    '''
    Downloads data points as a csv file

    Parameters
    ----------
    data_points : ndarray
        An ndarray of the data points to download

    '''
    df = pd.DataFrame(data_points)
    csv = convert_df(df)

    st.download_button(
        label="Download points as CSV",
        data=csv,
        file_name=f'{name}.csv',
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

    device = IMSKPMPoint(k1=k1_input, k2=k2_input, k3=k3_input, carrier=carrier_input)
    device.lift_height = lift_input * 1e-9
    device.exc_source(intensity=intensity_input, wl=wl_input * 1e-9, NA=NA_input)
    device.make_pulse(rise=rise_input, fall=fall_input, pulse_time = 1/frequency,
                      start_time = 1/(4*frequency), pulse_width = 1/(2*frequency))
    device.pulse_train(max_cycles=cycles_input) # inserted from cycles function

    #     i = 1e3 # 10^15/cm^3 into /um^3
    gen = imskpm.calc_utils.gen_t(device.absorbance, device.pulse, device.thickness)

    st.write('**Change the Rate Equation**')
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


    # toggles + radio buttons
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        semilog_input = st.select_slider('Semilog', options=['off','on'], value='off')
    with col2:
        charge_input = st.select_slider('Charge only', options=['off', 'on'], value='off')
    with col3:
        fit = st.radio("Fit line for carrier lifetime plot", ('Monoexponential', 'Stretched exponential'))

    # adjust semilog
    if semilog_input=='on':
        semilog_input=True
    else:
        semilog_input=False

    # adjust charge
    if charge_input=='on':
        charge_input=True
    else:
        charge_input=False

    # adjust fit
    single_input = False
    stretched_input = False
    if fit == 'Monoexponential':
        single_input = True
    else:
        stretched_input = True

    # display variable values
    #     with st.expander(label="See Current Values",expanded=False):
    #         st.write(sidebar_data)

    try: # equation error checking; comment out try block when debugging
        with st.spinner('Loading graphs...'):
            device.func = new_dn_dt
            device.simulate()

            if single_input:
                popt = device.fit_single()
            else:
                popt = device.fit_stretched()

            fig_voltage, fig_dndt, fig_lifetime, _, _, ax_lifetime = device.plot(semilog=semilog_input,
                                                                                 charge_only=charge_input,
                                                                                 lifetime=True,
                                                                                 single=single_input,
                                                                                 stretched=stretched_input)

            tab1, tab2, tab3 = st.tabs(['Carriers Generated', 'Carrier Lifetime', 'Voltage'])
            with tab1:
                st.pyplot(fig_dndt)
            with tab2:
                st.pyplot(fig_lifetime)
            with tab3:
                st.pyplot(fig_voltage)

            # Allow point download
            t_arr = (device.sol.t.copy())*1e6
            t_arr = t_arr.reshape(len(t_arr), 1)
            n_dens_arr = device.n_dens.copy()
            n_dens_arr = n_dens_arr.reshape(len(n_dens_arr), 1)
            points = np.concatenate((t_arr, n_dens_arr), axis=1)


            download_points(points, 'carrier_density_v_time')
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

    # Determine lift height, total time, and max cycles
    col1, col2, col3 = st.columns(3, gap='medium')
    with col1:
        lh_input = st.number_input('Lift Height (nm)', min_value=0, max_value=10000, value=20, key=2)
    with col2:
        total_t_input = st.number_input('Total time (s)', value=1.6)
    with col3:
        cycles_input = st.number_input('Max cycles', min_value=1, max_value=50, value=20)

    # Determine intensity and carrier (+/-)
    c1, c2 = st.columns(2, gap='large')
    with c1:
        intensity_input = st.number_input('Intensity (0.1 = 100mW/cm^2 = 1 Sun)',
                                          min_value=0e0, max_value=1e8, value=1e1, format='%e')
    with c2:
        sweep_carrier_input = st.select_slider('Holes/Electrons', options=['positive','negative'], value='positive', key=2)

    if sweep_carrier_input == 'negative':
        sweep_carrier_input = -1
    else:
        sweep_carrier_input = 1

    # Choose start, stop, and num of frequencies
    st.write("*Select a range of frequencies:*")
    c1, c2, c3= st.columns(3, gap="medium")
    with c1:
        start_freq_input = st.number_input('Start (Hz)', min_value=0e0, max_value=1e12, value=1e3, format='%e')
    with c2:
        stop_freq_input = st.number_input('Stop (Hz)', min_value=0e0, max_value=1e12, value=1e6, format='%e')
    with c3:
        num_freq_input = st.number_input('Number of frequencies', min_value=2, max_value=50, value=7)

    #####################
    st.write('**Change the Rate Equation**')
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
        devicesweep = IMSKPMSweep(k1=k1_input, k2=k2_input, k3=k3_input, carrier=sweep_carrier_input)
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
                popt = devicesweep.fit(cfit=False)
                fig_voltage, fig_dndt, ax_voltage, _ = devicesweep.plot_sweep(sweep_input)

                st.write("**Fit Line:**")
                st.write('$Y_0 = $', popt[0], '$A = $', popt[1], '$\\tau = $', popt[2])

                st.pyplot(fig_voltage)
                st.pyplot(fig_dndt)

                with st.expander("See Data Points"):
                    with st.spinner('Loading data points...'):
                        freq_arr = (frequencies_input.copy()).reshape(len(frequencies_input), 1)
                        voltages = np.array(devicesweep.cpd_means)
                        volt_arr = voltages.reshape(len(voltages), 1)
                        points = np.concatenate((freq_arr, volt_arr), axis=1)
                        download_points(points, 'sweep_data')
                        st.write("Frequency (Hz) | Voltage (V)")
                        st.write(points)
    elif not valid_range:
        st.warning("Make sure the start frequency is less than the stop frequency.")


if __name__ == '__main__':
    intro()
    sidebar_data = sidebar_input()
    download_sidebar_data(sidebar_data)
    sim_charge_density()
    sweep()
