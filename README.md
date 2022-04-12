(note: frustratingly, Github STILL does not support LaTeX inline. The unedited version of this page is copied from the [Jupyter Notebook](https://github.com/rajgiriUW/imskpm/blob/main/imskpm/notebooks/IM-SKPM.ipynb): 

## Intensity-Modulated Scanning Kelvin Probe Microscopy (IM-SKPM)

A simple package for simulating IM-SKPM in photovoltaics based on conventional charge density recombination ODE. 

This approach simulates equations of the form:

```
dn/dt = G-k_1*n - k_2*n^2 - k_3*n^3
```
where:
* ```n``` = carrier density (/cm^3)
* ```dn/dt``` = change in carrier density (/cm^3s)
* ```G``` = generation rate (/cm^3)
* ```k_1``` = monomoecular recombination rate (/s), trapping/nonradiative recombination
* ```k_2``` = bimolecular recombination rate (cm^3/s), band-band/radiative recombination
* ```k_3``` = third order recombination rate (cm^6/s), Auger recombination

See, for example:
deQuilettes,et al. "Charge-Carrier Recombination in Halide Perovskites." Chemical Reviews **119**, 11007-11019 (2019).[DOI:10.1021/acs.chemrev.9b00169.](https://doi.org/10.1021/acs.chemrev.9b00169)

###### Package location:
* [IMSKPM](https://github.com/rajgiriUW/imskpm/)

###### Installation instructions:

* Clone or download the code from the link above (it is not on PyPi or other package sites...yet)
* In a command window, navigate to the folder where this is installed, then type:
```python setup.py```
or 
```python setup.py develop```

Then use 

```import imskpm``` 

or follow the commands in the cells below.

###### For more information:
```
Rajiv Giridharagopal, Ph.D.
University of Washington
rgiri@uw.edu
```

