# coding=utf-8
"""
@author: John Mark Mayhall
Code for homework 3 in AES 740
"""
import glob
import os

import matplotlib.pyplot as plt
import metpy.calc
import metpy.calc as mpcalc
import netCDF4
import numpy as np
import pandas as pd
from metpy.units import units

"""Grab the sounding and surface data."""
data = pd.read_csv('input_sounding', skiprows=[0], header=None, sep='\s+')
surface = pd.read_csv('input_sounding', skiprows=np.arange(1, len(data) + 1, 1),
                      header=None, sep='\s+')
"""Set the column names."""
data.columns = ['Height', 'Potential Temp', 'qv', 'u', 'v']
surface.columns = ['Pressure', 'Potential Temp', 'qv']

"""Grab the mixing ratio data and create the dewpoint data."""
qvs = np.array(data.qv)

"""Add units."""
p_surface = surface.Pressure.values[0] * units.hectopascals
t_surface = mpcalc.temperature_from_potential_temperature(p_surface,
                                                          surface['Potential Temp'].values[0] *
                                                          units.kelvin).to(units.celsius)
td_surface = mpcalc.dewpoint_from_specific_humidity(p_surface, surface.qv.values[0] * units('g/kg'))

"""Grab CM1 mixing ratio data."""
model_data = netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw3/cm1out.nc').variables
qc = np.array(model_data.get('qc'))[:, :, 0, :]
qr = np.array(model_data.get('qr'))[:, :, 0, :]
ql = qc + qr  #qc + qr = mixing ratio needed.

"""Create the needed 400 pressure levels and create the parcel profile."""
model_z = np.array(model_data.get('zh'))
model_p = mpcalc.height_to_pressure_std(model_z * units.kilometers)
prof = mpcalc.parcel_profile(model_p, t_surface, td_surface).to('degC')

"""Create the mixing ratio of the parcel, calculate the adiabatic mixing ratio, and grab the z and x data."""
parcel_mixing_ratio = mpcalc.saturation_mixing_ratio(model_p, (prof.magnitude + 273.15) * units.kelvin)
qla = np.subtract(surface.qv.values[0] / 1000, parcel_mixing_ratio.magnitude)
z = np.array(netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw3/cm1out.nc').variables.get('zh'))
x = np.array(netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw3/cm1out.nc').variables.get('xh'))

for i in range(ql.shape[0]):
    current_data = np.multiply(np.divide(ql[i, :, :], qla[:, np.newaxis]), 100)
    plt.imshow(current_data, vmax=45, aspect='auto', cmap='rainbow',
               extent=(np.min(x), np.max(x), np.max(z), np.min(z)))
    plt.gca().invert_yaxis()
    plt.ylabel('Height (km)')
    plt.xlabel('Distance (km)')
    plt.title(f'Adiabatic Fraction (%) at {2 * i} Minutes')
    plt.colorbar(label='%')
    plt.savefig(f'C:/Users/jmayhall/Downloads/aes740_hw3/af/af_time{i}.png')
    plt.close('all')
