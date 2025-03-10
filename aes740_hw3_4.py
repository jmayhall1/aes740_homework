# coding=utf-8
"""
@author: John Mark Mayhall
Code for homework 2 in AES 740
"""
import os
import glob
import netCDF4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, Hodograph, SkewT
from metpy.units import units

data = pd.read_csv('input_sounding', skiprows=[0], header=None, sep='\t')
surface = pd.read_csv('input_sounding', skiprows=np.arange(1, len(data) + 1, 1),
                      header=None, sep='\s+')
data.columns = ['Height', 'Potential Temp', 'qv', 'u', 'v']
surface.columns = ['Pressure', 'Potential Temp', 'qv']

temp = np.array(data.Height)
z = []
for height in temp:
    z.append(height * units.meters)
p = []
for height in z:
    p.append(mpcalc.height_to_pressure_std(height).magnitude)

T = []
thetas = np.array(data['Potential Temp'])
for i, theta in enumerate(thetas):
    T.append(mpcalc.temperature_from_potential_temperature(p[i] * units.hectopascals,
                                                           theta * units.kelvin).to(units.celsius).magnitude)

qvs = np.array(data.qv)
Td = []
for i, qv in enumerate(qvs):
    Td.append(mpcalc.dewpoint_from_specific_humidity(p[i] * units.hectopascal, qv * units('g/kg')).magnitude)

z = np.array(data.Height) * units.meters
T = np.array(T) * units.celsius
Td = np.array(Td) * units.celsius
p = np.array(p) * units.hectopascal

z = np.insert(z, 0, mpcalc.pressure_to_height_std(surface.Pressure.values[0] * units.hectopascal))
T = np.insert(T, 0, mpcalc.temperature_from_potential_temperature(surface.Pressure.values[0] * units.hectopascal,
                                                                  surface['Potential Temp'].values[0] *
                                                                  units.kelvin).to(units.celsius))
Td = np.insert(Td, 0, mpcalc.dewpoint_from_specific_humidity(surface.Pressure.values[0] * units.hectopascal,
                                                             surface.qv.values[0] * units('g/kg')))
p = np.insert(p, 0, surface.Pressure.values[0] * units.hectopascal)

p_surface = surface.Pressure.values[0] * units.hectopascals
t_surface = mpcalc.temperature_from_potential_temperature(p_surface,
                                                          surface['Potential Temp'].values[0] *
                                                          units.kelvin).to(units.celsius)
td_surface = mpcalc.dewpoint_from_specific_humidity(p_surface, surface.qv.values[0] * units('g/kg'))
lcl_pressure, lcl_temperature = mpcalc.lcl(p_surface, t_surface, td_surface)

model_data = netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw3/cm1out_shear.nc').variables
model_qv = np.array(model_data.get('qv'))[:, :, 0, :]
qc = np.array(model_data.get('qc'))[:, :, 0, :]
qr = np.array(model_data.get('qr'))[:, :, 0, :]
ql = qc + qr

model_z = np.array(model_data.get('zh'))
model_p = mpcalc.height_to_pressure_std(model_z * units.kilometers)
prof = mpcalc.parcel_profile(model_p, t_surface, td_surface).to('degC')

"""Source: https://stackoverflow.com/questions/76277163/plotting-the-parcel-virtual-temp-profile-in-metpy-1-5"""
parcel_mixing_ratio = mpcalc.saturation_mixing_ratio(model_p, (prof.magnitude + 273.15) * units.kelvin)
qla = np.subtract(surface.qv.values[0] / 1000, parcel_mixing_ratio.magnitude)


for i in range(ql.shape[0]):
    current_data = np.multiply(np.divide(ql[i, :, :], qla[:, np.newaxis]), 100)
    plt.imshow(current_data, vmin=0, vmax=15, aspect='auto', cmap='rainbow')
    plt.gca().invert_yaxis()
    plt.ylabel('Height (m)')
    plt.xlabel('Distance (m)')
    plt.title(f'Adiabatic Fraction (%) at {2 * i} Minutes')
    plt.colorbar(label='%')
    plt.savefig(f'C:/Users/jmayhall/Downloads/aes740_hw3/af_shear/af_time{i}.png')
    plt.close('all')

model_data2 = netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw3/cm1out.nc').variables
model_qv2 = np.array(model_data2.get('qv'))[:, :, 0, :]
qc2 = np.array(model_data2.get('qc'))[:, :, 0, :]
qr2 = np.array(model_data2.get('qr'))[:, :, 0, :]
ql2 = qc2 + qr2

"""Source: https://stackoverflow.com/questions/76277163/plotting-the-parcel-virtual-temp-profile-in-metpy-1-5"""
parcel_mixing_ratio2 = mpcalc.saturation_mixing_ratio(model_p, (prof.magnitude + 273.15) * units.kelvin)
qla2 = np.subtract(surface.qv.values[0] / 1000, parcel_mixing_ratio2.magnitude)
qla2[qla2 < 0] = 1

for i in range(ql.shape[0]):
    current_data = np.multiply(np.divide(ql[i, :, :], qla[:, np.newaxis]), 100)
    current_data2 = np.multiply(np.divide(ql2[i, :, :], qla2[:, np.newaxis]), 100)
    current_data -= current_data2
    plt.imshow(current_data, vmin=np.nanmin(current_data), vmax=-np.nanmin(current_data), aspect='auto', cmap='rainbow')
    plt.gca().invert_yaxis()
    plt.ylabel('Height (m)')
    plt.xlabel('Distance (m)')
    plt.title(f'Difference between AF and AF with Stronger Shear at {2 * i} Minutes')
    plt.colorbar(label='%')
    plt.savefig(f'C:/Users/jmayhall/Downloads/aes740_hw3/afvsaf_shear/af_time{i}.png')
    plt.close('all')
