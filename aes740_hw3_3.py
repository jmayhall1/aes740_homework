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

# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.plots import add_metpy_logo, Hodograph, SkewT
from metpy.units import units

data = pd.read_csv('input_sounding', skiprows=[0], header=None, sep=' ')
surface = pd.read_csv('input_sounding', skiprows=np.arange(1, len(data) + 1, 1),
                      header=None, sep='       ')
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

# Calculate full parcel profile and add to plot as black line
prof = mpcalc.parcel_profile(p, t_surface, td_surface).to('degC')

model_qv = np.array(netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw3/cm1out.nc').variables.get('qv'))[:, :, 0, :]
qc = np.array(netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw3/cm1out.nc').variables.get('qc'))[:, :, 0, :]
qr = np.array(netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw3/cm1out.nc').variables.get('qr'))[:, :, 0, :]
ql = qc + qr

"""Source: https://stackoverflow.com/questions/76277163/plotting-the-parcel-virtual-temp-profile-in-metpy-1-5"""
parcel_mixing_ratio = mpcalc.saturation_mixing_ratio(p, (prof.magnitude + 273.15) * units.kelvin)
difference = np.subtract(surface.qv.values[0] / 1000, parcel_mixing_ratio.magnitude)
qla = np.sum(difference)

af = np.multiply(np.divide(ql, qla), 100)

for i in range(af.shape[0]):
    current_data = af[i, :, :]
    plt.imshow(current_data, vmin=0, vmax=3, aspect='auto', cmap='gist_ncar')
    plt.gca().invert_yaxis()
    plt.ylabel('Height (m)')
    plt.xlabel('Distance (m)')
    plt.title(f'Adiabatic Fraction (%) at {2 * i} Minutes')
    plt.colorbar(label='%')
    plt.savefig(f'C:/Users/jmayhall/Downloads/aes740_hw3/af/af_time{i}.png')
    plt.close('all')
