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

os.environ["CDF_LIB"] = 'C:/Users/jmayhall/.conda/envs/pythonProject1/Lib/site-packages/cdflib'
from spacepy import pycdf

nov_path = 'C:/Users/jmayhall/Downloads/aes740_hw2/November Case Data/'
dec_path = 'C:/Users/jmayhall/Downloads/aes740_hw2/December Case Data/'

ccn_data = glob.glob(f'{nov_path}/ccn_data/*.nc') + glob.glob(f'{dec_path}/ccn_data/*.nc')
ceilometer_data = glob.glob(f'{nov_path}/celiometer_data/*.nc') + glob.glob(f'{dec_path}/celiometer_data/*.nc')
cpc_data = glob.glob(f'{nov_path}/cpc_data/*.nc') + glob.glob(f'{dec_path}/cpc_data/*.nc')
kazr_data = glob.glob(f'{nov_path}/kazr_data/*.nc') + glob.glob(f'{dec_path}/kazr_data/*.nc')
met_data = glob.glob(f'{nov_path}/met_data/*.cdf') + glob.glob(f'{dec_path}/met_data/*.cdf')
sonde_data = glob.glob(f'{nov_path}/sonde_data/*.cdf') + glob.glob(f'{dec_path}/sonde_data/*.cdf')

# for file in ccn_data:
#     data = np.array(netCDF4.Dataset(file).variables.get('N_CCN_dN')).T
#     bounds = np.array(netCDF4.Dataset(file).variables.get('droplet_size_bounds'))
#     print(bounds)
#     plt.gca().invert_yaxis()
#     plt.imshow(data,vmin=0, vmax=1000, aspect='auto')
#     plt.yticks(np.arange(0, 21, 5))
#     plt.xlabel('Time')
#     plt.ylabel('Bin')
#     plt.title(f'CCN Count by Bin Size ({file[-18:-3]})')
#     plt.colorbar(label='CCN Count', ticks=[0, 250, 500, 750, 1000])
#     plt.savefig(f'CCN_{file[-18:-3]}.png', dpi=1000)
#     plt.close('all')
#
# for file in cpc_data:
#     data = np.array(netCDF4.Dataset(file).variables.get('concentration'))
#     plt.plot(data)
#     plt.yticks(np.arange(0, 8001, 500))
#     plt.xlabel('Time')
#     plt.ylabel(r'CPC Count ($\frac{1}{cm^3}$)')
#     plt.title(f'CPC Count by per cubic centimeter ({file[-18:-3]})')
#     plt.savefig(f'CPC_{file[-18:-3]}.png', dpi=1000)
#     plt.close('all')
#
# for file in ceilometer_data:
#     data = np.array(netCDF4.Dataset(file).variables.get('first_cbh'))
#     data[data < 0] = np.nan
#     plt.plot(data)
#     plt.yticks(np.arange(0, 10000, 1000))
#     plt.xticks(np.arange(0, 6001, 1000))
#     plt.xlabel('Time')
#     plt.ylabel(r'CBH (m)')
#     plt.title(f'CBH in meter over time ({file[-18:-3]})')
#     plt.savefig(f'CBH_{file[-18:-3]}.png', dpi=1000)
#     plt.close('all')
#
# for file in kazr_data:
#     data = np.array(netCDF4.Dataset(file).variables.get('reflectivity')).T
#     data[data < -100] = np.nan
#     plt.imshow(data, aspect='auto', vmin=-50, vmax=50)
#     plt.gca().invert_yaxis()
#     plt.yticks(np.arange(0, 500, 100))
#     plt.xticks(np.arange(0, 35001, 5000))
#     plt.xlabel('Time')
#     plt.ylabel(r'Range')
#     plt.title(f'Equivalent Reflectivity over Range and Time ({file[-18:-3]})')
#     plt.colorbar(label='Equivalent Reflectivity (dBZ)')
#     plt.savefig(f'kazr_{file[-18:-3]}.png', dpi=1000)
#     plt.close('all')

for file in met_data:
    data = pycdf.CDF(file)
    data = np.array(netCDF4.Dataset(file).variables.get('reflectivity')).T
    data[data < -100] = np.nan
    plt.imshow(data, aspect='auto', vmin=-50, vmax=50)
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(0, 500, 100))
    plt.xticks(np.arange(0, 35001, 5000))
    plt.xlabel('Time')
    plt.ylabel(r'Range')
    plt.title(f'Equivalent Reflectivity over Range and Time ({file[-18:-3]})')
    plt.colorbar(label='Equivalent Reflectivity (dBZ)')
    plt.savefig(f'kazr_{file[-18:-3]}.png', dpi=1000)
    plt.close('all')

exit()