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

data = np.array(netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw3/cm1out.nc').variables.get('dbz'))[:, :, 0, :]
for i in range(data.shape[0]):
    current_data = data[i, :, :]
    plt.imshow(current_data, vmin=-50, vmax=50, aspect='auto', cmap='gist_ncar')
    plt.gca().invert_yaxis()
    plt.ylabel('Height (m)')
    plt.xlabel('Distance (m)')
    plt.title(f'Simulated Reflectivity (dBZ) at {2 * i} Minutes')
    plt.colorbar(label='dBZ')
    plt.savefig(f'C:/Users/jmayhall/Downloads/aes740_hw3/reflectivity/dbz_time{i}.png')
    plt.close('all')

data = np.array(netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw3/cm1out.nc').variables.get('tke'))[:, :, 0, :]
for i in range(data.shape[0]):
    current_data = data[i, :, :]
    plt.imshow(current_data, vmin=0, vmax=0.5, aspect='auto', cmap='terrain_r')
    plt.gca().invert_yaxis()
    plt.ylabel('Height (m)')
    plt.xlabel('Distance (m)')
    plt.title(f'TKE at {2 * i} Minutes')
    plt.colorbar(label=r'TKE ($\frac{m^2}{s^2}$)')
    plt.savefig(f'C:/Users/jmayhall/Downloads/aes740_hw3/tke/tke_time{i}.png')
    plt.close('all')

data = np.array(netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw3/cm1out.nc').variables.get('w'))[:, :, 0, :]
for i in range(data.shape[0]):
    current_data = data[i, :, :]
    plt.imshow(current_data, vmin=-1, vmax=1, aspect='auto', cmap='rainbow')
    plt.gca().invert_yaxis()
    plt.ylabel('Height (m)')
    plt.xlabel('Distance (m)')
    plt.title(f'Vertical Velocity (w) at {2 * i} Minutes')
    plt.colorbar(label=r'w ($\frac{m}{s}$)')
    plt.savefig(f'C:/Users/jmayhall/Downloads/aes740_hw3/w/w_time{i}.png')
    plt.close('all')
