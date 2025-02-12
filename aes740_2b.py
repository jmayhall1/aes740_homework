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

data = np.array(netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw2/cm1out_original.nc').variables.get('rain'))[:, 0, :].T
plt.imshow(data, vmin=0, vmax=0.75, aspect='auto', cmap='terrain_r')
plt.gca().invert_yaxis()
plt.ylabel('Distance (* 2000 km)')
plt.xlabel('Time')
plt.title('Rain Accumulation in Centimeters over Time and Distance')
plt.colorbar(label='Rain Accumulation (cm)')
plt.savefig('original_cm1.png')
plt.close('all')

data = np.array(netCDF4.Dataset('C:/Users/jmayhall/Downloads/aes740_hw2/cm1out_double.nc').variables.get('rain'))[:, 0, :].T
plt.imshow(data, vmin=0, vmax=0.75, aspect='auto', cmap='terrain_r')
plt.gca().invert_yaxis()
plt.ylabel('Distance (* 1000 km)')
plt.xlabel('Time')
plt.title('Rain Accumulation in Centimeters over Time and Distance')
plt.colorbar(label='Rain Accumulation (cm)')
plt.savefig('cm1_double.png')
