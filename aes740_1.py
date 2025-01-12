# coding=utf-8
"""
@author: John Mark Mayhall
Code for homework 1 in AES 740
"""
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import netCDF4
import numpy as np
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units

data = netCDF4.Dataset('C:/Users/jmayhall/Downloads/example_model_output.nc').variables
x, z, theta, pi, rv, rc, rr, w = (data.get('x_scl'), data.get('z_scl'), data.get('theta'), data.get('pi'),
                                  data.get('rv'), data.get('rc'), data.get('rr'), data.get('w'))
x, z, theta, pi, rv, rc, rr, w = (np.array(x), np.array(z), np.array(theta), np.array(pi), np.array(rv), np.array(rc),
                                  np.array(rr), np.array(w))
temp_kel = np.multiply(theta, pi)  # Temp in Kelvin.
temp_cel = np.subtract(temp_kel, 273.15)  # Temp in celsius
pressure = np.multiply(np.power(pi, 1005.7 / 287), 1000)

e_s = np.multiply(np.multiply(0.6112, np.exp(np.divide(np.multiply(17.67, temp_cel), np.add(temp_cel, 243.5)))),
                  1000)  # Saturated Vapor pressure of water in Pa using Bolton's formula
e = np.divide(np.multiply(rv, np.multiply(pressure, 100)),
              np.add(0.622, rv))  # Vapor pressure of water in Pa using Bolton's formula
s = np.divide(np.add(np.multiply(1005.7, temp_kel), np.multiply(9.81, z)), 1000)  # Dry static energy (kJ/kg)
h = np.divide(np.add(np.add(np.multiply(1005.7, temp_kel), np.multiply(9.81, z)),
                     np.multiply(2.5 * (10 ** 6), rv)), 1000)  # Moist static energy (kJ/kg)

r_total = np.add(np.add(rv, rc), rr)  # Total water condensate mixing ratio.

theta_e = np.add(theta, np.multiply(np.multiply(np.divide(2.5 * (10 ** 6), 1005.7), pi),
                                    rv))  # Equivalent potential temperature

td = np.divide(np.multiply(243.5, np.log(np.divide(np.divide(e, 100), 6.112))),
               np.subtract(17.67, np.log(np.divide(np.divide(e, 100), 6.112))))  # Dewpoint using inverted bolton

theta_rho = np.multiply(np.divide(temp_kel, pi),
                        np.divide(np.add(np.divide(rv, 0.622), 1),
                                  np.add(np.add(1, rv), rc)))  # Density potential temperature

fig, axes = plt.subplots(nrows=2, ncols=2)
temps = [theta, theta_e, theta_rho, temp_kel]
labels = [r'Potential Temperature ($\theta$)', r'Equivalent Potential Temperature ($\theta_e$)',
          r'Density Potential Temperature ($\theta_{\rho}$)', r'Temperature']
for i, ax in enumerate(axes.flat):
    im = ax.pcolormesh(x, z, temps[i].T, shading='nearest', cmap='rainbow', vmin=200, vmax=425)
    ax.set_title(labels[i])
    if i == 0 or i == 2:
        ax.set_ylabel('Height (m)')
    if i == 2 or i == 3:
        ax.set_xlabel('Distance from Origin (m)')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Degrees Kelvin')
plt.show()
plt.close('all')

fig, axes = plt.subplots(nrows=2, ncols=2)
temps = [theta, theta_e, theta_rho, temp_kel]
labels = [r'Potential Temperature ($\theta$)', r'Equivalent Potential Temperature ($\theta_e$)',
          r'Density Potential Temperature ($\theta_{\rho}$)', r'Temperature']
for i, ax in enumerate(axes.flat):
    im = ax.pcolormesh(x, z, temps[i].T, shading='nearest', cmap='rainbow', vmin=300, vmax=350)
    ax.set_title(labels[i])
    if i == 0 or i == 2:
        ax.set_ylabel('Height (m)')
    if i == 2 or i == 3:
        ax.set_xlabel('Distance from Origin (m)')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Degrees Kelvin')
plt.show()
plt.close('all')

fig, ax = plt.subplots()
temps = [pi]
labels = [r'Exner Function ($\Pi$)']
im = ax.pcolormesh(x, z, pi.T, shading='nearest', cmap='rainbow')
ax.set_title(labels[0])
ax.set_ylabel('Height (m)')
ax.set_xlabel('Distance from Origin (m)')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Unitless')
plt.show()
plt.close('all')

fig, ax = plt.subplots()
labels = [r'Vertical Velocity ($\frac{m}{s}$)']
im = ax.pcolormesh(x, z, w.T, shading='nearest', cmap='viridis', vmin=0, vmax=40)
ax.set_title(labels[0])
ax.set_ylabel('Height (m)')
ax.set_xlabel('Distance from Origin (m)')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
plt.show()
plt.close('all')

fig, axes = plt.subplots(ncols=2)
vaporpressure = [np.divide(e, 100), np.divide(e_s, 100)]
labels = [r'Vapor Pressure (e)', r'Saturated Vapor Pressure ($e_s$)']
for i, ax in enumerate(axes.flat):
    im = ax.pcolormesh(x, z, vaporpressure[i].T, shading='nearest', cmap='viridis', vmin=0, vmax=25)
    ax.set_title(labels[i])
    if i == 0:
        ax.set_ylabel('Height (m)')
    ax.set_xlabel('Distance from Origin (m)')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax, label='hPa/mb')
plt.show()
plt.close('all')

fig, axes = plt.subplots(ncols=2)
rs = [r_total, rv]
labels = [r'Mixing Ratio of Total Condensate', r'Mixing Ratio of Water Vapor ($r_v$)']
for i, ax in enumerate(axes.flat):
    im = ax.pcolormesh(x, z, rs[i].T, shading='nearest', cmap='terrain_r', vmin=0, vmax=0.02)
    ax.set_title(labels[i])
    if i == 0:
        ax.set_ylabel('Height (m)')
    ax.set_xlabel('Distance from Origin (m)')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax, label=r'$\frac{kg}{kg}$')
plt.show()
plt.close('all')

fig, axes = plt.subplots(ncols=2)
energy = [s, h]
labels = [r'Dry Static Energy (s)', r'Moist Static Energy (h)']
for i, ax in enumerate(axes.flat):
    im = ax.pcolormesh(x, z, energy[i].T, shading='nearest', cmap='jet', vmin=300, vmax=380)
    ax.set_title(labels[i])
    if i == 0 or i == 2:
        ax.set_ylabel('Height (m)')
    ax.set_xlabel('Distance from Origin (m)')
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax, label=r'$\frac{kJ}{kg}$')
plt.show()
plt.close('all')

p = pressure[0] * units.hPa
T = temp_cel[0] * units.degC
Td = td[0] * units.degC
ds = mpcalc.parcel_profile_with_lcl_as_dataset(p, T, Td)
fig = plt.figure(figsize=(10, 8))
skew = SkewT(fig, rotation=45)
# Plot the data using the data from the xarray Dataset including the parcel temperature with
# the LCL level included
skew.plot(ds.isobaric, ds.ambient_temperature, 'r')
skew.plot(ds.isobaric, ds.ambient_dew_point, 'g')
# skew.plot(ds.isobaric, ds.parcel_temperature.metpy.convert_units('degC'), 'black')
# Add the relevant special lines
pressure = np.arange(1000, 499, -50) * units('hPa')
mixing_ratio = np.array([0.1, 0.2, 0.4, 0.6, 1, 1.5, 2, 3, 4,
                         6, 8, 10, 13, 16, 20, 25, 30, 36, 42]).reshape(-1, 1) * units('g/kg')
skew.plot_dry_adiabats(t0=np.arange(233, 533, 10) * units.K, alpha=0.25,
                       colors='orangered', linewidths=1)
skew.plot_moist_adiabats(t0=np.arange(233, 400, 5) * units.K, alpha=0.25,
                         colors='tab:green', linewidths=1)
skew.plot_mixing_lines(pressure=pressure, mixing_ratio=mixing_ratio, linestyles='dotted',
                       colors='dodgerblue', linewidths=1)
prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
skew.plot(p, prof, 'k', linewidth=2)
cape, cin = mpcalc.cape_cin(p, T, Td, prof)

print(f'CAPE: {cape:.2f}')
print(f'CIN: {cin:.2f}')
wmax = np.sqrt(2 * cape)
print(f'CIN: {wmax:.2f}')

# Shade areas of CAPE and CIN
skew.shade_cin(p, T, prof, Td)
skew.shade_cape(p, T, prof)
skew.ax.set_ylim(1000, 100)
# Add the MetPy logo!
add_metpy_logo(fig, 350, 200)
# Add some titles
plt.title('Example Sounding from First Point')
plt.xlabel('Temperature (C)')
plt.ylabel('Pressure (hPa/mb)')
plt.show()
