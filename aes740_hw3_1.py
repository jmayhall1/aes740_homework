# coding=utf-8
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
from metpy.plots import add_metpy_logo, Hodograph, SkewT
from metpy.units import units

data = pd.read_csv('input_sounding', skiprows=[0], header=None, sep=' ')
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

u = np.array(data.u) * units.knots
v = np.array(data.v) * units.knots
z = np.array(data.Height) * units.meters
T = np.array(T) * units.celsius
Td = np.array(Td) * units.celsius
p = np.array(p) * units.hectopascal

u = np.insert(u, 0, np.nan)
v = np.insert(v, 0, np.nan)
z = np.insert(z, 0, mpcalc.pressure_to_height_std(surface.Pressure.values[0] * units.hectopascal))
T = np.insert(T, 0, mpcalc.temperature_from_potential_temperature(surface.Pressure.values[0] * units.hectopascal,
                                                                  surface['Potential Temp'].values[0] *
                                                                  units.kelvin).to(units.celsius))
Td = np.insert(Td, 0, mpcalc.dewpoint_from_specific_humidity(surface.Pressure.values[0] * units.hectopascal,
                                                             surface.qv.values[0] * units('g/kg')))
p = np.insert(p, 0, surface.Pressure.values[0] * units.hectopascal)
# STEP 1: CREATE THE SKEW-T OBJECT AND MODIFY IT TO CREATE A
# NICE, CLEAN PLOT
# Create a new figure. The dimensions here give a good aspect ratio
fig = plt.figure(figsize=(18, 12))
skew = SkewT(fig, rotation=45, rect=(0.05, 0.05, 0.50, 0.90))

# add the Metpy logo
add_metpy_logo(fig, 105, 85)

# Change to adjust data limits and give it the semblance of what we want
skew.ax.set_adjustable('datalim')
skew.ax.set_ylim(1020, 100)
skew.ax.set_xlim(-20, 30)

# Set some better labels than the default to increase readability
skew.ax.set_xlabel(str.upper(f'Temperature ({T.units:~P})'), weight='bold')
skew.ax.set_ylabel(str.upper(f'Pressure ({p.units:~P})'), weight='bold')

# Set the facecolor of the skew-t object and the figure to white
fig.set_facecolor('#ffffff')
skew.ax.set_facecolor('#ffffff')

# Here we can use some basic math and Python functionality to make a cool
# shaded isotherm pattern.
x1 = np.linspace(-100, 40, 8)
x2 = np.linspace(-90, 50, 8)
y = [1100, 50]
for i in range(0, 8):
    skew.shade_area(y=y, x1=x1[i], x2=x2[i], color='gray', alpha=0.02, zorder=1)

# STEP 2: PLOT DATA ON THE SKEW-T. TAKE A COUPLE EXTRA STEPS TO
# INCREASE READABILITY
# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
# Set the linewidth to 4 for increased readability.
# We will also add the 'label' keyword argument for our legend.
skew.plot(p, T, 'r', lw=4, label='TEMPERATURE')
skew.plot(p, Td, 'g', lw=4, label='DEWPOINT')

# Again, we can use some simple Python math functionality to 'resample'
# the wind barbs for a cleaner output with increased readability.
# Something like this would work.
interval = np.logspace(2, 3, 40) * units.hPa
idx = mpcalc.resample_nn_1d(p, interval)
skew.plot_barbs(pressure=p[idx], u=u[idx], v=v[idx])

# Add the relevant special lines native to the Skew-T Log-P diagram and
# provide basic adjustments to linewidth and alpha to increase readability
# first, we add a matplotlib axvline to highlight the 0-degree isotherm
skew.ax.axvline(0 * units.degC, linestyle='--', color='blue', alpha=0.3)
skew.plot_dry_adiabats(lw=1, alpha=0.3)
skew.plot_moist_adiabats(lw=1, alpha=0.3)
skew.plot_mixing_lines(lw=1, alpha=0.3)

# Calculate LCL height and plot as a black dot. Because `p`'s first value is
# ~1000 mb and its last value is ~250 mb, the `0` index is selected for
# `p`, `T`, and `Td` to lift the parcel from the surface. If `p` was inverted,
# i.e., start from a low value, 250 mb, to a high value, 1000 mb, the `-1` index
# should be selected.
p_surface = surface.Pressure.values[0] * units.hectopascals
t_surface = mpcalc.temperature_from_potential_temperature(p_surface,
                                                          surface['Potential Temp'].values[0] *
                                                          units.kelvin).to(units.celsius)
td_surface = mpcalc.dewpoint_from_specific_humidity(p_surface, surface.qv.values[0] * units('g/kg'))
lcl_pressure, lcl_temperature = mpcalc.lcl(p_surface, t_surface, td_surface)
skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black',
          label=f'LCL: {round(lcl_pressure.magnitude)}mb, {round(lcl_temperature.magnitude)}C')

lfc_pressure, lfc_temperature = mpcalc.lfc(p, T, Td)
skew.plot(lfc_pressure, lfc_temperature, 'ko', markerfacecolor='yellow',
          label=f'LFC: {round(lfc_pressure.magnitude)}mb, {round(lfc_temperature.magnitude)}C')

el_pressure, el_temperature = mpcalc.el(p, T, Td)
skew.plot(el_pressure, el_temperature, 'ko', markerfacecolor='blue',
          label=f'EL: {round(el_pressure.magnitude)}mb, {round(el_temperature.magnitude)}C')

# Calculate full parcel profile and add to plot as black line
prof = mpcalc.parcel_profile(p, t_surface, td_surface).to('degC')
skew.plot(p, prof, 'k', linewidth=2, label='SB PARCEL PATH')

# Shade areas of CAPE and CIN
skew.shade_cin(p, T, prof, Td, alpha=0.2, label='SBCIN')
skew.shade_cape(p, T, prof, alpha=0.2, label='SBCAPE')

# STEP 3: CREATE THE HODOGRAPH INSET. TAKE A FEW EXTRA STEPS TO
# INCREASE THE READABILITY.
# Create a hodograph object: first we need to add an axis,
# then we can create the Metpy Hodograph
hodo_ax = plt.axes((0.48, 0.45, 0.5, 0.5))
h = Hodograph(hodo_ax)

# Add two separate grid increments for a cooler look. This also
# helps to increase readability
h.add_grid(increment=20, ls='-', lw=1.5, alpha=0.5)
h.add_grid(increment=10, ls='--', lw=1, alpha=0.2)

# The next few steps make for a clean hodograph inset, removing the
# tick marks, tick labels, and axis labels
h.ax.set_box_aspect(1)
h.ax.set_yticklabels([])
h.ax.set_xticklabels([])
h.ax.set_xticks([])
h.ax.set_yticks([])
h.ax.set_xlabel(' ')
h.ax.set_ylabel(' ')

# Here we can add a simple Python for loop that adds tick marks
# to the inside of the hodograph plot to increase readability!
plt.xticks(np.arange(0, 0, 1))
plt.yticks(np.arange(0, 0, 1))
for i in range(10, 120, 10):
    h.ax.annotate(str(i), (i, 0), xytext=(0, 2), textcoords='offset pixels',
                  clip_on=True, fontsize=10, weight='bold', alpha=0.3, zorder=0)
for i in range(10, 120, 10):
    h.ax.annotate(str(i), (0, i), xytext=(0, 2), textcoords='offset pixels',
                  clip_on=True, fontsize=10, weight='bold', alpha=0.3, zorder=0)

# plot the hodograph itself, using plot_colormapped, colored
# by height
h.plot_colormapped(u, v, c=z, linewidth=6, label='0-12km WIND')
# compute Bunkers storm motion so we can plot it on the hodograph!
RM, LM, MW = mpcalc.bunkers_storm_motion(p, u, v, z)
h.ax.text((RM[0].m + 0.5), (RM[1].m - 0.5), 'RM', weight='bold', ha='left',
          fontsize=10, alpha=0.6)
h.ax.text((LM[0].m + 0.5), (LM[1].m - 0.5), 'LM', weight='bold', ha='left',
          fontsize=10, alpha=0.6)
h.ax.text((MW[0].m + 0.5), (MW[1].m - 0.5), 'MW', weight='bold', ha='left',
          fontsize=10, alpha=0.6)
h.ax.arrow(0, 0, RM[0].m - 0.3, RM[1].m - 0.3, linewidth=2, color='black',
           alpha=0.2, label='Bunkers RM Vector',
           length_includes_head=True, head_width=2)

# STEP 4: ADD A FEW EXTRA ELEMENTS TO REALLY MAKE A NEAT PLOT
# First we want to actually add values of data to the plot for easy viewing
# To do this, let's first add a simple rectangle using Matplotlib's 'patches'
# functionality to add some simple layout for plotting calculated parameters
#                                  xloc   yloc   xsize  ysize
fig.patches.extend([plt.Rectangle((0.563, 0.05), 0.334, 0.37,
                                  edgecolor='black', facecolor='white',
                                  linewidth=1, alpha=1, transform=fig.transFigure,
                                  figure=fig)])

# Now let's take a moment to calculate some simple severe-weather parameters using
# metpy's calculations
# Here are some classic severe parameters!
kindex = mpcalc.k_index(p, T, Td)
total_totals = mpcalc.total_totals_index(p, T, Td)

# mixed layer parcel properties!
ml_t, ml_td = mpcalc.mixed_layer(p, T, Td, depth=50 * units.hPa)
ml_p, _, _ = mpcalc.mixed_parcel(p, T, Td, depth=50 * units.hPa)
mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p, T, prof, depth=50 * units.hPa)

# most unstable parcel properties!
mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(p, T, Td, depth=50 * units.hPa)
mucape, mucin = mpcalc.most_unstable_cape_cin(p, T, Td, depth=50 * units.hPa)

# Estimate the height of LCL in meters from hydrostatic thickness (for sig_tor)
new_p = np.append(p[p > lcl_pressure], lcl_pressure)
new_t = np.append(T[p > lcl_pressure], lcl_temperature)
lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)

# Compute Surface-based CAPE
sbcape, sbcin = mpcalc.surface_based_cape_cin(p, T, Td)
# Compute SRH
(u_storm, v_storm), *_ = mpcalc.bunkers_storm_motion(p, u, v, z)
*_, total_helicity1 = mpcalc.storm_relative_helicity(z, u, v, depth=1 * units.km,
                                                     storm_u=u_storm, storm_v=v_storm)
*_, total_helicity3 = mpcalc.storm_relative_helicity(z, u, v, depth=3 * units.km,
                                                     storm_u=u_storm, storm_v=v_storm)
*_, total_helicity6 = mpcalc.storm_relative_helicity(z, u, v, depth=6 * units.km,
                                                     storm_u=u_storm, storm_v=v_storm)

# Copmute Bulk Shear components and then magnitude
ubshr1, vbshr1 = mpcalc.bulk_shear(p, u, v, height=z, depth=1 * units.km)
bshear1 = mpcalc.wind_speed(ubshr1, vbshr1)
ubshr3, vbshr3 = mpcalc.bulk_shear(p, u, v, height=z, depth=3 * units.km)
bshear3 = mpcalc.wind_speed(ubshr3, vbshr3)
ubshr6, vbshr6 = mpcalc.bulk_shear(p, u, v, height=z, depth=6 * units.km)
bshear6 = mpcalc.wind_speed(ubshr6, vbshr6)

# Use all computed pieces to calculate the Significant Tornado parameter
sig_tor = mpcalc.significant_tornado(sbcape, lcl_height,
                                     total_helicity3, bshear3).to_base_units()

# Perform the calculation of supercell composite if an effective layer exists
super_comp = mpcalc.supercell_composite(mucape, total_helicity3, bshear3)

# There is a lot we can do with this data operationally, so let's plot some of
# these values right on the plot, in the box we made
# First lets plot some thermodynamic parameters
plt.figtext(0.58, 0.37, 'SBCAPE: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.71, 0.37, f'{sbcape:.0f~P}', weight='bold',
            fontsize=10, color='orangered', ha='right')
plt.figtext(0.58, 0.34, 'SBCIN: ', weight='bold',
            fontsize=10, color='black', ha='left')
plt.figtext(0.71, 0.34, f'{sbcin:.0f~P}', weight='bold',
            fontsize=10, color='lightblue', ha='right')
plt.figtext(0.58, 0.29, 'MLCAPE: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.71, 0.29, f'{mlcape:.0f~P}', weight='bold',
            fontsize=10, color='orangered', ha='right')
plt.figtext(0.58, 0.26, 'MLCIN: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.71, 0.26, f'{mlcin:.0f~P}', weight='bold',
            fontsize=10, color='lightblue', ha='right')
plt.figtext(0.58, 0.21, 'MUCAPE: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.71, 0.21, f'{mucape:.0f~P}', weight='bold',
            fontsize=10, color='orangered', ha='right')
plt.figtext(0.58, 0.18, 'MUCIN: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.71, 0.18, f'{mucin:.0f~P}', weight='bold',
            fontsize=10, color='lightblue', ha='right')
plt.figtext(0.58, 0.13, 'TT-INDEX: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.71, 0.13, f'{total_totals:.0f~P}', weight='bold',
            fontsize=10, color='orangered', ha='right')
plt.figtext(0.58, 0.10, 'K-INDEX: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.71, 0.10, f'{kindex:.0f~P}', weight='bold',
            fontsize=10, color='orangered', ha='right')

# now some kinematic parameters
plt.figtext(0.73, 0.37, '0-1km SRH: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.88, 0.37, f'{total_helicity1:.0f~P}',
            weight='bold', fontsize=10, color='navy', ha='right')
plt.figtext(0.73, 0.34, '0-1km SHEAR: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.88, 0.34, f'{bshear1:.0f~P}', weight='bold',
            fontsize=10, color='blue', ha='right')
plt.figtext(0.73, 0.29, '0-3km SRH: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.88, 0.29, f'{total_helicity3:.0f~P}',
            weight='bold', fontsize=10, color='navy', ha='right')
plt.figtext(0.73, 0.26, '0-3km SHEAR: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.88, 0.26, f'{bshear3:.0f~P}', weight='bold',
            fontsize=10, color='blue', ha='right')
plt.figtext(0.73, 0.21, '0-6km SRH: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.88, 0.21, f'{total_helicity6:.0f~P}',
            weight='bold', fontsize=10, color='navy', ha='right')
plt.figtext(0.73, 0.18, '0-6km SHEAR: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.88, 0.18, f'{bshear6:.0f~P}', weight='bold',
            fontsize=10, color='blue', ha='right')
plt.figtext(0.73, 0.13, 'SIG TORNADO: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.88, 0.13, f'{sig_tor[0]:.0f~P}', weight='bold', fontsize=10,
            color='orangered', ha='right')
plt.figtext(0.73, 0.10, 'SUPERCELL COMP: ', weight='bold', fontsize=10,
            color='black', ha='left')
plt.figtext(0.88, 0.10, f'{super_comp:.0f~P}', weight='bold', fontsize=10,
            color='orangered', ha='right')

# Add legends to the skew and hodo
skewleg = skew.ax.legend(loc='upper left')
hodoleg = h.ax.legend(loc='upper left')

# add a quick plot title, this could be automated by
# declaring a station and datetime variable when using
# realtime observation data from Siphon.
plt.figtext(0.45, 0.97, 'Inputted Sounding',
            weight='bold', fontsize=10, ha='center')

# Show the plot
plt.savefig('sounding.png')
plt.show()