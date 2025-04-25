# coding=utf-8
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
import numpy as np
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units
from metpy.calc.indices import WK82

ds = WK82()
p_lst, t_lst, td_lst = ds.variables.get('pressure'), ds.variables.get('temperature'), ds.variables.get('dewpoint')
p_lst, t_lst, td_lst = np.array(p_lst) * units('hPa'), (np.array(t_lst) * units('K')).to('degC'), np.array(td_lst) * units('degC')

fig = plt.figure(figsize=(18, 12))
skew = SkewT(fig, rotation=45, rect=(0.05, 0.05, 0.50, 0.90))

# add the Metpy logo
add_metpy_logo(fig, 105, 85)

# Change to adjust data limits and give it the semblance of what we want
skew.ax.set_adjustable('datalim')
skew.ax.set_ylim(1020, 100)
skew.ax.set_xlim(-20, 30)

# Set some better labels than the default to increase readability
skew.ax.set_xlabel(str.upper(f'Temperature ({t_lst.units:~P})'), weight='bold')
skew.ax.set_ylabel(str.upper(f'Pressure ({p_lst.units:~P})'), weight='bold')

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
skew.plot(p_lst.magnitude, t_lst.magnitude, 'r', lw=4, label='TEMPERATURE')
skew.plot(p_lst.magnitude, td_lst.magnitude, 'g', lw=4, label='DEWPOINT')

# Again, we can use some simple Python math functionality to 'resample'
# the wind barbs for a cleaner output with increased readability.
# Something like this would work.
interval = np.logspace(2, 3, 40) * units.hPa
idx = mpcalc.resample_nn_1d(p_lst, interval)

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
p_surface = p_lst[0]
t_surface = t_lst[0]
td_surface = td_lst[0]
lcl_pressure, lcl_temperature = mpcalc.lcl(p_surface, t_surface, td_surface)
skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black',
          label=f'LCL: {round(lcl_pressure.magnitude)}mb, {round(lcl_temperature.magnitude)}C')

lfc_pressure, lfc_temperature = mpcalc.lfc(p_lst, t_lst, td_lst)
skew.plot(lfc_pressure, lfc_temperature, 'ko', markerfacecolor='yellow',
          label=f'LFC: {round(lfc_pressure.magnitude)}mb, {round(lfc_temperature.magnitude)}C')

el_pressure, el_temperature = mpcalc.el(p_lst, t_lst, td_lst)
skew.plot(el_pressure, el_temperature, 'ko', markerfacecolor='blue',
          label=f'EL: {round(el_pressure.magnitude)}mb, {round(el_temperature.magnitude)}C')

# Calculate full parcel profile and add to plot as black line
prof = mpcalc.parcel_profile(p_lst, t_surface, td_surface).to('degC')
skew.plot(p_lst, prof, 'k', linewidth=2, label='SB PARCEL PATH')

# Shade areas of CAPE and CIN
skew.shade_cin(p_lst, t_lst, prof, td_lst, alpha=0.2, label='SBCIN')
skew.shade_cape(p_lst, t_lst, prof, alpha=0.2, label='SBCAPE')

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
kindex = mpcalc.k_index(p_lst, t_lst, td_lst)
total_totals = mpcalc.total_totals_index(p_lst, t_lst, td_lst)

# mixed layer parcel properties!
ml_t, ml_td = mpcalc.mixed_layer(p_lst, t_lst, td_lst, depth=50 * units.hPa)
ml_p, _, _ = mpcalc.mixed_parcel(p_lst, t_lst, td_lst, depth=50 * units.hPa)
mlcape, mlcin = mpcalc.mixed_layer_cape_cin(p_lst, t_lst, prof, depth=50 * units.hPa)

# most unstable parcel properties!
mu_p, mu_t, mu_td, _ = mpcalc.most_unstable_parcel(p_lst, t_lst, td_lst, depth=50 * units.hPa)
mucape, mucin = mpcalc.most_unstable_cape_cin(p_lst, t_lst, td_lst, depth=50 * units.hPa)

# Estimate the height of LCL in meters from hydrostatic thickness (for sig_tor)
new_p = np.append(p_lst[p_lst > lcl_pressure], lcl_pressure)
new_t = np.append(t_lst[p_lst > lcl_pressure], lcl_temperature)
lcl_height = mpcalc.thickness_hydrostatic(new_p, new_t)

# Compute Surface-based CAPE
sbcape, sbcin = mpcalc.surface_based_cape_cin(p_lst, t_lst, td_lst)

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

skewleg = skew.ax.legend(loc='upper left')

# add a quick plot title, this could be automated by
# declaring a station and datetime variable when using
# realtime observation data from Siphon.
plt.figtext(0.45, 0.97, 'WK Analytic Sounding',
            weight='bold', fontsize=10, ha='center')

# Show the plot
plt.savefig('wk_sounding.png')
plt.show()