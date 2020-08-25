''' Script for processing and visualising Antarctic surface mass balance data from model projections.

Author: Ella Gilbert, 2020.

Dependencies:
- python 3
- cartopy 0.18.0
- iris 2.2

'''


host = 'bsl'

import sys
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/AntSMB/'
    sys.path.append('/gws/nopw/j04/bas_climate/users/ellgil82/scripts/Tools/')
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/AntSMB/'
    sys.path.append('/users/ellgil82/scripts/Tools/')

import iris
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import threddsclient
import iris.plot as iplt
import iris.quickplot as qplt
import iris.analysis.cartography
import iris.coord_categorisation
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature
from divg_temp_colourmap import shiftedColorMap
import matplotlib

# Create domain constraints.
Antarctic_peninsula = iris.Constraint(longitude=lambda v: -80 <= v <= -55,
                                latitude=lambda v: -75 <= v <= -55)
East_Antarctica = iris.Constraint(longitude=lambda v: -40 <= v <= 180,# & longitude=lambda v: 320 <= v <=360,
                                latitude=lambda v: -90 <= v <= -60)
West_Antarctica = iris.Constraint(longitude=lambda v: -179 <= v <= -40,
                                  latitude=lambda v: -90 <= v <= -72)
# Choose region of interest
region = 'AIS'

# Load orography and coastlines
orog_full = iris.load_cube(filepath + 'orog.nc')
lsm_full = iris.load_cube(filepath + 'sftlf.nc')

# Load relevant SMB files
if region == 'AIS':
    pr = iris.load_cube(filepath + 'pr_ccam_eraint_ant-44i_50km_day.historical.nc')
    evap = iris.load_cube(filepath + 'evspsbl_ccam_eraint_ant-44i_50km_day.historical.nc')
    mrro = iris.load_cube(filepath + 'mrro_ccam_eraint_ant-44i_50km_day.historical.nc')
    mrros = iris.load_cube(filepath + 'mrros_ccam_eraint_ant-44i_50km_day.historical.nc')
    snm = iris.load_cube(filepath + 'snm_ccam_eraint_ant-44i_50km_day.historical.nc')
    #Load orography and coastlines
    orog = iris.load_cube(filepath + 'orog.nc')
    lsm = iris.load_cube(filepath + 'sftlf.nc')
    lsm = lsm / 100
    grid_area = iris.load_cube(filepath + 'grid_area.nc')
else:
    pr = iris.load_cube(filepath + 'pr_ccam_eraint_ant-44i_50km_day.historical.nc', region)
    evap = iris.load_cube(filepath + 'evspsbl_ccam_eraint_ant-44i_50km_day.historical.nc', region)
    mrro = iris.load_cube(filepath + 'mrro_ccam_eraint_ant-44i_50km_day.historical.nc', region)
    mrros = iris.load_cube(filepath + 'mrros_ccam_eraint_ant-44i_50km_day.historical.nc', region)
    snm = iris.load_cube(filepath + 'snm_ccam_eraint_ant-44i_50km_day.historical.nc', region)
    #Load orography and coastlines
    orog = iris.load_cube(filepath + 'orog.nc', region)
    lsm = iris.load_cube(filepath + 'sftlf.nc', region)
    lsm = lsm / 100
    grid_area = iris.load_cube(filepath + 'grid_area.nc', region)

# Convert units
for v in [pr, evap, mrro, mrros, snm]:
    v.convert_units('kg/m2/d')
    iris.coord_categorisation.add_year(v, 'time', name='year')

pr_annual_tot = pr.aggregated_by(['year'],iris.analysis.SUM)
evap_annual_tot = evap.aggregated_by(['year'], iris.analysis.SUM)
mrro_annual_tot = mrro.aggregated_by(['year'], iris.analysis.SUM)
mrros_annual_tot = mrros.aggregated_by(['year'], iris.analysis.SUM)
snm_annual_tot = snm.aggregated_by(['year'], iris.analysis.SUM)

grid_area_masked = grid_area * lsm

pr_an_tot = (pr_annual_tot.collapsed('year', iris.analysis.MEAN) * grid_area_masked).data.sum() / 1e12
evap_an_tot = (evap_annual_tot.collapsed('year', iris.analysis.MEAN) * grid_area_masked).data.sum() / 1e12
mrro_an_tot = (mrro_annual_tot.collapsed('year', iris.analysis.MEAN) * grid_area_masked).data.sum() / 1e12
mrros_an_tot = (mrros_annual_tot.collapsed('year', iris.analysis.MEAN) * grid_area_masked).data.sum() / 1e12
snm_an_tot = (snm_annual_tot.collapsed('year', iris.analysis.MEAN) * grid_area_masked).data.sum() / 1e12

## Calculate SMB
# SMB = precip - evap - sublim - runoff
SMB = pr - evap - mrros # no sublimation file
SMB_annual_tot = pr_annual_tot - evap_annual_tot# - mrro_annual_tot
SMB_an_tot = (SMB_annual_tot.collapsed('year', iris.analysis.MEAN) * grid_area_masked).data.sum() / 1e12

print('area ' + str(grid_area_masked.data.sum()/1e12) + ' x 10^6 km2') # in 10^6 km2
print('pr ' + str(pr_an_tot) + ' Gt yr-1')
print('evap ' + str(evap_an_tot) + ' Gt yr-1')
print('snm ' + str(snm_an_tot) + ' Gt yr-1')
print('mrro ' + str(mrro_an_tot) + ' Gt yr-1')
print('SMB  ' + str(SMB_an_tot) + ' Gt yr-1')

# put bounds on a simple point coordinate.
lsm.coord('latitude').guess_bounds()
lsm.coord('longitude').guess_bounds()

# turn the iris Cube data structure into numpy arrays
gridlons = lsm.coord('longitude').contiguous_bounds()
gridlats = lsm.coord('latitude').contiguous_bounds()

def plot_SMB(SMBvar):
    fig = plt.figure(figsize=[10, 6])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.04, right=0.85, wspace=0.02)
    # Limit the map to -60 degrees latitude and below.
    ax.set_extent([-180, 180, -90, -59.5], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular. !!! NOT COMPATIBLE WITH GRIDLINE LABELS !!!
    #theta = np.linspace(0, 2 * np.pi, 100)
    #center, radius = [0.5, 0.5], 0.5
    #verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    #circle = mpath.Path(verts * radius + center)
    #ax.set_boundary(circle, transform=ax.transAxes)
    ax.spines['geo'].set_visible(False)
    mean_data = SMBvar.collapsed('time', iris.analysis.MEAN).data
        #mean_data[lsm.data==0] = 0
    squished_cmap = shiftedColorMap(cmap=matplotlib.cm.coolwarm, min_val=-100., max_val=1000., name='squished_cmap', var=mean_data,
                                    start=0.3, stop=0.7)
    f = ax.pcolormesh(gridlons, gridlats, np.ma.masked_where(lsm.data == 0, mean_data), transform = ccrs.PlateCarree(),
                    cmap=squished_cmap, vmin = -100, vmax = 1000)
    ax.contour(lsm_full.coord('longitude').points, lsm_full.coord('latitude').points, lsm_full.data>0, levels = [0], linewidths = 2, colors = 'k', transform = ccrs.PlateCarree())
    CbAx = fig.add_axes([0.85, 0.15, 0.02, 0.6])
    cb = plt.colorbar(f, orientation='vertical',  cax = CbAx) #shrink = 0.8,
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=20, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.set_title('Annual mean\nSMB (kg m£^{-2}$ yr$^{-1}$)', fontname='Helvetica', color='dimgrey', fontsize=20, pad = 20)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,  y_inline = True, linewidth=2, color='gray', alpha=0.5, linestyle=':')
    gl.xlabel_style = {'color': 'dimgrey', 'size': 20, }
    gl.ylabel_style =  {'color': 'dimgrey', 'size': 20, }
    #ax.outline_patch.set_linewidth(2)
    if region == 'AIS':
        plt.savefig(filepath + 'ERA-Interim_historical_mean_SMB_AIS.png')
    elif region == Antarctic_peninsula:
        plt.savefig(filepath + 'ERA-Interim_historical_mean_SMB_AP.png')
    elif region == West_Antarctica:
        plt.savefig(filepath + 'ERA-Interim_historical_mean_SMB_WA.png')
    elif region == East_Antarctica:
        plt.savefig(filepath + 'ERA-Interim_historical_mean_SMB_EA.png')
    plt.show()

plot_SMB(SMB_annual_tot)

## Validation? - Do I need to do this or will Chris's script do it?
# After Mottram et al. (2020): SMB values compared with observations in three steps:
# 1. modelled SMB interpolated onto observation location
# 2. interpolated SMB values from the same grid cell averaged (same for obs)
# 3. produces 923 comparison pairs (923 grid cells with averaged obs and model values)
#


# gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=2, color='gray', alpha=0.5, linestyle='--')
# gl.xlabels_top = False
# gl.ylabels_left = False
# gl.xlines = False
# gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# gl.xlabel_style = {'size': 15, 'color': 'gray'}
# gl.xlabel_style = {'color': 'red', 'weight': 'bold'}