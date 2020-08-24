host = 'jasmin'

import sys
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/AntSMB/'
    sys.path.append('/gws/nopw/j04/bas_climate/users/ellgil82/scripts/Tools/')
elif host == 'bas':
    filepath = '/data/mac/ellgil82/AntSMB/'
    sys.path.append('/users/ellgil82/scripts/Tools/')

import iris
import numpy as np
import matplotlib.pyplot as plt
import PyMC3 as pm
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
region = West_Antarctica

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
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    ax.set_extent([-180, 180, -90, -60], ccrs.PlateCarree())
    mean_data = SMBvar.collapsed('time', iris.analysis.MEAN).data
    #mean_data[lsm.data==0] = 0
    squished_cmap = shiftedColorMap(cmap=matplotlib.cm.coolwarm, min_val=-100., max_val=1000., name='squished_cmap', var=mean_data,
                                    start=0.3, stop=0.7)
    f = ax.pcolormesh(gridlons, gridlats, np.ma.masked_where(lsm.data == 0, mean_data), transform = ccrs.PlateCarree(),
                    cmap=squished_cmap, vmin = -100, vmax = 1000)
    ax.contour(lsm_full.coord('longitude').points, lsm_full.coord('latitude').points, lsm_full.data>0, levels = [0], lw = 2, color = 'k', transform = ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,  linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.ylabels_left = False
    gl.xlines = False
    #gl.xlocator = mticker.FixedLocator([-180, -45, 0, 45, 180])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 15, 'color': 'gray'}
    gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
    cb = plt.colorbar(f, orientation='vertical')
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=24, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
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