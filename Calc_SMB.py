''' Script for processing and visualising Antarctic surface mass balance data from model projections.

Author: Ella Gilbert, 2020.

Dependencies:
- python 3
- cartopy 0.18.0
- iris 2.2

'''

# Define host HPC where script is running
host = 'bsl'

# Define host-specific filepath and location of additional python tool scripts
import sys
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/AntSMB/new_run/'
    sys.path.append('/gws/nopw/j04/bas_climate/users/ellgil82/scripts/Tools/')
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/AntSMB/'
    sys.path.append('/users/ellgil82/scripts/Tools/')

import iris
import numpy as np
import matplotlib.pyplot as plt
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
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import pandas as pd
import matplotlib.colors as colors

# Create domain constraints.
Antarctic_peninsula = iris.Constraint(longitude=lambda v: -80 <= v <= -55,
                                latitude=lambda v: -75 <= v <= -55)
East_Antarctica = iris.Constraint(longitude=lambda v: -40 <= v <= 180,# & longitude=lambda v: 320 <= v <=360,
                                latitude=lambda v: -90 <= v <= -60)
West_Antarctica = iris.Constraint(longitude=lambda v: -179 <= v <= -40,
                                  latitude=lambda v: -90 <= v <= -72)
# Choose region of interest
#region = 'AIS'
string_dict = {'AIS': 'AIS',
               Antarctic_peninsula: 'AP',
               West_Antarctica: 'WA',
               East_Antarctica: 'EA'}

def plot_SMB(SMBvar, comp_dict, region, mean_or_SD, calc):
    range_dict = {'pr': (0,1000),
                  'sbl': (0, 300),
                  'snm': (0, 10),
                  'mrro': (0, 10),
                  'SMB': (-200, 1000),
                  'evap': (0,300)}
    fig = plt.figure(figsize=[10, 6])
    ax = plt.subplot(1, 1, 1, projection=ccrs.SouthPolarStereo())
    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.04, right=0.85, wspace=0.02)
    # Limit the map to -60 degrees latitude and below.
    ax.set_extent([-180, 180, -90, -59.5], ccrs.PlateCarree())
    lsm_full = iris.load_cube(filepath + 'sftlf.nc')
    if region == 'AIS':
        lsm = lsm_full
    else:
        lsm = iris.load_cube(filepath + 'sftlf.nc', region)
    gridlons = lsm.coord('longitude').contiguous_bounds()
    gridlats = lsm.coord('latitude').contiguous_bounds()
    ax.spines['geo'].set_visible(False)
    if calc == 'yes':
        if mean_or_SD == 'mean':
            mean_data = comp_dict[SMBvar].collapsed('time', iris.analysis.MEAN).data
        elif mean_or_SD == 'SD':
            mean_data = comp_dict[SMBvar].collapsed('time', iris.analysis.STD_DEV).data
    else:
        mean_data = comp_dict[SMBvar]
    squished_cmap = shiftedColorMap(cmap=matplotlib.cm.coolwarm_r, min_val=range_dict[SMBvar][0], max_val=range_dict[SMBvar][1], name='squished_cmap', var=mean_data,
                                    start=0.3, stop=0.7)
    f = ax.pcolormesh(gridlons, gridlats, np.ma.masked_where(lsm.data == 0, mean_data), transform = ccrs.PlateCarree(),
                    cmap=squished_cmap, vmin = range_dict[SMBvar][0], vmax = range_dict[SMBvar][1])
    ax.contour(lsm_full.coord('longitude').points, lsm_full.coord('latitude').points, lsm_full.data>0, levels = [0], linewidths = 2, colors = 'k', transform = ccrs.PlateCarree())
    CbAx = fig.add_axes([0.85, 0.15, 0.02, 0.6])
    cb = plt.colorbar(f, orientation='vertical',  cax = CbAx, extend = 'both', ticks = [range_dict[SMBvar][0], 0, range_dict[SMBvar][1] / 2, range_dict[SMBvar][1]]) #shrink = 0.8,
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=20, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.set_title('Annual mean\n'+ SMBvar + ' (kg m$^{-2}$ yr$^{-1}$)', fontname='Helvetica', color='dimgrey', fontsize=20, pad = 20)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,  y_inline = True, linewidth=2, color='gray', alpha=0.5, linestyle=':')
    gl.xlabel_style = {'color': 'dimgrey', 'size': 20, }
    gl.ylabel_style =  {'color': 'dimgrey', 'size': 20, }
    plt.savefig(filepath + 'ERA-Interim_historical_'+ mean_or_SD + SMBvar + string_dict[region] + '.png')
    plt.show()

for v in ['SMB', 'pr', 'evap', 'sbl', 'snm']:
    plot_SMB(v, comp_dict, region = Antarctic_peninsula, mean_or_SD='mean', calc = 'yes')

def plot_SMB_components(region, mean_or_SD, components):
    orog_full = iris.load_cube(filepath + 'orog.nc')
    lsm_full = iris.load_cube(filepath + 'sftlf.nc')
    if region == 'AIS':
        lsm = lsm_full
    else:
        lsm = iris.load_cube(filepath + 'sftlf.nc', region)
    lsm.coord('latitude').guess_bounds()
    lsm.coord('longitude').guess_bounds()
    # turn the iris Cube data structure into numpy arrays
    gridlons = lsm.coord('longitude').contiguous_bounds()
    gridlats = lsm.coord('latitude').contiguous_bounds()
    projection = ccrs.SouthPolarStereo()
    axes_class = (GeoAxes,
                  dict(map_projection=projection))
    fig = plt.figure(figsize=(20, 14))
    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(2, 3),
                    axes_pad=2,
                    #cbar_location='right',
                    #cbar_mode='single',
                    #cbar_pad=0.4,
                    #cbar_size='2.5%',
                    label_mode='')
    titles = ['pr', 'evapsbl', 'sbl', 'mrro', 'snm', 'SMB', 'SD of SMB'] # Will miss SD SMB out if sbl inc.
    maxvals = [3000, 100, 100, 100, 10, 3000, 300]
    minvals = [-100,-100,-100,-100,-10,-100,-10]
    for i, ax in enumerate(axgr):
        ax.set_extent([-180, 180, -90, -59.5], ccrs.PlateCarree())
        squished_cmap = shiftedColorMap(cmap=matplotlib.cm.coolwarm_r, min_val=-200., max_val=2000., name='squished_cmap',
                                        var=components[i],
                                        start=0.3, stop=0.7)
        ax.contour(lsm_full.coord('longitude').points, lsm_full.coord('latitude').points, lsm_full.data>0, levels = [0], linewidths = 2, colors = 'k', transform = ccrs.PlateCarree())
        f = ax.pcolormesh(gridlons, gridlats, np.ma.masked_where(lsm.data == 0, components[i]),
                          norm = colors.SymLogNorm(linthresh = 0.1, linscale = 0.1, vmin=minvals[i], vmax = maxvals[i]),
                          cmap='coolwarm_r', transform=ccrs.PlateCarree())
        ax.spines['geo'].set_visible(False)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, y_inline=True, linewidth=2, color='gray', alpha=0.5,
                          linestyle=':')
        gl.xlabel_style = {'color': 'dimgrey', 'size': 20, }
        gl.ylabel_style = {'color': 'dimgrey', 'size': 20, }
        ax.set_title(titles[i], fontsize = 24, fontweight = 'bold', color = 'dimgrey', pad = 20)
        cb = plt.colorbar(f, extend = 'both', orientation = 'vertical', ax = ax)
        cb.solids.set_edgecolor("face")
        cb.outline.set_edgecolor('dimgrey')
        cb.ax.tick_params(which='both', axis='both', labelsize=20, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                          tick2On=False)
        cb.outline.set_linewidth(2)
    #cb = axgr.cbar_axes[0].colorbar(f, extend='both')#, shrink = 0.8), orientation='vertical',
    #CbAx = fig.add_axes([0.9, 0.15, 0.015, 0.6])
    #cb = fig.colorbar(f, extend = 'both', orientation = 'vertical',  cax = CbAx)
    #cb.solids.set_edgecolor("face")
    #cb.outline.set_edgecolor('dimgrey')
    #cb.ax.tick_params(which='both', axis='both', labelsize=20, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
    #                  tick2On=False)
    #cb.outline.set_linewidth(2)
    #cb.ax.set_title('Annual mean\n (kg m$^{-2}$ yr$^{-1}$)', fontname='Helvetica', color='dimgrey', fontsize=20,
    #                pad=20)
    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.85, wspace=0.18, hspace = 0.18)
    plt.savefig(filepath + 'ERA-Interim_historical_SMB_components_' + mean_or_SD + '_' + string_dict[region] + '.png')
    plt.show()

plot_SMB_components('AIS', 'mean', components)

def process_data(region, mean_or_SD):
    # Load orography and coastlines
    orog_full = iris.load_cube(filepath + 'orog.nc')
    lsm_full = iris.load_cube(filepath + 'sftlf.nc')
    # Load relevant SMB files
    if region == 'AIS':
        pr = iris.load_cube(filepath + 'pr_ccam_eraint_ant-44i_50km_day.1980-1989.nc') # historical
        evap = iris.load_cube(filepath + 'evspsbl_ccam_eraint_ant-44i_50km_day.1980-1989.nc')
        mrro = iris.load_cube(filepath + 'mrro_ccam_eraint_ant-44i_50km_day.1980-1989.nc')
        mrros = iris.load_cube(filepath + 'mrros_ccam_eraint_ant-44i_50km_day.1980-1989.nc')
        snm = iris.load_cube(filepath + 'snm_ccam_eraint_ant-44i_50km_day.1980-1989.nc')
        sbl = iris.load_cube(filepath + 'sbl_ccam_eraint_ant-44i_50km_day.1980-1989.nc')
        #Load orography and coastlines
        orog = iris.load_cube(filepath + 'orog.nc')
        lsm = iris.load_cube(filepath + 'sftlf.nc')
        lsm = lsm / 100
        grid_area = iris.load_cube(filepath + 'grid_area.nc')
    else:
        pr = iris.load_cube(filepath + 'pr_ccam_eraint_ant-44i_50km_day.1980-1989.nc', region) #historical
        evap = iris.load_cube(filepath + 'evspsbl_ccam_eraint_ant-44i_50km_day.1980-1989.nc', region)
        mrro = iris.load_cube(filepath + 'mrro_ccam_eraint_ant-44i_50km_day.1980-1989.nc', region)
        mrros = iris.load_cube(filepath + 'mrros_ccam_eraint_ant-44i_50km_day.1980-1989.nc', region)
        snm = iris.load_cube(filepath + 'snm_ccam_eraint_ant-44i_50km_day.1980-1989.nc', region)
        sbl = iris.load_cube(filepath + 'sbl_ccam_eraint_ant-44i_50km_day.1980-1989.nc', region)
        #Load orography and coastlines
        orog = iris.load_cube(filepath + 'orog.nc', region)
        lsm = iris.load_cube(filepath + 'sftlf.nc', region)
        lsm = lsm / 1001e12
        grid_area = iris.load_cube(filepath + 'grid_area.nc', region)
    # Convert units
    for v in [pr, evap, mrro, mrros, snm, sbl]:
        v.convert_units('kg/m2/d')
        iris.coord_categorisation.add_year(v, 'time', name='year')
    pr_annual_tot = pr.aggregated_by(['year'],iris.analysis.SUM)
    evap_annual_tot = evap.aggregated_by(['year'], iris.analysis.SUM)
    mrro_annual_tot = mrro.aggregated_by(['year'], iris.analysis.SUM)
    snm_annual_tot = snm.aggregated_by(['year'], iris.analysis.SUM)
    sbl_annual_tot = sbl.aggregated_by(['year'], iris.analysis.SUM)
    grid_area_masked = grid_area * lsm
    ## Calculate SMB
    # SMB = precip - evap - sublim - runoff
    SMB_annual_tot = pr_annual_tot - evap_annual_tot - sbl_annual_tot# - mrro_annual_tot
    # put bounds on a simple point coordinate.
    lsm.coord('latitude').guess_bounds()
    lsm.coord('longitude').guess_bounds()
    # turn the iris Cube data structure into numpy arrays
    gridlons = lsm.coord('longitude').contiguous_bounds()
    gridlats = lsm.coord('latitude').contiguous_bounds()
    #plot_SMB(SMB_annual_tot, mean_or_SD= 'mean', region = region)
    #plot_SMB(SMB_annual_tot, mean_or_SD= 'SD', region= region)
    if mean_or_SD == 'mean':
        anly_meth = iris.analysis.MEAN
    elif mean_or_SD == 'SD':
        anly_meth = iris.analysis.STD_DEV
    components = [pr_annual_tot.collapsed('time', anly_meth).data, evap_annual_tot.collapsed('time', anly_meth).data,  sbl_annual_tot.collapsed('time', anly_meth).data,
                  mrro_annual_tot.collapsed('time', anly_meth).data, snm_annual_tot.collapsed('time', anly_meth).data,
                  SMB_annual_tot.collapsed('time', anly_meth).data, SMB_annual_tot.collapsed('time', iris.analysis.STD_DEV).data,]
    annual_series = [pr_annual_tot, evap_annual_tot, sbl_annual_tot, mrro_annual_tot, snm_annual_tot, SMB_annual_tot]
    try:
        plot_SMB_components(region=region, mean_or_SD='mean', components = components)
    except:
        print('no plotting today')
    pr_an_srs = np.sum(pr_annual_tot.data * np.broadcast_to(grid_area_masked.data, pr_annual_tot.data.shape),
                       axis=(1, 2)) / 1e12
    evap_an_srs = np.sum(evap_annual_tot.data * np.broadcast_to(grid_area_masked.data, evap_annual_tot.data.shape),
                         axis=(1, 2)) / 1e12
    sbl_an_srs = np.sum(sbl_annual_tot.data * np.broadcast_to(grid_area_masked.data, sbl_annual_tot.data.shape),
                         axis=(1, 2)) / 1e12
    mrro_an_srs = np.sum(mrro_annual_tot.data * np.broadcast_to(grid_area_masked.data, mrro_annual_tot.data.shape),
                         axis=(1, 2)) / 1e12
    snm_an_srs = np.sum(snm_annual_tot.data * np.broadcast_to(grid_area_masked.data, snm_annual_tot.data.shape),
                        axis=(1, 2)) / 1e12
    SMB_an_srs = np.sum(SMB_annual_tot.data * np.broadcast_to(grid_area_masked.data, SMB_annual_tot.data.shape),
                        axis=(1, 2)) / 1e12
    component_stats = pd.DataFrame(index=['pr', 'evap', 'sbl', 'mrro', 'snm', 'SMB'])
    component_stats['Mean'] = pd.Series(
        [pr_an_srs.mean(), evap_an_srs.mean(), sbl_an_srs.mean(), mrro_an_srs.mean(), snm_an_srs.mean(), SMB_an_srs.mean()],
        index=['pr', 'evap', 'sbl', 'mrro', 'snm', 'SMB'])
    component_stats['SD'] = pd.Series(
        [np.std(pr_an_srs), np.std(evap_an_srs), np.std(sbl_an_srs), np.std(mrro_an_srs), np.std(snm_an_srs), np.std(SMB_an_srs)],
        index=['pr', 'evap', 'sbl', 'mrro', 'snm', 'SMB'])
    component_stats.to_csv(filepath + string_dict[region] + '_summary_stats_SMB_components_Gt_yr.csv')
    print(component_stats)
    print('area ' + str(grid_area_masked.data.sum()/1e12) + ' x 10^6 km2') # in 10^6 km2
    comp_dict = {'pr': pr_annual_tot, 'evap': evap_annual_tot, 'sbl': sbl_annual_tot, 'mrro': mrro_annual_tot, 'snm': snm_annual_tot, 'SMB': SMB_annual_tot}
    return components, annual_series, comp_dict

#components, annual_series, comp_dict = process_data(Antarctic_peninsula, mean_or_SD='mean')

for r in [ Antarctic_peninsula, East_Antarctica, West_Antarctica, 'AIS',]:
    components, annual_series, comp_dict = process_data(r, mean_or_SD='mean')

plt.show()

# Calculate SMB with and without ice shelves - approximate ice shelves as having elevation < 150 m (no land ice mask available)


## NOTES
## Validation? - Do I need to do this or will Chris's script do it?
# After Mottram et al. (2020): SMB values compared with observations in three steps:
# 1. modelled SMB interpolated onto observation location
# 2. interpolated SMB values from the same grid cell averaged (same for obs)
# 3. produces 923 comparison pairs (923 grid cells with averaged obs and model values)

# to limit map extent to circular boundary:

# Compute a circle in axes coordinates, which we can use as a boundary
# for the map. We can pan/zoom as much as we like - the boundary will be
# permanently circular. !!! NOT COMPATIBLE WITH GRIDLINE LABELS !!!
# theta = np.linspace(0, 2 * np.pi, 100)
# center, radius = [0.5, 0.5], 0.5
# verts = np.vstack([np.sin(theta), np.cos(theta)]).T
# circle = mpath.Path(verts * radius + center)
# ax.set_boundary(circle, transform=ax.transAxes)

