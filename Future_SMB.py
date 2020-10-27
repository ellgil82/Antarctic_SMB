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
from divg_temp_colourmap import shiftedColorMap
import cartopy.crs as ccrs
import cartopy.feature
from divg_temp_colourmap import shiftedColorMap
import matplotlib
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import pandas as pd


# Create domain constraints.
Antarctic_peninsula = iris.Constraint(longitude=lambda v: -80 <= v <= -55,
                                latitude=lambda v: -75 <= v <= -55)
East_Antarctica = iris.Constraint(longitude=lambda v: -40 <= v <= 180,# & longitude=lambda v: 320 <= v <=360,
                                latitude=lambda v: -90 <= v <= -60)
West_Antarctica = iris.Constraint(longitude=lambda v: -179 <= v <= -40,
                                  latitude=lambda v: -90 <= v <= -72)
string_dict = {'AIS': 'AIS',
               Antarctic_peninsula: 'AP',
               West_Antarctica: 'WA',
               East_Antarctica: 'EA'}

# Calculate historical mean
mn_SMB_hist = np.mean(hist_SMB_srs[39:59])
std_SMB_hist = np.std(hist_SMB_srs[39:59])
p5_SMB_hist = np.percentile(hist_SMB_srs[39:59], q=5)
p95_SMB_hist = np.percentile(hist_SMB_srs[39:59], q = 95)

# Calculate 2040-2060 mean
mn_SMB_2050 = np.mean(future_SMB_srs[39:59])
std_SMB_2050 = np.std(future_SMB_srs[39:59])
p5_SMB_2050 = np.percentile(future_SMB_srs[39:59], q=5)
p95_SMB_2050 = np.percentile(future_SMB_srs[39:59], q = 95)

# Calculate 2080-2100 mean
mn_SMB_2100 = np.mean(future_SMB_srs[79:99])
std_SMB_2100 = np.std(future_SMB_srs[79:99])
p5_SMB_2100 = np.percentile(future_SMB_srs[79:99], q=5)
p95_SMB_2100 = np.percentile(future_SMB_srs[79:99], q = 95)


def load_model_data(region_name, model_name, scenarios):
    #  Load data for each scenario.
    var_names = ['pr','mrro', 'sbl',   'snm']#'prsn',
    if region_name == 'AIS':
        reg_str = ''
    else:
        reg_str = region_name
    historical_cube_dict = {}
    for var_name in var_names:
        historical = iris.load_cube(filepath + var_name + '_ccam_'+ model_name + '_ant-44i_50km_day.1960-1969.nc', reg_str) # historical
        iris.coord_categorisation.add_season(historical, 'time', name='clim_season')
        iris.coord_categorisation.add_season_year(historical, 'time', name='season_year')
        iris.coord_categorisation.add_year(historical, 'time', name='year')
        historical.convert_units('kg/m2/d')
        historical_cube_dict[var_name] = historical
    historical_cube_dict['SMB'] = historical_cube_dict['pr'] - historical_cube_dict['mrro'] - historical_cube_dict['sbl']
    #historical_cube_dict['rain'] = historical_cube_dict['pr'] - historical_cube_dict['prsn']
    # update labels of x axis to be seasonal (i.e. construct strings from series_labs[0]+series_labs[1])
    if scenarios == 'yes' or scenarios == 'y' or scenarios == 1:
        rcp45_cube_dict = {}
        rcp85_cube_dict = {}
        for var_name in var_names:
            rcp45 = iris.load_cube(filepath + var_name + '_ccam_' + model_name + '_ant-44i_50km_day.rcp45.nc', region_name)
            rcp85 = iris.load_cube(filepath + var_name + '_ccam_' + model_name + '_ant-44i_50km_day.rcp85.nc', region_name)
            iris.coord_categorisation.add_season(rcp45, 'time', name='clim_season')
            iris.coord_categorisation.add_season_year(rcp45, 'time', name='season_year')
            iris.coord_categorisation.add_year(rcp45, 'time', name='year')
            rcp45.convert_units('kg/m2/d')
            rcp45_cube_dict[var_name] = rcp45
            iris.coord_categorisation.add_season(rcp85, 'time', name='clim_season')
            iris.coord_categorisation.add_season_year(rcp85, 'time', name='season_year')
            iris.coord_categorisation.add_year(rcp85, 'time', name='year')
            rcp85.convert_units('kg/m2/d')
            rcp85_cube_dict[var_name] = rcp85
        rcp45_cube_dict['SMB'] = rcp45_cube_dict['pr'] - rcp45_cube_dict['mrro'] - rcp45_cube_dict['sbl']
        rcp45_cube_dict['rain'] = rcp45_cube_dict['pr'] - rcp45_cube_dict['prsn']
        rcp85_cube_dict['SMB'] = rcp85_cube_dict['pr'] - rcp85_cube_dict['mrro'] - rcp85_cube_dict['sbl']
        rcp85_cube_dict['rain'] = rcp85_cube_dict['pr'] - rcp85_cube_dict['prsn']
    if scenarios=='y' or scenarios=='yes':
        return historical_cube_dict, rcp45_cube_dict, rcp85_cube_dict
    else:
        return historical_cube_dict

def plot_SMB_components(region, mean_or_SD, components, scenario):
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
    titles = ['pr', 'sbl', 'mrro', 'snm', 'SMB', 'SD of SMB'] # Will miss SD SMB out if sbl inc.
    for i, ax in enumerate(axgr):
        ax.set_extent([-180, 180, -90, -59.5], ccrs.PlateCarree())
        squished_cmap = shiftedColorMap(cmap=matplotlib.cm.coolwarm_r, min_val=-100., max_val=1000., name='squished_cmap',
                                        var=components[i],
                                        start=0.3, stop=0.7)
        ax.contour(lsm_full.coord('longitude').points, lsm_full.coord('latitude').points, lsm_full.data>0, levels = [0], linewidths = 2, colors = 'k', transform = ccrs.PlateCarree())
        f = ax.pcolormesh(gridlons, gridlats, np.ma.masked_where(lsm.data == 0, components[i]),
                          transform=ccrs.PlateCarree(),
                          cmap=squished_cmap, vmin=-100, vmax=1000)
        ax.spines['geo'].set_visible(False)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, y_inline=True, linewidth=2, color='gray', alpha=0.5,
                          linestyle=':')
        gl.xlabel_style = {'color': 'dimgrey', 'size': 20, }
        gl.ylabel_style = {'color': 'dimgrey', 'size': 20, }
        ax.set_title(titles[i], fontsize = 24, fontweight = 'bold', color = 'dimgrey', pad = 20)
    #cb = axgr.cbar_axes[0].colorbar(f, extend='both')#, shrink = 0.8), orientation='vertical',
    CbAx = fig.add_axes([0.9, 0.15, 0.015, 0.6])
    cb = fig.colorbar(f, extend = 'both', orientation = 'vertical',  cax = CbAx)
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=20, labelcolor='dimgrey', pad=10, size=0, tick1On=False,
                      tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.set_title('Annual mean\n (kg m$^{-2}$ yr$^{-1}$)', fontname='Helvetica', color='dimgrey', fontsize=20,
                    pad=20)
    fig.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.85, wspace=0.18, hspace = 0.18)
    plt.savefig(filepath + 'SMB_components_' + mean_or_SD + '_' + string_dict[region] + '_' + scenario + '.png')
    #plt.show()

def process_data(region, mean_or_SD, cube_list, scenario):
    ''' Function to process SMB data from model runs.

    Inputs:

        - Region: geographical iris constraint with which to load lsm and orography data.

        - mean_or_SD; analysis method to process spatial data with.

        - cube_list: list of cubes loaded by load_model_data() function, containing SMB components from specific
         models (e.g. ERA-Interi, Nor-ESM1, HadGEM2 etc.0, over specific time periods (e.g. historical, rcp4.5, rcp8.5)

         - scenario: string to describe the scenario for saving summary statistics file and figures.

    Outputs:

        - components: list of 2D cubes containing time-mean or SD data (i.e. maps of SMB components, either means or SD
        over time)

        - annual_series: list of 1D cubes containing spatial mean or SD data (i.e. time series of SMB component, either
        means or SD over space).

        - component_stats: saves .csv file with summary statistics (mean/standard deviation of individual SMB components)
         to file location.

        '''
    # Load orography and coastlines
    orog_full = iris.load_cube(filepath + 'orog.nc')
    lsm_full = iris.load_cube(filepath + 'sftlf.nc')
    if region == 'AIS':
        #Load orography and coastlines
        orog = iris.load_cube(filepath + 'orog.nc')
        lsm = iris.load_cube(filepath + 'sftlf.nc')
        lsm = lsm / 100
        grid_area = iris.load_cube(filepath + 'grid_area.nc')
        grid_area_masked = grid_area * lsm
    else:
        #Load orography and coastlines
        orog = iris.load_cube(filepath + 'orog.nc', region)
        lsm = iris.load_cube(filepath + 'sftlf.nc', region)
        lsm = lsm / 100
        grid_area = iris.load_cube(filepath + 'grid_area.nc', region)
        grid_area_masked = grid_area * lsm
    annual_totals = {}
    # Convert units
    for v in cube_list.keys():
        cube_list[v].convert_units('kg/m2/d')
        yr_tot = cube_list[v].aggregated_by(['year'],iris.analysis.SUM)
        annual_totals[v] = yr_tot
    ## Calculate SMB
    # SMB = precip - evap - sublim - runoff
    SMB_annual_tot = annual_totals['pr'] - annual_totals['sbl'] - annual_totals['mrro']# - mrro_annual_tot
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
    # Create list of spatial means/standard deviations: 1) pr, 2) evap, 3) sbl, 4) mrro, 5) prsn, 6) snm, 7) SMB, 8) SD of SMB
    components = [annual_totals['pr'].collapsed('time', anly_meth).data,
                  annual_totals['sbl'].collapsed('time', anly_meth).data, annual_totals['mrro'].collapsed('time', anly_meth).data,
                  annual_totals['snm'].collapsed('time', anly_meth).data,
                  SMB_annual_tot.collapsed('time', anly_meth).data, SMB_annual_tot.collapsed('time', iris.analysis.STD_DEV).data,]# annual_totals['prsn'].collapsed('time', anly_meth).data,annual_totals['evapsbl'].collapsed('time', anly_meth).data,
    # Create list of spatially averaged mean time series: 1) prsn, 2) evap, 3) sbl, 4) mrro, 5) prsn, 6) snm, 7) SMB, 8) SD of SMB
    annual_series = []
    for h in annual_totals.keys():
        srs =  np.sum(annual_totals[h].data * np.broadcast_to(grid_area_masked.data, annual_totals[h].data.shape), axis=(1, 2)) / 1e12
        annual_series.append(srs)
    #plot_SMB_components(region=region, mean_or_SD='mean', components = components, scenario = scenario)
    SMB_an_srs = np.sum(SMB_annual_tot.data * np.broadcast_to(grid_area_masked.data, SMB_annual_tot.data.shape), axis=(1, 2)) / 1e12
    annual_series.append(SMB_an_srs)
    component_stats = pd.DataFrame(index=['pr', 'sbl', 'mrro', 'snm', 'SMB'])
    component_stats['Mean'] = pd.Series(
        [annual_series[0].mean(), annual_series[1].mean(), annual_series[2].mean(), annual_series[3].mean(),  SMB_an_srs.mean()],
        index=['pr','sbl', 'mrro', 'snm', 'SMB'])
    component_stats['SD'] = pd.Series(
        [np.std(annual_series[0]), np.std(annual_series[1]), np.std(annual_series[2]), np.std(annual_series[3]),  np.std(SMB_an_srs)],
        index=['pr', 'sbl', 'mrro', 'snm', 'SMB'])
    component_stats.to_csv(filepath + string_dict[region] + '_' + scenario + '_summary_stats_SMB_components_Gt_yr.csv')
    print(component_stats)
    print('area ' + str(grid_area_masked.data.sum()/1e12) + ' x 10^6 km2') # in 10^6 km2
    return components, annual_series

historical_gfdl  = load_model_data('AIS', 'gfdl-esm2m', scenarios='no')
historical_access = load_model_data('AIS', 'access1-0', scenarios='no')
historical_hadgem = load_model_data('AIS', 'hadgem2', scenarios= 'no')
hist_comp_gfdl, hist_srs_gfdl = process_data('AIS', mean_or_SD='mean', cube_list=historical_gfdl, scenario = 'gfdl-esm2m_60s')
hist_comp_access, hist_srs_access = process_data('AIS', mean_or_SD='mean', cube_list=historical_access, scenario = 'access_60s')
hist_comp_hadgem, hist_srs_hadgem = process_data('AIS', mean_or_SD='mean', cube_list=historical_hadgem, scenario = 'hadgem_60s')

for r in [Antarctic_peninsula, West_Antarctica, East_Antarctica]:
    historical_era = load_model_data(r, 'eraint', scenarios='no')
    historical_gfdl, rcp45_gfdl, rcp85_gfdl = load_model_data(r, 'gfdl-esm2m', scenarios='yes')
    historical_access, rcp45_access, rcp85_access = load_model_data(r, 'access1-0', scenarios='yes')
    historical_hadgem, rcp45_hadgem, rcp85_hadgem = load_model_data(r, 'hadgem2', scenarios='yes')
    historical_noresm, rcp45_noresm, rcp85_noresm = load_model_data(r, 'noresm1-m', scenarios='yes')
    hist_comp_era, hist_srs_era = process_data(r, mean_or_SD='mean', cube_list=historical_era, scenario = 'ERA-Interim_historical')
    hist_comp_gfdl, hist_srs_gfdl = process_data(r, mean_or_SD='mean', cube_list=historical_gfdl, scenario = 'gfdl-esm2m_historical')
    hist_comp_access, hist_srs_access = process_data(r, mean_or_SD='mean', cube_list=historical_access, scenario='access1-0_historical')
    hist_comp_hadgem, hist_srs_hadgem = process_data(r, mean_or_SD='mean', cube_list=historical_hadgem, scenario='hadgem2_historical')
    hist_comp_noresm, hist_srs_noresm = process_data(r, mean_or_SD='mean', cube_list=historical_noresm, scenario='noresm_historical')
    fig, ax = plt.subplots(1,1)
    #ax = ax.flatten
    an_model_list = [hist_srs_gfdl[-1].data, hist_srs_access[-1].data, hist_srs_noresm[-1].data,
                     hist_srs_hadgem[-1].data]
    labels_list = ['GFDL-esm2m', 'ACCESS1-0', 'Nor-ESM1', 'HadGEM2']
    MMM = np.mean(an_model_list, axis = 0) # Multi-model mean of all models, not inc. ERA "truth"
    max_an = np.max(an_model_list, axis=0)
    min_an = np.min(an_model_list, axis = 0)
    p95 = np.percentile(an_model_list, 95, axis = 0)
    p5 = np.percentile(an_model_list, 5, axis=0)
    plt.plot(gfdl_an.coord('year').points, MMM, lw = 3, color = 'k', label = 'multi-model mean' )
    plt.fill_between(gfdl_an.coord('year').points, p5, p95, color = 'lightgrey', zorder = 1)
    plt.ylabel('Annual mean ' + v + ' (kg/m2/yr)', rotation=90)
    era = plt.plot(range(1980, 2016, 1), hist_srs_era.data, label='ERA-Interim', lw=1, color='royalblue')
    #ax[1].plot(np.arange(1980, 2016, 0.25), era_seas.data, label='historical ERA-Interim', lw=1, color='royalblue')
    for m, l in zip(an_model_list, labels_list):
        plt.plot(np.arange(1960, 2000.25, 1), m, label='historical ' + l , lw=0.5, color='dimgrey', linestyle='--', zorder = 10)
        #ax[1].plot(np.arange(1960, 2000.25, 0.25), m, label='historical ' + l, lw=1.5)
    # Plot the datasets
    #qplt.plot(rcp45_mean, label='RCP 4.5 scenario' + model_name, lw=1.5, color='blue')
    #qplt.plot(rcp85_mean, label='RCP 8.5 scenario' + model_name, lw=1.5, color='red')
    plt.legend()
    plt.title('Historical annual mean SMB' + ' ' + string_dict[r] )
    plt.savefig(filepath + string_dict[r]+ '_SMB_historical_rough.png')
    #plt.show()




    '''
    # Import relevant files
    historical = iris.load_cube('/path/to/file/')
    CCAM_ACCESS1_0_rcp45 = iris.load_cube('/path/to/file/')
    CCAM_ACCESS1_0_rcp85 = iris.load_cube('/path/to/file/')
    CCAM_canESM_rcp45 = iris.load_cube('/path/to/file/')
    CCAM_canESM_rcp85 = iris.load_cube('/path/to/file/')
    CCAM_CESM_CAM5_rcp45 = iris.load_cube('/path/to/file/')
    CCAM_CESM_CAM5_rcp85 = iris.load_cube('/path/to/file/')
    CCAM_CNRM_CM5_rcp45 = iris.load_cube('/path/to/file/')
    CCAM_CNRM_CM5_rcp85 = iris.load_cube('/path/to/file/')
    CCAM_GFDL_ESM2M_rcp45 = iris.load_cube('/path/to/file/')
    CCAM_GFDL_ESM2M_rcp85 = iris.load_cube('/path/to/file/')
    CCAM_HadGEM2_CC_rcp45 = iris.load_cube('/path/to/file/')
    CCAM_HadGEM2_CC_rcp85 = iris.load_cube('/path/to/file/')
    CCAM_MIROC5_rcp45 = iris.load_cube('/path/to/file/')
    CCAM_MIROC5_rcp85 = iris.load_cube('/path/to/file/')
    CCAM_NorESM1_M_rcp45 = iris.load_cube('/path/to/file/')
    CCAM_NorESM1_M_rcp85 = iris.load_cube('/path/to/file/')
    '''
