''' Script for processing and visualising Antarctic surface mass balance data from model projections.

Author: Ella Gilbert, 2020.

Dependencies:
- python 3
- cartopy 0.18.0
- iris 2.2

'''

## ========= TO DO LIST ========== ##

## Figure out how to do constrained loading, look at ice shelves in specific sectors DONE (ish)
## Sort MMM figure DONE
## Calculate melt extent - values (above threshold and > 1 mm w.e.) DONE
## Calculate isotherm - needs tas data


# Define host HPC where script is running
host = 'bsl'

# Define host-specific filepath and location of additional python tool scripts
import sys
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/AntSMB/MAR/'
    sys.path.append('/gws/nopw/j04/bas_climate/users/ellgil82/scripts/Tools/')
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/AntSMB/MAR/'
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
import xarray as xr
from divg_temp_colourmap import shiftedColorMap
import matplotlib
from matplotlib import rcParams
import seaborn as sns
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import pandas as pd

# Create domain constraints.
Antarctic_peninsula = iris.Constraint(x=lambda v: -80 <= v <= -55,
                                y=lambda v: -75 <= v <= -55)
East_Antarctica = iris.Constraint(x=lambda v: -40 <= v <= 180,# & longitude=lambda v: 320 <= v <=360,
                                y=lambda v: -90 <= v <= -60)
West_Antarctica = iris.Constraint(x=lambda v: -179 <= v <= -40,
                                  y=lambda v: -90 <= v <= -72)

# Isolate region of interest ('' for whole continent)
region = 'AP'

def load_sims(region):
    ''' Load data for region of interest ('' for whole continent).

    Inputs:

    - region: iris constraint to limit loading.

    Outputs:

    - dict_list: list of dictionaries (one for each GCM) containing the three variables of interest, as simulated by MAR forced by each
    GCM.

    - stats: dictionary of invariant variables (orograph, latitude etc.)

    - masks: dictionary of masks for relevant areas (shelves and grounded ice).'''
    ACCESS_dict = {}
    CNRM_dict = {}
    NOR_dict = {}
    CESM_dict = {}
    dict_list = {'ACCESS1.3': ACCESS_dict,  'CESM2': CESM_dict, 'NorESM1-M': NOR_dict, 'CNRM-CM6-1': CNRM_dict}
    for i, j in enumerate(['ACCESS1.3', 'CESM2', 'NorESM1-M', 'CNRM-CM6-1']):
        try:
            dict_list[j]['RU'] = iris.load_cube(filepath + 'RU_MAR_'+ j +'_rcp8.5_future.nc')
            dict_list[j]['ME'] = iris.load_cube(filepath + 'ME_MAR_' + j + '_rcp8.5_future.nc')
            dict_list[j]['SMB'] = iris.load_cube(filepath + 'SMB_MAR_' + j + '_rcp8.5_future.nc')
            dict_list[j]['TT'] = iris.load_cube(filepath + 'TT_MAR_' + j + '_rcp8.5_future.nc')
        except:
            dict_list[j]['RU'] = iris.load_cube(filepath + 'RU_MAR_'+ j +'_ssp585_future.nc')
            dict_list[j]['ME'] = iris.load_cube(filepath + 'ME_MAR_' + j + '_ssp585_future.nc')
            dict_list[j]['SMB'] = iris.load_cube(filepath + 'SMB_MAR_' + j + '_ssp585_future.nc')
            dict_list[j]['TT'] = iris.load_cube(filepath + 'TT_MAR_' + j + '_ssp585_future.nc')
        for k in ['RU', 'ME', 'SMB']:
            dict_list[j][k] = dict_list[j][k][:,0,:,:]
            iris.coord_categorisation.add_year(dict_list[j][k], 'time', name='year')
            dict_list[j][k] = dict_list[j][k].aggregated_by(['year'], iris.analysis.SUM) # return as annual mean
        dict_list[j]['TT'] = dict_list['TT'][:,0,:,:]
        dict_list[j]['TT'] = dict_list['TT'].aggregated_by(['year'], iris.analysis.MEAN) # return as mean
    # Load invariant data
    grd_ice = iris.load_cube(filepath + 'MARcst-AN35km-176x148.cdf', 'Grounded ice')
    continent = iris.load_cube(filepath + 'MARcst-AN35km-176x148.cdf', 'IF SOL3 EQ 4 THEN 1 ELSE 0')
    ice_mask = iris.load_cube(filepath + 'MARcst-AN35km-176x148.cdf', 'Ice mask (if mw=2)')
    orog = iris.load_cube(filepath + 'MARcst-AN35km-176x148.cdf', 'Surface height')
    lon = iris.load_cube(filepath + 'MARcst-AN35km-176x148.cdf', 'Longitude')
    lat = iris.load_cube(filepath + 'MARcst-AN35km-176x148.cdf', 'Latitude')
    rignot = iris.load_cube(filepath + 'MARcst-AN35km-176x148.cdf', 'Rignot bassins')
    grid_area = iris.load_cube(filepath + 'MARcst-AN35km-176x148.cdf', 'Area')
    rock = iris.load_cube(filepath + 'MARcst-AN35km-176x148.cdf', 'Rock')
    # Create dictionary of variables for later use
    stats = {'grd_ice': grd_ice, 'continent': continent, 'ice_mask': ice_mask, 'orog': orog, 'lat': lat, 'lon': lon,
            'rignot': rignot, 'grid_area': grid_area, 'rock': rock}
    if region == 'AP':
        for m in dict_list.keys():
            for k in ['RU', 'ME', 'SMB', 'TT']:
                dict_list[m][k] = dict_list[m][k][:, 80:120, :50]
        for k in stats.keys():
            stats[k] = stats[k][80:120,:50]
    # Create masks
    ice = np.ma.masked_less(stats['ice_mask'].data, 30)  # mask areas with < 30% ice coverage
    ais = np.ma.masked_equal(stats['continent'].data, 0)  # mask areas of ocean
    ice_msk = ais * ice * stats['grid_area'].data / 100  # mask non-ice, non-land grid points and multiply by grid area to find true area
    grd = np.ma.masked_less(stats['grd_ice'].data, 30)
    grd_msk = ais * grd * stats['grid_area'].data / 100  # mask non-ice, non-grounded ice grid points and multiply by grid area to find true area
    grounded_mask = grd * ais
    grounded_mask[grounded_mask >= 30] = 1
    #shf_msk = np.ma.masked_where((grd_msk.data > 50) & (rock.data > 30), ice_msk)
    shf = np.ma.masked_greater(stats['grd_ice'].data, 50)
    shf_msk = ais * shf
    shf_msk = np.ma.masked_where(stats['rock'].data > 30, shf_msk)
    shelf_mask = ~shf_msk.mask  # Shelf areas are = 1
    masks = {'shelf': shelf_mask, 'grounded': grounded_mask, 'ais': ais}
    return dict_list, masks, stats

#AP_dict_list, AP_masks, AP_stats = load_sims('AP')
#WA_dict_list, WA_masks, WA_stats = load_sims(West_Antarctica)
#EA_dict_list, EA_masks, EA_stats = load_sims(East_Antarctica)
dict_list, masks, stats = load_sims('')

if region == 'AP':
    dict_list = AP_dict_list
    masks = AP_masks
    stats = AP_stats

def melt_dur():
    ACCESS_melt = {}
    CNRM_melt = {}
    NOR_melt = {}
    CESM_melt = {}
    melt_dict = {'ACCESS1.3': ACCESS_melt, 'CESM2': CESM_melt, 'NorESM1-M': NOR_melt, 'CNRM-CM6-1': CNRM_melt}
    for i, j in enumerate(['ACCESS1.3', 'CESM2', 'NorESM1-M', 'CNRM-CM6-1']):
        try:
            melt_dict[j]['ME'] = iris.load_cube(filepath + 'ME_MAR_' + j + '_ssp585_future.nc')
        except:
            melt_dict[j]['ME'] = iris.load_cube(filepath + 'ME_MAR_' + j + '_rcp8.5_future.nc')
        if region == 'AP':
            melt_dict[j]['ME'] = melt_dict[j]['ME'][:,0, 80:120, :50]
        else:
            melt_dict[j]['ME'] = melt_dict[j]['ME'][:,0, :, :]
        iris.coord_categorisation.add_year(melt_dict[j]['ME'], 'time', name='year')
        melt_dict[j]['ME'].data[melt_dict[j]['ME'].data < 1] = 0
        melt_dict[j]['ME'].data[np.broadcast_to(masks['shelf'], melt_dict[j]['ME'].shape) == 0] = np.nan
        #msk_melt = melt_dict[j]['ME'] * (np.broadcast_to(masks['shelf'], melt_dict[j]['ME'].shape))
        #melt_dict[j]['ME_dur'] = msk_melt.copy()
        melt_dict[j]['ME_dur'] = melt_dict[j]['ME'].copy()
        melt_dict[j]['ME_dur'].data[melt_dict[j]['ME_dur'].data < 1] = 0
        melt_dict[j]['ME_dur'].data[melt_dict[j]['ME_dur'].data >= 1] = 1
        melt_dict[j]['ME_dur'] = melt_dict[j]['ME_dur'].aggregated_by(['year'], iris.analysis.SUM)
    return melt_dict

melt_dict = melt_dur()

# Find GCM warming periods ref 1950-79
ACCESS_T = iris.load_cube(filepath + 'year_ACCESS1-3.nc2', 'Near-Surface Air Temperature')
CESM_T = iris.load_cube(filepath + 'year_CESM2.nc2', 'Near-Surface Air Temperature')
NorESM_T = iris.load_cube(filepath + 'year_NorESM1-M.nc2', 'Near-Surface Air Temperature')
CNRM_T = iris.load_cube(filepath + 'year_CNRM-CM6-1.nc2', 'Near-Surface Air Temperature')
MMM_T = (ACCESS_T.data.mean(axis = (1,2)) + CNRM_T.data.mean(axis = (1,2)) + CESM_T.data.mean(axis = (1,2)) + NorESM_T.data.mean(axis = (1,2)))/4
#MMM_T = (ACCESS_T.data + CNRM_T.data + CESM_T.data + NorESM_T.data)/4

GCM_T = {'ACCESS-1.3': ACCESS_T, 'CESM2': CESM_T, 'NorESM1-M': NorESM_T, 'CNRM-CM6-1':CNRM_T}


ACCESS_preind = iris.load_cube(filepath + 'ACCES1-3_185001-190012.nc', 'Near-Surface Air Temperature')
CESM_preind = iris.load_cube(filepath + 'CESM2_185001-190012.nc', 'Near-Surface Air Temperature')
NorESM_preind = iris.load_cube(filepath + 'NorESM1-M_185001-190012.nc', 'Near-Surface Air Temperature')
CNRM_preind = iris.load_cube(filepath + 'CNRM-CM6-1_185001-190012.nc', 'Near-Surface Air Temperature')
for c in [ACCESS_preind, CESM_preind, NorESM_preind, CNRM_preind]:
    iris.coord_categorisation.add_year(c, 'time', name='month')

MMM_preind = (CNRM_preind[:359].aggregated_by(['month'], iris.analysis.MEAN).data.mean() + NorESM_preind[:359].aggregated_by(['month'], iris.analysis.MEAN).data.mean()+ CESM_preind[:359].aggregated_by(['month'], iris.analysis.MEAN).data.mean()+ ACCESS_preind[:359].aggregated_by(['month'], iris.analysis.MEAN).data.mean())/4



preind_refs = {'ACCESS1.3': ACCESS_preind,
              'CESM2': CESM_preind,
              'CNRM-CM6-1': CNRM_preind,
              'NorESM1-M': NorESM_preind,
              'MMM': MMM_preind}

# create spatial maps of pre-industrial temps for each model
preind_list = [np.mean(ACCESS_preind[:359,1:].data, axis = 0), np.mean(CESM_preind[:359].data, axis = 0), np.mean(NorESM_preind[:359].data, axis = 0), np.mean(CNRM_preind[:359].data, axis = 0),]

labs = ['ACCESS-1.3', 'CESM2', 'NorESM1-M', 'CNRM-CM6-1']
for i, j in enumerate([ACCESS_T, CESM_T, NorESM_T, CNRM_T]):
    plt.plot(range(1950,2101), np.mean(j.data - np.broadcast_to(preind_list[i].data, j.shape), axis = (1,2)) , label = labs[i])

plt.legend()
plt.savefig(filepath + 'GCM_global_mean_temp.png')
#plt.show()

# Subtract spatial mean of GCM temp from pre-industrial at each timestep to find anomaly
anom_dict = {}
for i, k in enumerate(labs):
    anom_dict[k] = np.mean(GCM_T[k].data - np.broadcast_to(preind_list[i].data, GCM_T[k].shape), axis = (1,2))
    #anom_dict[k] = np.mean(GCM_T[k].data, axis = (1,2))- preind_list[i].mean()
    plt.plot(range(1950,2101), anom_dict[k], label = k)

plt.legend()
plt.ylabel('Global mean near-surface temperature\nanomaly relative to 1850-1880', labelpad = 30)
plt.xlim(1950,2101)
plt.subplots_adjust(left = 0.2)
plt.savefig(filepath + 'GCM_global_mean_temp_anomaly.png')
#plt.show()

#wget -q ftp://ftp.climato.be/climato/ckittel/MARv3.11/Ella/CNRM-CM6-1/TT*

rolling_mn = pd.Series(anom_dict['NorESM1-M']).rolling(window=30).mean().iloc[30-1:].values
np.where(rolling_mn>1.5)


#Ref: 1850-1879
rolling_mn = pd.Series(np.mean(MMM_T, axis = (1,2))).rolling(window=30).mean().iloc[30-1:].values
np.where(rolling_mn-np.mean(MMM_preind)>1.5)

# dictionary of end years for averages of +1.5, +2 and +4 deg above 1950-79 reference.
# Remember to -29 to each to get the start year. (MAR simulations 1980-2100 only)
# Historical periods last
slice_dict = {'ACCESS1.3': (61, 71, 103, 30),
              'CESM2': (52, 62, 95, 29),
              'CNRM-CM6-1': (55, 68, 98, 29),
              'NorESM1-M': (60, 73, 116, 29),
              'MMM': (57, 69, 103, 29)}

def plot_scenarios(v, dT, difs_or_abs, threshold):
    '''Figure caption: '''
    # plot differences between 1950-79 and +1.5/2/4 degree world
    if dT == '1p5':
        tuple_idx = 0
    elif dT == '2':
        tuple_idx = 1
    elif dT == '4':
        tuple_idx = 2
    elif dT == 'hist':
        tuple_idx = 3
    fig, axs = plt.subplots(2,2, figsize = (12,10))
    axs = axs.flatten()
    CbAx = fig.add_axes([0.85,0.25, 0.03, 0.5])
    for i, j in enumerate(dict_list.keys()):
        axs[i].contourf(masks['ais'], cmap='Greys', vmin=0, vmax=1)  # manually plot ocean as white
        if difs_or_abs == 'difs':
            if v == 'SMB':
                lims = (-500, 500)
                bwr_zero = shiftedColorMap(cmap=matplotlib.cm.RdBu, min_val=lims[0], max_val=lims[1], name='bwr_zero',
                                           var=(np.mean(
                                               dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data,
                                               axis=0)), start=0.15, stop=0.85)
                cm = bwr_zero
            else:
                lims = (-500, 500)
                bwr_zero = shiftedColorMap(cmap=matplotlib.cm.RdBu_r, min_val=lims[0], max_val=lims[1],  name='bwr_zero',
                                           var=(np.mean(
                                               dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data,
                                               axis=0)), start=0.15, stop=0.85)
                cm = bwr_zero
            # Find mean difference between slice where warming = +dT and start of the simulation (i.e. the difference at dT)
            c = axs[i].pcolormesh(np.mean(dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data, axis=0)
                                  -np.mean(dict_list[j][v][slice_dict[j][:29]].data, axis=0), cmap = cm, vmin = -500, vmax = 500) # calculate difference relative to historical period
        elif difs_or_abs == 'abs':
            if v == 'SMB':
                lims = (-1000, 2000)
                bwr_zero = shiftedColorMap(cmap=matplotlib.cm.RdBu, min_val=lims[0], max_val=lims[1], name='bwr_zero',
                                           var=(np.mean(
                                               dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data,
                                               axis=0)), start=0.15, stop=0.85)
                cm = bwr_zero
            else:
                cm = 'Reds'
                lims = (0,1000)
            if threshold == 'yes':
                c = axs[i].pcolormesh((np.mean(dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data, axis=0)>725.), cmap=cm, vmin=lims[0], vmax=lims[1])
            else:
                c = axs[i].pcolormesh( np.mean(dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data, axis=0), cmap=cm, vmin=lims[0], vmax=lims[1])
        axs[i].contour(masks['shelf'], levels=  [0], colors = 'k', linewidths = 0.8)
        axs[i].set_title(j, fontsize=20, color='dimgrey', )
        axs[i].axis('off')
    cb = plt.colorbar(c, cax = CbAx, ticks = [lims[0], 0,  lims[1]], extend = 'both')
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=20, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    if difs_or_abs == 'difs':
        if v == 'ME_dur':
            cb.ax.set_title('d yr$^{-1}$', fontname='Helvetica', color='dimgrey', fontsize=20, pad=20)
        else:
            cb.ax.set_title('kg m$^{-2}$ yr$^{-1}$', fontname='Helvetica', color='dimgrey', fontsize=20, pad=20)
    elif difs_or_abs == 'abs':
        if v == 'ME_dur':
            cb.ax.set_title('d yr$^{-1}$', fontname='Helvetica', color='dimgrey', fontsize=20, pad=20)
        else:
            cb.ax.set_title('kg m$^{-2}$ yr$^{-1}$', fontname='Helvetica', color='dimgrey', fontsize=20,pad=20)
    plt.subplots_adjust(right = 0.8)
    plt.savefig(filepath + 'GCM_' + v + '_' + difs_or_abs + '_' + dT + '_deg_warming.png')
    #plt.savefig(filepath + 'GCM_' + v + '_difs_+' + dT + '_deg_warming.eps')
    #plt.show()

for i in ['ME', 'RU', 'SMB']:
    for j in ['1p5', '2', '4']:
        plot_scenarios(i, j, 'abs', threshold='no')

for j in ['1p5', '2', '4']:
    plot_melt_dur('ME_dur', dT = j, difs_or_abs='dif', threshold = 'yes')

def plot_melt_dur(melt_dict, dT):
    # plot differences between 1950-79 and +1.5/2/4 degree world
    if dT == '1p5':
        tuple_idx = 0
    elif dT == '2':
        tuple_idx = 1
    elif dT == '4':
        tuple_idx = 2
    fig, axs = plt.subplots(2,2, figsize = (12,10))
    axs = axs.flatten()
    CbAx = fig.add_axes([0.85,0.25, 0.03, 0.5])
    for i, j in enumerate(melt_dict.keys()):
        # Find mean difference between slice where warming = +dT and start of the simulation (i.e. the difference at dT)
        c = axs[i].pcolormesh(masks['shelf'] * (np.mean(melt_dict[j]['ME_dur'][slice_dict[j][tuple_idx]-29:slice_dict[j][tuple_idx]].data, axis = 0)-
                                                (np.mean(melt_dict[j]['ME_dur'][:29].data, axis = 0))), cmap = 'Reds', vmin = 0, vmax = 100)
        axs[i].contour(masks['shelf'], levels=  [0], colors = 'k', linewidths = 0.8)
        axs[i].set_title(j, fontsize=20, color='dimgrey', )
        axs[i].axis('off')
    cb = plt.colorbar(c, cax = CbAx, ticks = [0, 100, 365], extend = 'max')
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=20, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    #cb.set_label(v, fontsize=20, rotation = 0, color='dimgrey', labelpad=30)
    cb.ax.set_title('Mean melt duration\n (d yr$^{-1}$)', fontname='Helvetica', color='dimgrey', fontsize=20, pad=20)
    plt.subplots_adjust(right = 0.8)
    plt.savefig(filepath + 'GCM_melt_duration_abs_+'+ dT + '_deg_warming.png')
    #plt.savefig(filepath + 'GCM_' + v + '_difs_+' + dT + '_deg_warming.eps')
    plt.show()

for t in ['1p5', '2', '4']:
    plot_melt_dur(melt_dict, t)

dif_dict = {}
for n in dict_list.keys():
    dif_dict[n] = {}
    for m in dict_list[n].keys():
        dif_dict[n][m] = np.mean(dict_list[n][m][slice_dict[n][2]-29:slice_dict[n][2]].data, axis = 0) - np.mean(dict_list[n][m][slice_dict[n][0]-29:slice_dict[n][0]].data, axis = 0)

MMM_dict = {}
for j in dict_list['CESM2'].keys():
    print(j)
    MMM_dict[j] = (dict_list['ACCESS1.3'][j][1:].data + dict_list['CESM2'][j].data  + dict_list['NorESM1-M'][j].data
                   + dict_list['CNRM-CM6-1'][j].data )/4 # Find multi-model mean, ignoring differing time coordinate definitions

MMM_dict['ME_dur'] = (melt_dict['ACCESS1.3']['ME_dur'][1:].data + melt_dict['CESM2']['ME_dur'].data + melt_dict['NorESM1-M']['ME_dur'].data + melt_dict['CNRM-CM6-1']['ME_dur'].data)/4

dif_dict['MMM'] = {}
for m in MMM_dict.keys():
    dif_dict['MMM'][m]   = np.mean(MMM_dict[m][slice_dict['MMM'][2]-29:slice_dict['MMM'][2]], axis = 0) - np.mean(MMM_dict[m][slice_dict['MMM'][0]-29:slice_dict['MMM'][0]], axis = 0)

def plot_dif_1p5_to_4(dif, dif_str):
    fig, axs = plt.subplots(1,1, figsize = (12,8))
    # Find mean difference between slice where warming = +dT and start of the simulation (i.e. the difference at dT)
    CbAx = fig.add_axes([0.85, 0.25, 0.03, 0.5])
    if dif_str[-3:] == 'SMB':
        c = axs.pcolormesh(dif, cmap = 'RdBu', vmin = -500, vmax = 500) # masks['shelf'] * dif
    else:
        c = axs.pcolormesh(dif, cmap='RdBu_r', vmin=-500, vmax=500)
    axs.contour(masks['shelf'], levels=  [0], colors = 'k', linewidths = 0.8)
    axs.axis('off')
    cb = plt.colorbar(c, cax = CbAx, ticks = [-500, 0, 500], extend = 'both')
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=20, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.ax.set_title('Difference in \n' + dif_str[4:] +'\n (kg m$^{-2}$ yr$^{-1}$)', fontname='Helvetica', color='dimgrey', fontsize=20, pad=20)
    plt.subplots_adjust(right = 0.8)
    plt.savefig(filepath + dif_str + '_difs_btw_1p5_and_4_deg_warming.png')
    #plt.savefig(filepath + 'GCM_' + v + '_difs_+' + dT + '_deg_warming.eps')
    plt.show()

#plot_dif_1p5_to_4(dif_dict['MMM']['ME'], dif_str = 'MMM_melt_amnt')
#plot_dif_1p5_to_4(dif_dict['MMM']['RU'], dif_str = 'MMM_runoff')
#plot_dif_1p5_to_4(dif_dict['MMM']['SMB'], dif_str = 'MMM_SMB')

def mega_plot():
    # Caption: Multi-model mean differences in surface melt (ME), runoff (RU) and surface mass balance (SMB) between the
    # reference "historical" period (1980-2010) and the 30-year period where the multi-model mean global mean surface
    # warming is equal to 1.5, 2 and 4 degrees celsius above pre-industrial temperatures. The location of ice shelves is
    # indicated with the solid contour. Note that the colourbar scale is reversed for SMB.
    # plot differences between 1950-79 and +1.5/2/4 degree world
    fig, axs = plt.subplots(3,3, figsize = (20,14))
    axs = axs.flatten()
    for ax in axs:
        ax.contour(masks['shelf'], levels=[0], colors='k', linewidths=0.8)
        ax.axis('off')
    CbAx = fig.add_axes([0.85,0.25, 0.03, 0.5])
    CbAx2 = CbAx.twinx()
    plid = 0
    axs[0].annotate('1.5$^{\circ}$C', (-0.3, 0.5), xycoords = 'axes fraction', fontsize=20, color='dimgrey', )
    axs[3].annotate('2$^{\circ}$C',  (-0.3, 0.5), xycoords = 'axes fraction', fontsize=20, color='dimgrey', )
    axs[6].annotate('4$^{\circ}$C',  (-0.3, 0.5), xycoords = 'axes fraction', fontsize=20, color='dimgrey', )
    for dT in [0,1,2]: # for each dT
        for i, j in enumerate(MMM_dict.keys()):
            if j == 'SMB':
                cm = 'RdBu'
                cax = CbAx
                tick_pos = 'left'
                tick_labs = ['+500 (SMB)', '0', '-500 (SMB)']
            else:
                cm = 'RdBu_r'
                cax = CbAx2
                tick_pos = 'right'
                tick_labs = ['-500 (ME, RU)', '0', '+500 (ME, RU)']
            # Find mean difference between slice where warming = +dT and start of the simulation (i.e. the difference at dT)
            c = axs[plid + i].pcolormesh(masks['shelf'] * (np.mean(MMM_dict[j][slice_dict['MMM'][dT]-29:slice_dict['MMM'][dT]].data, axis = 0)-(np.mean(MMM_dict[j][:29].data, axis = 0))), cmap = cm, vmin = -500, vmax = 500)
            axs[plid + i].set_title(j, fontsize=20, color='dimgrey', )
            cb = plt.colorbar(c, cax=cax, ticks=[-500, 0, 500], extend='both')
            cb.set_ticklabels(tick_labs)
            cb.solids.set_edgecolor("face")
            cb.outline.set_edgecolor('dimgrey')
            cb.ax.tick_params(which='both', axis='both', labelsize=20, labelcolor='dimgrey', pad=10, size=0,
                              tick1On=False, tick2On=False)
            cb.outline.set_linewidth(2)
            cb.ax.xaxis.set_ticks_position('bottom')
            # cb.set_label(v, fontsize=20, rotation = 0, color='dimgrey', labelpad=30)
            cb.ax.set_title('Mean difference\n (kg m$^{-2}$ yr$^{-1}$)', fontname='Helvetica', color='dimgrey',
                            fontsize=20, pad=20)
            cb.ax.yaxis.set_ticks_position(tick_pos)
            cb.ax.yaxis.set_label_position(tick_pos)
        plid = plid+3
    plt.subplots_adjust(right = 0.75, left = 0.1)
    plt.savefig(filepath + 'MMM_difs_at_each_deg_warming.png')
    #plt.savefig(filepath + 'GCM_' + v + '_difs_+' + dT + '_deg_warming.eps')
    plt.show()

#mega_plot()

# Produce data frame with values under different scenarios
# Ice shelf areas, difference between 1.5 and 4 deg, as simulated by all models
df = pd.DataFrame(index = ['ME', 'RU', 'SMB'])
for i in dif_dict.keys():
    sum_ME = np.nansum(dif_dict[i]['ME'] * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
    sum_RU = np.nansum(dif_dict[i]['RU'] * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    sum_SMB = np.nansum(dif_dict[i]['SMB'] * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    df[i] = pd.Series([sum_ME, sum_RU, sum_SMB], index = ['ME', 'RU', 'SMB'])

df.to_csv(filepath + 'Ice_shelf_difs_Gt_1p5_to_4_deg' +  region + '.csv')

df = pd.DataFrame(index = ['ME', 'RU', 'SMB'])
dT = ['1p5', '2', '4', 'HIST' ]
for n, j in enumerate(dT):
    for i in dict_list.keys():
        sum_ME = np.nansum(np.mean(dict_list[i]['ME'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        sum_RU = np.nansum(np.mean(dict_list[i]['RU'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        sum_SMB = np.nansum(np.mean(dict_list[i]['SMB'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        df[i] = pd.Series([sum_ME, sum_RU, sum_SMB], index = ['ME', 'RU', 'SMB'])
    sum_ME = np.nansum(np.mean(MMM_dict['ME'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    sum_RU = np.nansum(np.mean(MMM_dict['RU'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    sum_SMB = np.nansum(np.mean(MMM_dict['SMB'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    df['MMM'] = pd.Series([sum_ME, sum_RU, sum_SMB], index=['ME', 'RU', 'SMB'])
    df.to_csv(filepath + 'Ice_shelf_abs_Gt_'+ j + region + '_deg.csv')

# Calculate melt extent
for m in dict_list.keys():
    dict_list[m]['melt_ext'] = dict_list[m]['ME'].copy()
    dict_list[m]['melt_ext'].data[dict_list[m]['melt_ext'].data>1.] = 1
    dict_list[m]['melt_ext'].data[dict_list[m]['melt_ext'].data<1.] = 0
    dict_list[m]['melt_ext'] = (np.sum(dict_list[m]['melt_ext'].data * stats['grid_area'].data * masks['shelf'] * (35000 * 35000), axis = (1,2))/(stats['grid_area'].data * masks['shelf'] * (35000 * 35000)).sum())*100

for m in dict_list.keys():
    plt.plot(dict_list[m]['melt_ext'], label = m)

# Calculate melt extent above 725 mm w.e yr-1
for m in dict_list.keys():
    dict_list[m]['melt_ext_725'] = dict_list[m]['ME'].copy()
    dict_list[m]['melt_ext_725'].data[dict_list[m]['melt_ext_725'].data < 725.] = 0
    dict_list[m]['melt_ext_725'].data[dict_list[m]['melt_ext_725'].data>725.] = 1
    dict_list[m]['melt_ext_725'] = (np.sum(dict_list[m]['melt_ext_725'].data * stats['grid_area'].data * masks['shelf'] * (35000 * 35000), axis = (1,2))/(stats['grid_area'].data * masks['shelf'] * (35000 * 35000)).sum())*100

dict_list['ACCESS1.3']['melt_ext_725'] = dict_list['ACCESS1.3']['melt_ext_725'][1:]

def melt_extent_plot():
    fig, ax = plt.subplots(1,1,figsize = (14,8))
    ax.fill_between(range(1980, 2100), np.max([dict_list['CESM2']['melt_ext_725'],dict_list['NorESM1-M']['melt_ext_725'], dict_list['ACCESS1.3']['melt_ext_725'],
                      dict_list['CNRM-CM6-1']['melt_ext_725']], axis = 0), np.min([dict_list['CESM2']['melt_ext_725'],dict_list['NorESM1-M']['melt_ext_725'],
                                                                                   dict_list['ACCESS1.3']['melt_ext_725'],dict_list['CNRM-CM6-1']['melt_ext_725']], axis = 0), color = 'dimgrey', alpha = 0.5, zorder = 1)
    for m in dict_list.keys():
        ax.plot(range(1980, 2100), dict_list[m]['melt_ext_725'], label = m, linestyle = '--', alpha = 0.8, zorder = 2)
    ax.plot(range(1980, 2100),np.mean([dict_list['CESM2']['melt_ext_725'],dict_list['NorESM1-M']['melt_ext_725'], dict_list['ACCESS1.3']['melt_ext_725'],
                      dict_list['CNRM-CM6-1']['melt_ext_725']], axis = 0), color = 'k', linewidth = 3, label = 'MMM', zorder = 3)
    ax.set_xticks([1980, 2000, 2020, 2040, 2060, 2080, 2100])
    ax.set_xlim(1980,2100)
    ax.set_ylabel('Ice shelf melt extent\n> 725 mm w.e. yr$^{-1}$ (%)', labelpad = 150, rotation = 0, fontsize = 20, color = 'dimgrey')
    lgd = ax.legend(bbox_to_anchor=(0.05, 1.), loc = 2, fontsize = 20)
    frame = lgd.get_frame()
    frame.set_facecolor('white')
    for ln in lgd.get_texts():
        plt.setp(ln, color='dimgrey', fontsize = 18)
    lgd.get_frame().set_linewidth(0.0)
    plt.setp(ax.spines.values(), linewidth=2, color='dimgrey')
    ax.tick_params(axis='both', which='both', labelsize=24, tick1On=False, tick2On=False, labelcolor='dimgrey', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.subplots_adjust(left = 0.35, right = 0.97)
    plt.savefig(filepath + 'pct_ice_shf_area_'+ region + '_abv_725.png')
    plt.show()

melt_extent_plot()

MMM_dict['melt_ext_725'] = np.mean([dict_list['CESM2']['melt_ext_725'],dict_list['NorESM1-M']['melt_ext_725'], dict_list['ACCESS1.3']['melt_ext_725'],
                      dict_list['CNRM-CM6-1']['melt_ext_725']], axis = 0)

df = pd.DataFrame(index = ['HIST', '1.5', '2', '4'])
for m in dict_list.keys():
    df[m] = pd.Series([np.mean(dict_list[m]['melt_ext_725'].data[:29]),
                       np.mean(dict_list[m]['melt_ext_725'].data[slice_dict[m][0] - 29: slice_dict[m][0]]),
                               np.mean(dict_list[m]['melt_ext_725'].data[slice_dict[m][1] - 29: slice_dict[m][1]]),
                                       np.mean(dict_list[m]['melt_ext_725'].data[
                                               slice_dict[m][2] - 29: slice_dict[m][2]])], index = ['HIST', '1.5', '2', '4'])
    df['MMM'] = pd.Series([np.mean(MMM_dict['melt_ext_725'].data[:29]), np.mean(MMM_dict['melt_ext_725'].data[slice_dict['MMM'][0] - 29: slice_dict['MMM'][0]]),
                               np.mean(MMM_dict['melt_ext_725'].data[slice_dict['MMM'][1] - 29: slice_dict['MMM'][1]]),
                                       np.mean(MMM_dict['melt_ext_725'].data[
                                               slice_dict['MMM'][2] - 29: slice_dict['MMM'][2]])], index = [ 'HIST','1.5', '2', '4'])

df.to_csv(filepath + 'pct_ice_shelf_area_'+ region + '_abv_725.csv')

rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 20
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans', 'Verdana']
rcParams['ytick.color'] = "dimgrey"
rcParams['xtick.color'] = "dimgrey"
rcParams['ytick.color'] = "dimgrey"
rcParams['legend.fontsize'] = 18
rcParams['legend.edgecolor'] = []
rcParams['figure.figsize'] = [12,8]

def boxplots():
    # Create box plot for each variable
    df_1p5 = pd.read_csv(filepath + 'Ice_shelf_abs_Gt_1p5_deg.csv', index_col = 0, )
    df_1p5 = df_1p5.transpose()
    df_1p5 = df_1p5.assign(Scenario=1.5)#header = ['ACCESS1.3','CESM2', 'NorESM1-M', 'CNRM-CM6-1']
    df_2 = pd.read_csv(filepath + 'Ice_shelf_abs_Gt_2_deg.csv', index_col = 0)
    df_2 = df_2.transpose()
    df_2 = df_2.assign(Scenario=2)
    df_4 = pd.read_csv(filepath + 'Ice_shelf_abs_Gt_4_deg.csv', index_col = 0)
    df_4 = df_4.transpose()
    df_4 = df_4.assign(Scenario=4)
    cdf = pd.concat([df_1p5, df_2, df_4])
    mdf = pd.melt(cdf, id_vars=['Scenario'], var_name=['Var'])
    st = sns.axes_style("white")
    sns.set(style ="white", font_scale =1.5, rc={'figure.figsize':(12,8)})
    g = sns.catplot(x="Var", y="value", hue="Scenario", palette = 'Reds', data=mdf, kind="box")
    g.spines['left'].set_linewidth(2)
    g.spines['left'].set_color('dimgrey')
    g.spines['bottom'].set_linewidth(2)
    g.spines['bottom'].set_color('dimgrey')
    g.set_xlabel('Component', fontsize = 20, color = 'dimgrey')
    g.set_ylabel('Difference\n(Gt yr$^{-1}$)', labelpad = 50, rotation = 0, fontsize = 20, color = 'dimgrey')
    g.tick_params(labelsize=20, labelcolor = 'dimgrey')
    # Boxplot of differences
    df = pd.read_csv(filepath + 'Ice_shelf_difs_Gt_1p5_to_4_deg.csv', index_col = 0)
    dif_data = pd.melt(df.transpose())
    st = sns.axes_style("white")
    sns.set(style ="white", font_scale =1.5, rc={'figure.figsize':(12,8)})
    g = sns.boxplot(x="variable", y="value", palette = 'Blues', data=dif_data, width = 0.35)
    g.set_xlabel('Component', fontsize = 20, color = 'dimgrey')
    g.set_ylabel('Difference\n(Gt yr$^{-1}$)', labelpad = 50, rotation = 0, fontsize = 20, color = 'dimgrey')
    g.tick_params(labelsize=20, labelcolor = 'dimgrey')
    sns.despine()
    g.spines['left'].set_linewidth(2)
    g.spines['left'].set_color('dimgrey')
    g.spines['bottom'].set_linewidth(2)
    g.spines['bottom'].set_color('dimgrey')
    plt.axhline(y = 0, linewidth = 2, color = 'dimgrey', linestyle = '--')
    plt.subplots_adjust(left = 0.2)
    #plt.savefig(filepath + 'Boxplot_difs_components.png')
    plt.show()


# Questions:
# 1. what is current melt duration over ice shelves?
##      a. Step 1: mask melt over ice shelves
##      b. Step 2: calculate number of days per year where melting is detected - over each ice shelf? What about sectors?
# 2. what is the current melt and runoff extent over ice shelves?
# 3. What will be the likely melt duration in a + 2 degree world? What about + 4?
# 4. What will be the likely melt and runoff extent in a +2 or +4 degree world?
# 5. How does the 0 degree isotherm change between present climate and these scenarios?

# Extract warmer world periods
# For each GCM, need to find 30-year time period where average mean surface temperature increase is equal to 1.5 degrees,
# 2 degrees, 4 degrees etc. above reference period (1950-79)

# Time periods:
#
# ACCESS1-3:
# 2002-31 (1.5) / 2014-43 (2) / 2054-83 (4)
#
# CESM2:
# 2003-32 (1.5) / 2013-42 (2) / 2045-74 (4)
#
# NorESM1-M:
# 2010-39 (1.5) / 2023-22 (2) / 2066-95 (4)
#
# CNRM-CM6-1:
# 2005-34 (1.5) / 2018-47 (2) / 2047-76 (4)

# Isotherm. 4 panel plot showing:
# Plot contour of 0 deg in a) present, b) +1.5 deg, c) +2 deg, d) +4 deg

# Calculate MMM of SMB, ME, RU, ME dur, isotherm etc.