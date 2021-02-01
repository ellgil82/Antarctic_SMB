''' Script for processing and visualising Antarctic surface mass balance data from model projections.

Author: Ella Gilbert, 2020.

Dependencies:
- python 3
- cartopy 0.18.0
- iris 2.2

'''

# Define host HPC where script is running
host = 'hd'

# Define host-specific filepath and location of additional python tool scripts
import sys
if host == 'jasmin':
    filepath = '/gws/nopw/j04/bas_climate/users/ellgil82/AntSMB/MAR/'
    sys.path.append('/gws/nopw/j04/bas_climate/users/ellgil82/scripts/Tools/')
elif host == 'bsl':
    filepath = '/data/mac/ellgil82/AntSMB/MAR/'
    sys.path.append('/users/ellgil82/scripts/Tools/')
elif host == 'rdg':
    filepath = '/home/users/ke923690/AntSMB/'
    sys.path.append('/home/users/ke923690/Python\ Scripts/Tools/')
elif host == 'hd':
    filepath = 'D:\\Data\\AntSMB\\MAR\\'
    sys.path.append('C:/Users/Ella/OneDrive - University of Reading/Scripts/Tools/')

import iris
import numpy as np
import matplotlib.pyplot as plt
import iris.analysis.cartography
import iris.coord_categorisation
from divg_temp_colourmap import shiftedColorMap
import matplotlib
import pandas as pd
from matplotlib.lines import Line2D

# Isolate region of interest ('' for whole continent)
region = input("Which region are you interested in? \n\nPress enter for whole continent\n"
               "Enter \"WA\" for West Antarctica\nEnter \"EA\" for East Antarctica\n"
               "Enter \"AP\" for Antarctic Peninsula.\n\nSo, what will it be?\n\n")#region = '' #'AP'

def load_sims():
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
            dict_list[j]['SF'] = iris.load_cube(filepath + 'SF_MAR_' + j + '_rcp8.5_future.nc')
            dict_list[j]['RF'] = iris.load_cube(filepath + 'RF_MAR_' + j + '_rcp8.5_future.nc')
            dict_list[j]['TT'] = iris.load_cube(filepath + 'TT_MAR_' + j + '_rcp8.5_future.nc')
        except:
            dict_list[j]['RU'] = iris.load_cube(filepath + 'RU_MAR_'+ j +'_ssp585_future.nc')
            dict_list[j]['ME'] = iris.load_cube(filepath + 'ME_MAR_' + j + '_ssp585_future.nc')
            dict_list[j]['SMB'] = iris.load_cube(filepath + 'SMB_MAR_' + j + '_ssp585_future.nc')
            dict_list[j]['SF'] = iris.load_cube(filepath + 'SF_MAR_' + j + '_ssp585_future.nc')
            dict_list[j]['RF'] = iris.load_cube(filepath + 'RF_MAR_' + j + '_ssp585_future.nc')
            dict_list[j]['TT'] = iris.load_cube(filepath + 'TT_MAR_' + j + '_ssp585_future.nc')
        for k in ['RU', 'ME', 'SMB']:
            dict_list[j][k] = dict_list[j][k][:,0,:,:]
            iris.coord_categorisation.add_year(dict_list[j][k], 'time', name='year')
            dict_list[j][k] = dict_list[j][k].aggregated_by(['year'], iris.analysis.SUM) # return as annual sum
        for k in ['RF', 'SF']:
            iris.coord_categorisation.add_year(dict_list[j][k], 'time', name='year')
            dict_list[j][k] = dict_list[j][k].aggregated_by(['year'], iris.analysis.SUM) # return as annual sum
        dict_list[j]['TT'] = dict_list[j]['TT'][:,0,:,:]
        iris.coord_categorisation.add_year(dict_list[j]['TT'], 'time', name='year')
        dict_list[j]['TT'] = dict_list[j]['TT'].aggregated_by(['year'], iris.analysis.MEAN) # return as mean
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

dict_list, masks, stats = load_sims()
AIS_dict_list = dict_list.copy()
AIS_masks = masks.copy()
AIS_stats = stats.copy()

if region == 'AP':
    for m in dict_list.keys():
        for k in ['RU', 'ME', 'SMB', 'TT', 'SF', 'RF']:
            dict_list[m][k] = dict_list[m][k][:, 80:120, :50]
    for k in stats.keys():
        stats[k] = stats[k][80:120, :50]
    for n in masks.keys():
        masks[n] = masks[n][80:120, :50]
elif region == 'WA':
    for m in dict_list.keys():
        for k in ['RU', 'ME', 'SMB', 'TT', 'SF', 'RF']:
            dict_list[m][k] = dict_list[m][k][:, 20:78, 30:97]
    for k in stats.keys():
        stats[k] = stats[k][20:78, 30:97]
    for n in masks.keys():
        masks[n] = masks[n][20:78, 30:97]
elif region == 'EA':
    for k in stats.keys():
        stats[k] = stats[k][:, 67:]
    for k in stats.keys():
        stats[k].data[30:65, :35] = 0
    for n in masks.keys():
        masks[n] = masks[n][:, 67:]
        masks[n][30:65, :35] = 0
    for m in dict_list.keys():
        for k in ['RU', 'ME', 'SMB', 'TT', 'SF', 'RF']:
            dict_list[m][k] = dict_list[m][k][:, :, 67:]
            dict_list[m][k].data[:, 30:65, :35] = 0

def runoff_dur():
    ''' Load dictionary of runoff duration for each GCM'''
    ACCESS_melt = {}
    CNRM_melt = {}
    NOR_melt = {}
    CESM_melt = {}
    runoff_dict = {'ACCESS1.3': ACCESS_melt, 'CESM2': CESM_melt, 'NorESM1-M': NOR_melt, 'CNRM-CM6-1': CNRM_melt}
    # Load invariant data
    grd_ice = iris.load_cube(filepath + 'MARcst-AN35km-176x148.cdf', 'Grounded ice')
    continent = iris.load_cube(filepath + 'MARcst-AN35km-176x148.cdf', 'IF SOL3 EQ 4 THEN 1 ELSE 0')
    rock = iris.load_cube(filepath + 'MARcst-AN35km-176x148.cdf', 'Rock')
    # Create masks
    ais = np.ma.masked_equal(continent.data, 0)  # mask areas of ocean
    grd = np.ma.masked_less(grd_ice.data, 30)
    # mask non-ice, non-grounded ice grid points and multiply by grid area to find true area
    grounded_mask = grd * ais
    grounded_mask[grounded_mask >= 30] = 1
    shf = np.ma.masked_greater(grd_ice.data, 50)
    shf_msk = ais * shf
    shf_msk = np.ma.masked_where(rock.data > 30, shf_msk)
    shelf_mask = ~shf_msk.mask  # Shelf areas are = 1
    for i, j in enumerate(['ACCESS1.3', 'CESM2', 'NorESM1-M', 'CNRM-CM6-1']):
        try:
            runoff_dict[j]['RU_dur'] = iris.load_cube(filepath + 'RU_MAR_' + j + '_rcp8.5_future.nc')
        except:
            runoff_dict[j]['RU_dur'] = iris.load_cube(filepath + 'RU_MAR_' + j + '_ssp585_future.nc')
        runoff_dict[j]['RU_dur'] = runoff_dict[j]['RU_dur'][:,0,:,:]
        iris.coord_categorisation.add_year(runoff_dict[j]['RU_dur'], 'time', name='year')
        runoff_dict[j]['RU_dur'].data[runoff_dict[j]['RU_dur'].data < 1] = 0
        runoff_dict[j]['RU_dur'].data[np.broadcast_to(shelf_mask, runoff_dict[j]['RU_dur'].shape) == 0] = np.nan
        runoff_dict[j]['RU_dur'].data[runoff_dict[j]['RU_dur'].data >= 1] = 1.
        runoff_dict[j]['RU_dur'] = runoff_dict[j]['RU_dur'].aggregated_by(['year'], iris.analysis.SUM)
    return runoff_dict

runoff_dict= runoff_dur()

AIS_mask = np.zeros((148, 176))
AIS_mask[AIS_mask == 0] = np.nan
AIS_mask[masks['ais'] == 1] = 1.
WA_mask = np.zeros((148, 176))
WA_mask[WA_mask == 0] = np.nan
WA_mask[ 20:78, 30:97] = 1.
EA_mask = np.zeros((148, 176))
EA_mask[EA_mask == 0] = np.nan
EA_mask[:, 67:] = 1.
EA_mask[30:65, :35] = np.nan
AP_mask = np.zeros((148, 176))
AP_mask[AP_mask == 0] = np.nan
AP_mask[ 80:120, :50] = 1.


# Find GCM warming periods ref 1950-79
ACCESS_T = iris.load_cube(filepath + 'year_ACCESS1-3.nc2', 'Near-Surface Air Temperature')
CESM_T = iris.load_cube(filepath + 'year_CESM2.nc2', 'Near-Surface Air Temperature')
NorESM_T = iris.load_cube(filepath + 'year_NorESM1-M.nc2', 'Near-Surface Air Temperature')
CNRM_T = iris.load_cube(filepath + 'year_CNRM-CM6-1.nc2', 'Near-Surface Air Temperature')
MMM_T = (ACCESS_T.data.mean(axis = (1,2)) + CNRM_T.data.mean(axis = (1,2)) + CESM_T.data.mean(axis = (1,2)) + NorESM_T.data.mean(axis = (1,2)))/4

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

rolling_mn = pd.Series(anom_dict['NorESM1-M']).rolling(window=30).mean().iloc[30-1:].values
np.where(rolling_mn>1.5) # or 2/ 4 deg

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
        axs[i].contourf(masks['ais'], colors='white', vmin=0, vmax=1, zorder = 1)  # manually plot ocean as white
        if difs_or_abs == 'difs':
            if v == 'SMB' or v == 'SF':
                lims = (-200, 500)
                bwr_zero = shiftedColorMap(cmap=matplotlib.cm.RdBu, min_val=lims[0], max_val=lims[1], name='bwr_zero',
                                           var=np.mean(MMM_dict[v][slice_dict['MMM'][tuple_idx] - 29:slice_dict['MMM'][tuple_idx]].data, axis=0) -np.mean(MMM_dict[v][:29].data, axis=0), start=0.15, stop=0.85)
                cm = bwr_zero
                #cm = 'RdBu'
                c = axs[i].pcolormesh(np.ma.masked_where(masks['ais']==0, np.mean(dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data, axis=0) -np.mean(dict_list[j][v][:29].data, axis=0)), cmap = cm, vmin = -200, vmax = 500) # calculate difference relative to historical period
            elif v == 'RF':
                lims = (0, 100)
                shiftedColorMap(cmap=matplotlib.cm.Blues, min_val=lims[0], max_val=lims[1], name='blues_tight',
                                var=(np.mean(dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data,axis=0)), start=0.15, stop=0.85)
                cm = matplotlib.cm.Blues
                c = axs[i].pcolormesh(np.ma.masked_where(masks['ais']==0, np.mean(dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data, axis=0) -np.mean(dict_list[j][v][:29].data, axis=0)), cmap = cm, vmin = 00, vmax = 100) # calculate difference relative to historical period
            else:
                lims = (-500, 500)
                bwr_zero = shiftedColorMap(cmap=matplotlib.cm.RdBu_r, min_val=lims[0], max_val=lims[1],  name='bwr_zero',
                                           var=(np.mean(
                                               dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data,
                                               axis=0)), start=0.15, stop=0.85)
                cm = bwr_zero
                # Find mean difference between slice where warming = +dT and start of the simulation (i.e. the difference at dT)
                c = axs[i].pcolormesh(np.ma.masked_where(masks['ais']==0, np.mean(dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data, axis=0) -np.mean(dict_list[j][v][:29].data, axis=0)), cmap = cm, vmin = -500, vmax = 500) # calculate difference relative to historical period
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
                c = axs[i].pcolormesh(np.ma.masked_where((np.mean(dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data, axis=0))<=(np.mean(dict_list[j]['SF'][:29].data, axis = 0)*0.7),(np.mean(dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data, axis=0))), zorder = 4, cmap=cm, vmin=lims[0], vmax=lims[1])
                c2 = axs[i].pcolormesh(np.ma.masked_where(
                    (np.mean(dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data, axis=0)) > (
                                np.mean(dict_list[j]['SF'][:29].data, axis=0) * 0.7),
                    (np.mean(dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data, axis=0))),
                                      color=['white'], vmin=lims[0], vmax=lims[1], zorder = 2)
            else:
                c = axs[i].pcolormesh( np.mean(dict_list[j][v][slice_dict[j][tuple_idx] - 29:slice_dict[j][tuple_idx]].data, axis=0), cmap=cm, vmin=lims[0], vmax=lims[1])
        axs[i].contour(masks['shelf'], levels=  [0], colors = 'k', linewidths = 0.8, zorder = 5)
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
    plt.savefig(filepath + 'GCM_' + v + '_' + difs_or_abs + '_' + dT + '_deg_warming.eps')
    plt.show()

#plot_scenarios('SF', '4', 'difs', 'no')

#for i in ['RF', 'SF']:
#    for j in ['1p5', '2', '4']:
#       plot_scenarios(i, j, 'difs', threshold='no')

#for j in ['hist', '1p5', '2', '4']:
#    plot_scenarios('ME', j, 'abs', threshold = 'yes')

def plot_runoff_dur(runoff_dict, dT):
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
    for i, j in enumerate(runoff_dict.keys()):
        # Find mean difference between slice where warming = +dT and start of the simulation (i.e. the difference at dT)
        c = axs[i].pcolormesh(masks['shelf'] * (np.mean(runoff_dict[j]['RU_dur'][slice_dict[j][tuple_idx]-29:slice_dict[j][tuple_idx]].data, axis = 0)-
                                                (np.mean(runoff_dict[j]['RU_dur'][:29].data, axis = 0))), cmap = 'Reds', vmin = 0, vmax = 60)
        axs[i].contour(masks['shelf'], levels=  [0], colors = 'k', linewidths = 0.8)
        axs[i].set_title(j, fontsize=20, color='dimgrey', )
        axs[i].axis('off')
    cb = plt.colorbar(c, cax = CbAx, ticks = [0, 30, 60, 100, 365], extend = 'max')
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=20, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    #cb.set_label(v, fontsize=20, rotation = 0, color='dimgrey', labelpad=30)
    cb.ax.set_title('Mean runoff duration\n (d yr$^{-1}$)', fontname='Helvetica', color='dimgrey', fontsize=20, pad=20)
    plt.subplots_adjust(right = 0.8)
    plt.savefig(filepath + 'GCM_runoff_duration_difs_+'+ dT + '_deg_warming.png')
    plt.savefig(filepath + 'GCM_runoff_duration_difs_+' + dT + '_deg_warming.eps')
    plt.show()

#or t in ['hist', '1p5', '2', '4']:
#    plot_runoff_dur(runoff_dict, t)

#for j in ['hist', '1p5', '2', '4']:
#    plot_scenarios('RU_dur', dT = j, difs_or_abs='abs', threshold = 'no')

# Create dictionar of MMM values
MMM_dict = {}
for j in dict_list['CESM2'].keys():
    print(j)
    MMM_dict[j] = (dict_list['ACCESS1.3'][j][:120].data + dict_list['CESM2'][j].data  + dict_list['NorESM1-M'][j].data
                   + dict_list['CNRM-CM6-1'][j].data )/4 # Find multi-model mean, ignoring differing time coordinate definitions

MMM_dict['RU_dur'] = (runoff_dict['ACCESS1.3']['RU_dur'][:120].data + runoff_dict['CESM2']['RU_dur'].data + runoff_dict['NorESM1-M']['RU_dur'].data + runoff_dict['CNRM-CM6-1']['RU_dur'].data)/4

def plot_difs(dif, dif_str):
    fig, axs = plt.subplots(1,1, figsize = (12,9))
    # Find mean difference between slice where warming = +dT and start of the simulation (i.e. the difference at dT)
    CbAx = fig.add_axes([0.85, 0.25, 0.03, 0.5])
    if dif_str[-3:] == 'SMB':
        c = axs.pcolormesh(dif, cmap = 'RdBu', vmin = -500, vmax = 500) # masks['shelf'] * dif
    elif dif_str[4:7] == 'melt':
        c = axs.pcolormesh(np.ma.masked_where((np.mean(MMM_dict['ME'][slice_dict['MMM'][2]-29:slice_dict['MMM'][2]].data, axis=0) < 725), dif), cmap='RdBu_r', vmin=-500, vmax=500)
    elif dif_str[4:] == 'RU_dur':
        c = axs.pcolormesh((np.mean(MMM_dict['RU_dur'][slice_dict['MMM'][2]-29:slice_dict['MMM'][2]].data, axis =0)-np.mean(MMM_dict['RU_dur'][:29].data, axis = 0)), cmap='RdBu_r', vmin=-60, vmax=60)
    else:
        c = axs.pcolormesh(dif, cmap='RdBu_r', vmin=-50, vmax=50)
    #axs.contour(np.mean(MMM_dict['ME'][slice_dict['MMM'][2]-29:slice_dict['MMM'][2]].data, axis=0),  levels = [725], colors = 'cyan', linewidths = 3)
    axs.contour(masks['shelf'], levels=  [0], colors = 'k', linewidths = 0.8)
    axs.axis('off')
    cb = plt.colorbar(c, cax = CbAx, ticks = [-60, 0, 60], extend = 'both')
    cb.solids.set_edgecolor("face")
    cb.outline.set_edgecolor('dimgrey')
    cb.ax.tick_params(which='both', axis='both', labelsize=20, labelcolor='dimgrey', pad=10, size=0, tick1On=False, tick2On=False)
    cb.outline.set_linewidth(2)
    cb.ax.xaxis.set_ticks_position('bottom')
    cb.ax.set_title('Difference in \n' + dif_str[4:] +'\n (kg m$^{-2}$ yr$^{-1}$)', fontname='Helvetica', color='dimgrey', fontsize=20, pad=20)
    plt.subplots_adjust(right = 0.8)
    plt.savefig(filepath + dif_str + '_difs_btw_1p5_and_4_deg_warming.png')
    plt.savefig(filepath + dif_str + '_difs_btw_1p5_and_4_deg_warming.eps')
    plt.show()

#plot_difs(dif_dict['MMM']['ME'], dif_str = 'MMM_melt_amnt')
#plot_difs(dif_dict['MMM']['RU'], dif_str = 'MMM_runoff')
#plot_difs(dif_dict['MMM']['SMB'], dif_str = 'MMM_SMB')
#plot_difs(MMM_dict['RU_dur'], dif_str = 'MMM_RU_dur')

def mega_plot():
    # plot differences between 1950-79 and +1.5/2/4 degree world
    # Caption: Multi-model mean melt (ME), runoff (RU), and surface mass balance (SMB) under the 1.5°C, 2°C and 4°C
    # warming scenarios, expressed as the difference relative to the historical period, 1980-2009. Warm and cool colours
    # indicate an increase and decrease relative to the historical period, respectively, for melt and runoff, while the
    # colour scale is reversed for SMB.
    fig, axs = plt.subplots(3,3, figsize = (20,22))
    axs = axs.flatten()
    for ax in axs:
        ax.contour(masks['shelf'], levels=[0], colors='k', linewidths=0.8)
        ax.axis('off')
    CbAx = fig.add_axes([0.25,0.05, 0.5, 0.03])
    CbAx2 = CbAx.twiny()
    plid = 0
    axs[0].annotate('1.5$^{\circ}$C', (-0.2, 0.5), xycoords = 'axes fraction', fontsize=36, color='dimgrey', )
    axs[3].annotate('2$^{\circ}$C',  (-0.2, 0.5), xycoords = 'axes fraction', fontsize=36, color='dimgrey', )
    axs[6].annotate('4$^{\circ}$C',  (-0.2, 0.5), xycoords = 'axes fraction', fontsize=36, color='dimgrey', )
    for i, j in enumerate(['ME', 'RU', 'SMB']):
        axs[i].set_title(j, fontsize=28, color='dimgrey', )
    for dT in [0,1,2]: # for each dT
        for i, j in enumerate(['ME', 'RU', 'SMB']):
            if j == 'SMB':
                cm = 'RdBu'
                cax = CbAx
                tick_pos = 'bottom'
                tick_labs = ['+500\n(SMB)', '+250', '0', '-250', '-500\n(SMB)']
            else:
                cm = 'RdBu_r'
                cax = CbAx2
                tick_pos = 'top'
                tick_labs = ['(ME, RU)\n-500', '-250', '0', '+250', '(ME, RU)\n+500']
            # Find mean difference between slice where warming = +dT and start of the simulation (i.e. the difference at dT)
            c = axs[plid + i].pcolormesh( np.ma.masked_where((masks['ais'] == 0),(np.mean(MMM_dict[j][slice_dict['MMM'][dT]-29:slice_dict['MMM'][dT]].data, axis = 0)-(np.mean(MMM_dict[j][:29].data, axis = 0)))), cmap = cm, vmin = -500, vmax = 500)
            #axs[plid + i].contourf(masks['ais'] == 1, colors= 'white')
            cb2 = plt.colorbar(c, cax=cax, ticks=[-500, -250, 0, 250, 500], extend='both', orientation = 'horizontal')
            cb = plt.colorbar(c, cax=cax, ticks=[-500, -250, 0, 250, 500], extend='both', orientation = 'horizontal')
            cb.set_ticklabels(tick_labs)
            cb.solids.set_edgecolor("face")
            cb.outline.set_edgecolor('dimgrey')
            cb.ax.tick_params(which='both', axis='both', labelsize=32, labelcolor='dimgrey', pad=10, size=0,
                              tick1On=False, tick2On=False)
            cb.outline.set_linewidth(2)
            cb.ax.set_title('Mean difference (kg m$^{-2}$ yr$^{-1}$)', fontname='Helvetica', color='dimgrey',
                            fontsize=32, pad=20)
            cb.ax.xaxis.set_ticks_position(tick_pos)
            cb.ax.xaxis.set_label_position(tick_pos)
        plid = plid+3
    l, b, w, h = cb.ax.get_position().bounds
    cb.ax.set_position([l, b, w, h])
    cb2.ax.set_position([l, b, w, h])
    plt.subplots_adjust(right = 0.95, bottom=0.18, top = 0.95, left = 0.07, wspace=0.0, hspace=0.0)
    plt.savefig(filepath + 'MMM_difs_at_each_deg_warming.png')
    plt.savefig(filepath + 'MMM_difs_at_each_deg_warming.eps')
    plt.show()

#mega_plot()

# Produce dataframe with absolute values for each SMB component, summed over ice shelves
df = pd.DataFrame(index = ['SF', 'RF', 'ME', 'RU', 'SMB'])#'ME_dur',
dT = ['1p5', '2', '4', 'HIST' ]
for n, j in enumerate(dT):
    for i in dict_list.keys():
        sum_SF = np.nansum(np.mean(dict_list[i]['SF'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        sum_RF = np.nansum(np.mean(dict_list[i]['RF'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        sum_ME = np.nansum(np.mean(dict_list[i]['ME'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        #mn_ME_dur = np.nanmean( np.mean(runoff_dict[i]['ME_dur'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) * stats['grid_area'].data *masks['shelf'] * (35000 * 35000)) / 1e12
        sum_RU = np.nansum(np.mean(dict_list[i]['RU'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        sum_SMB = np.nansum(np.mean(dict_list[i]['SMB'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        df[i] = pd.Series([sum_SF, sum_RF, sum_ME, sum_RU, sum_SMB], index = ['SF', 'RF', 'ME',   'RU', 'SMB'])
        sum_ME = np.nansum(np.mean(MMM_dict['ME'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    sum_SF = np.nansum(np.mean(MMM_dict['SF'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    sum_RF = np.nansum(np.mean(MMM_dict['RF'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    #mn_ME_dur =  np.nanmean(np.mean(MMM_dict['ME_dur'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    sum_RU = np.nansum(np.mean(MMM_dict['RU'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    sum_SMB = np.nansum(np.mean(MMM_dict['SMB'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    df['MMM'] = pd.Series([sum_SF, sum_RF, sum_ME,  sum_RU, sum_SMB], index=['SF', 'RF', 'ME',  'RU', 'SMB'])#'ME_dur',
    df.to_csv(filepath + 'Ice_shelf_abs_Gt_'+ j + region + '_deg.csv')

# Produce equivalent dataframe with increase/decrease above/below historical mean
df = pd.DataFrame(index = ['SF', 'RF', 'ME',   'RU', 'SMB'])#'ME_dur',
dT = ['1p5', '2', '4', 'HIST' ]
for n, j in enumerate(dT):
    for i in dict_list.keys():
        sum_SF = np.nansum((np.mean(dict_list[i]['SF'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0)- np.mean(dict_list[i]['SF'][:29].data, axis=0))  * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        sum_RF = np.nansum((np.mean(dict_list[i]['RF'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0)- np.mean(dict_list[i]['RF'][:29].data, axis=0)) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        sum_ME = np.nansum((np.mean(dict_list[i]['ME'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) - np.mean(dict_list[i]['ME'][:29].data, axis=0)) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        #mn_ME_dur = np.nanmean((np.mean(runoff_dict[i]['ME_dur'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) - np.mean(runoff_dict[i]['ME_dur'][:29].data, axis=0)) * stats[ 'grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
        sum_RU = np.nansum((np.mean(dict_list[i]['RU'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) - np.mean(dict_list[i]['RU'][:29].data, axis=0))  * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        sum_SMB = np.nansum((np.mean(dict_list[i]['SMB'][slice_dict[i][n] - 29:slice_dict[i][n]].data, axis=0) - np.mean(dict_list[i]['SMB'][:29].data, axis=0))  * stats['grid_area'].data * masks['shelf'] * (35000 * 35000))/1e12
        df[i] = pd.Series([sum_SF, sum_RF, sum_ME,  sum_RU, sum_SMB], index = ['SF', 'RF', 'ME',   'RU', 'SMB'])#'ME_dur',
    sum_SF = np.nansum((np.mean(MMM_dict['SF'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) - np.mean(MMM_dict['SF'][:29].data, axis=0)) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    sum_RF = np.nansum((np.mean(MMM_dict['RF'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) - np.mean(MMM_dict['RF'][:29].data, axis=0)) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    sum_ME = np.nansum((np.mean(MMM_dict['ME'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) - np.mean(MMM_dict['ME'][:29].data, axis=0)) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    #mn_ME_dur = np.nanmean((np.mean(MMM_dict['ME_dur'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) - np.mean( MMM_dict['ME_dur'][:29].data, axis=0)) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    sum_RU = np.nansum((np.mean(MMM_dict['RU'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) - np.mean(MMM_dict['RU'][:29].data, axis=0)) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    sum_SMB = np.nansum((np.mean(MMM_dict['SMB'][slice_dict['MMM'][n] - 29:slice_dict['MMM'][n]].data, axis=0) - np.mean(MMM_dict['SMB'][:29].data, axis=0)) * stats['grid_area'].data * masks['shelf'] * (35000 * 35000)) / 1e12
    df['MMM'] = pd.Series([sum_SF, sum_RF, sum_ME, sum_RU, sum_SMB], index=['SF', 'RF', 'ME',   'RU', 'SMB'])#'ME_dur',
    df.to_csv(filepath + 'Ice_shelf_difs_Gt_'+ j + region + '_deg.csv')

# Calculate runoff extent
for m in dict_list.keys():
    dict_list[m]['runoff_ext'] = dict_list[m]['RU'].copy()
    dict_list[m]['runoff_ext'].data[dict_list[m]['runoff_ext'].data>1.] = 1
    dict_list[m]['runoff_ext'].data[dict_list[m]['runoff_ext'].data<1.] = 0
    dict_list[m]['runoff_ext'] = (np.sum(dict_list[m]['runoff_ext'].data * stats['grid_area'].data * masks['shelf'] * (35000 * 35000), axis = (1,2))/(stats['grid_area'].data * masks['shelf'] * (35000 * 35000)).sum())*100

# Calculate melt extent above 0.7 * snowfall (Pfeffer et al., 1991)
for m in dict_list.keys():
    dict_list[m]['Pfeffer_threshold'] = np.mean(dict_list[m]['SF'][:29].data, axis=0) * 0.7 * masks['shelf']  # shld be 2D
    dict_list[m]['melt_ext_Pfeffer'] = dict_list[m]['ME'].copy()
    dict_list[m]['melt_ext_Pfeffer'].data[dict_list[m]['melt_ext_Pfeffer'].data < dict_list[m]['Pfeffer_threshold'].data] = 0
    dict_list[m]['melt_ext_Pfeffer'].data[dict_list[m]['melt_ext_Pfeffer'].data > dict_list[m]['Pfeffer_threshold'].data] = 1
    dict_list[m]['melt_ext_Pfeffer'] = (np.sum(dict_list[m]['melt_ext_Pfeffer'].data * stats['grid_area'].data * masks['shelf'] * (35000 * 35000), axis = (1,2))/(stats['grid_area'].data * masks['shelf'] * (35000 * 35000)).sum())*100

dict_list['ACCESS1.3']['melt_ext_Pfeffer'] = dict_list['ACCESS1.3']['melt_ext_Pfeffer'][:120] # amend for extra year in ACCESS1.3
dict_list['ACCESS1.3']['runoff_ext'] = dict_list['ACCESS1.3']['runoff_ext'][:120]

# Calculate runoff extent over ice shelves
def runoff_extent_plot():
    fig, ax = plt.subplots(1,1,figsize = (14,8))
    ax.fill_between(range(1980, 2100), y1 = np.max([dict_list['CESM2']['melt_ext_Pfeffer'],dict_list['NorESM1-M']['melt_ext_Pfeffer'], dict_list['ACCESS1.3']['melt_ext_Pfeffer'],
                      dict_list['CNRM-CM6-1']['melt_ext_Pfeffer']], axis = 0), y2 = np.min([dict_list['CESM2']['melt_ext_Pfeffer'],dict_list['NorESM1-M']['melt_ext_Pfeffer'],
                                                                                   dict_list['ACCESS1.3']['melt_ext_Pfeffer'],dict_list['CNRM-CM6-1']['melt_ext_Pfeffer']], axis = 0), color = 'lightgrey',  zorder = 1)
    for m in dict_list.keys():
        ax.plot(range(1980, 2100), dict_list[m]['melt_ext_Pfeffer'], label = m, zorder = 2)
    ax.plot(range(1980, 2100),np.mean([dict_list['CESM2']['melt_ext_Pfeffer'],dict_list['NorESM1-M']['melt_ext_Pfeffer'], dict_list['ACCESS1.3']['melt_ext_Pfeffer'],
                      dict_list['CNRM-CM6-1']['melt_ext_Pfeffer']], axis = 0), color = 'k', linewidth = 3, label = 'MMM', zorder = 3)
    ax.set_xticks([1980, 2000, 2020, 2040, 2060, 2080, 2100])
    ax.set_xlim(1980,2100)
    ax.set_ylim(0,100)
    ax.set_ylabel('Ice shelf melt extent\n> Pfeffer threshold (%)', labelpad = 150, rotation = 0, fontsize = 20, color = 'dimgrey')
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
    plt.savefig(filepath + 'pct_ice_shf_area_'+ region + '_Pfeffer_melt.png')
    plt.savefig(filepath + 'pct_ice_shf_area_' + region + '_Pfeffer_melt.eps')
    #plt.show()

#runoff_extent_plot()

#Add MMM runoff extent and Pfeffer melt extent to MMM dictionary
MMM_dict['melt_ext_Pfeffer'] = np.mean([dict_list['CESM2']['melt_ext_Pfeffer'],dict_list['NorESM1-M']['melt_ext_Pfeffer'], dict_list['ACCESS1.3']['melt_ext_Pfeffer'],
                      dict_list['CNRM-CM6-1']['melt_ext_Pfeffer']], axis = 0)
MMM_dict['runoff_ext'] = np.mean([dict_list['CESM2']['runoff_ext'],dict_list['NorESM1-M']['runoff_ext'], dict_list['ACCESS1.3']['runoff_ext'],
                      dict_list['CNRM-CM6-1']['runoff_ext']], axis = 0)

# Create dataframe of percentage of area where melt above the Pfeffer threshold is simulated
df = pd.DataFrame(index = ['HIST', '1.5', '2', '4'])
for m in dict_list.keys():
    df[m] = pd.Series([np.mean(dict_list[m]['melt_ext_Pfeffer'].data[:29]),
                       np.mean(dict_list[m]['melt_ext_Pfeffer'].data[slice_dict[m][0] - 29: slice_dict[m][0]]),
                               np.mean(dict_list[m]['melt_ext_Pfeffer'].data[slice_dict[m][1] - 29: slice_dict[m][1]]),
                                       np.mean(dict_list[m]['melt_ext_Pfeffer'].data[
                                               slice_dict[m][2] - 29: slice_dict[m][2]])], index = ['HIST', '1.5', '2', '4'])
    df['MMM'] = pd.Series([np.mean(MMM_dict['melt_ext_Pfeffer'].data[:29]), np.mean(MMM_dict['melt_ext_Pfeffer'].data[slice_dict['MMM'][0] - 29: slice_dict['MMM'][0]]),
                               np.mean(MMM_dict['melt_ext_Pfeffer'].data[slice_dict['MMM'][1] - 29: slice_dict['MMM'][1]]),
                                       np.mean(MMM_dict['melt_ext_Pfeffer'].data[
                                               slice_dict['MMM'][2] - 29: slice_dict['MMM'][2]])], index = [ 'HIST','1.5', '2', '4'])

df.to_csv(filepath + 'pct_ice_shelf_area_'+ region + '_abv_Pfeffer.csv')

# Create equivalent dataframe of area where runoff is simulated
df = pd.DataFrame(index = ['HIST', '1.5', '2', '4'])
for m in dict_list.keys():
    df[m] = pd.Series([np.mean(dict_list[m]['runoff_ext'].data[:29]),
                       np.mean(dict_list[m]['runoff_ext'].data[slice_dict[m][0] - 29: slice_dict[m][0]]),
                               np.mean(dict_list[m]['runoff_ext'].data[slice_dict[m][1] - 29: slice_dict[m][1]]),
                                       np.mean(dict_list[m]['runoff_ext'].data[
                                               slice_dict[m][2] - 29: slice_dict[m][2]])], index = ['HIST', '1.5', '2', '4'])
    df['MMM'] = pd.Series([np.mean(MMM_dict['runoff_ext'].data[:29]), np.mean(MMM_dict['runoff_ext'].data[slice_dict['MMM'][0] - 29: slice_dict['MMM'][0]]),
                               np.mean(MMM_dict['runoff_ext'].data[slice_dict['MMM'][1] - 29: slice_dict['MMM'][1]]),
                                       np.mean(MMM_dict['runoff_ext'].data[
                                               slice_dict['MMM'][2] - 29: slice_dict['MMM'][2]])], index = [ 'HIST','1.5', '2', '4'])

df.to_csv(filepath + 'pct_ice_shelf_area_'+ region + '_runoff.csv')

# Create dataframe of runoff duration in each sector of Antarctica
reg_masks = [AIS_mask, AP_mask, WA_mask, EA_mask]
for n, region in enumerate(['AIS', 'AP', 'WA', 'EA']):
    df = pd.DataFrame(index = ['HIST', '1.5', '2', '4'])
    for m in runoff_dict.keys():
        df[m] = pd.Series([np.nanmean(np.nanmean(runoff_dict[m]['RU_dur'].data[:29], axis = 0) * reg_masks[n]),
                           np.nanmean(np.nanmean(runoff_dict[m]['RU_dur'].data[slice_dict[m][0] - 29: slice_dict[m][0]], axis = 0) * reg_masks[n]),
                                   np.nanmean(np.nanmean(runoff_dict[m]['RU_dur'].data[slice_dict[m][1] - 29: slice_dict[m][1]], axis = 0) * reg_masks[n]),
                                           np.nanmean(np.nanmean(runoff_dict[m]['RU_dur'].data[slice_dict[m][2] - 29: slice_dict[m][2]], axis = 0) * reg_masks[n])], index = ['HIST', '1.5', '2', '4'])
        df['MMM'] = pd.Series([np.nanmean(np.nanmean(MMM_dict['RU_dur'].data[:29], axis = 0) * reg_masks[n]), np.nanmean(np.nanmean(MMM_dict['RU_dur'].data[slice_dict['MMM'][0] - 29: slice_dict['MMM'][0]], axis = 0) * reg_masks[n]),
                                   np.nanmean(np.nanmean(MMM_dict['RU_dur'].data[slice_dict['MMM'][1] - 29: slice_dict['MMM'][1]], axis = 0) * reg_masks[n]),
                                           np.nanmean(np.nanmean(MMM_dict['RU_dur'].data[
                                                   slice_dict['MMM'][2] - 29: slice_dict['MMM'][2]], axis = 0) * reg_masks[n])], index = [ 'HIST','1.5', '2', '4'])
    df.to_csv(filepath + 'ice_shelf_runoff_dur_'+ region + '.csv')


rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 20
rcParams['font.sans-serif'] = ['Segoe UI', 'Helvetica', 'Liberation sans', 'Tahoma', 'DejaVu Sans', 'Verdana']

# Plot scatter scenarios
def scatter_scen(abs_or_difs, reg):
    # Set up figure
    if abs_or_difs == 'abs':
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xticks(ticks=[2.5, 6.5, 10.5])
    elif abs_or_difs == 'difs':
        fig, ax = plt.subplots(figsize=(8,5 ))
        ax.set_xticks(ticks=[1,2,3])
    ax.spines['left'].set_linewidth(2)
    ax.spines['left'].set_color('dimgrey')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['bottom'].set_color('dimgrey')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticklabels(labels=['ME', 'RU', 'SMB'])
    ax.set_xlabel('Component', fontsize=24, labelpad=30, color='dimgrey')
    ax.set_ylabel('\nGt yr$^{-1}$', labelpad=50, rotation=0, fontsize=24, color='dimgrey')
    ax.tick_params(labelsize=24, labelcolor='dimgrey', width = 2, length = 5, color = 'dimgrey')
    plt.subplots_adjust(left=0.15, right=0.8)
    # Set up dictionaries
    scen_lookup = {0:0, 1.5: 1, 2.0: 2, 4.0: 3}
    if abs_or_difs == 'abs':
        col_dict = {0: '#F6BFB3', 1.5: '#F78F78', 2.0: '#FC441B', 4.0: '#A72609'}
        var_lookup = {'ME': 1, 'RU': 5, 'SMB': 9}
    elif abs_or_difs == 'difs':
        col_dict = {'ME': '#227BD5', 'RU': '#AB21D1', 'SMB': '#1E990A'}
        var_lookup = {'ME': 1, 'RU': 2, 'SMB': 3}
    mkr_dict = {'ACCESS1.3': 'x', 'CESM2': 'P', 'CNRM-CM6-1': '^', 'NorESM1-M': '*', 'MMM': 'o'}
    # Read data
    if abs_or_difs == 'abs':
        df_hist = pd.read_csv('D:\\Antarctic SMB\\Ice_shelf_abs_Gt_hist'+reg+'_deg.csv', index_col=0)#, skiprows=[2])
        df_hist = df_hist.transpose()
        df_hist = df_hist.assign(Scenario=0)  # header = ['ACCESS1.3','CESM2', 'NorESM1-M', 'CNRM-CM6-1']
        df_1p5 = pd.read_csv('D:\\Antarctic SMB\\Ice_shelf_'+ abs_or_difs+'_Gt_1p5'+reg+'_deg.csv', index_col=0)#, skiprows=[2])
        df_1p5 = df_1p5.transpose()
        df_1p5 = df_1p5.assign(Scenario=1.5)  # header = ['ACCESS1.3','CESM2', 'NorESM1-M', 'CNRM-CM6-1']
        df_2 = pd.read_csv('D:\\Antarctic SMB\\Ice_shelf_'+ abs_or_difs+'_Gt_2'+reg+'_deg.csv', index_col=0)#, skiprows=[2])
        df_2 = df_2.transpose()
        df_2 = df_2.assign(Scenario=2)
        df_4 = pd.read_csv('D:\\Antarctic SMB\\Ice_shelf_'+ abs_or_difs+'_Gt_4'+reg+'_deg.csv', index_col=0)#, skiprows=[2])
        df_4 = df_4.transpose()
        df_4 = df_4.assign(Scenario=4)
        cdf = pd.concat([df_hist, df_1p5, df_2, df_4])
        cdf['Model'] = cdf.index
        mdf = pd.melt(cdf, id_vars=['Scenario', 'Model'], var_name=['Var'])
        mins = mdf.groupby(['Var', 'Scenario']).min()
        maxs = mdf.groupby(['Var', 'Scenario']).max()
        # Plot data in categories, with markers to indicate model and colours to indicate scenario
        for i in range(60):
            data = mdf.iloc[i].value
            x_val = var_lookup[mdf.iloc[i].Var] + scen_lookup[mdf.iloc[i].Scenario]
            if mdf.iloc[i].Model == 'MMM':
                plt.scatter(x_val, data, marker=mkr_dict[mdf.iloc[i].Model], s=350, c=col_dict[mdf.iloc[i].Scenario], edgecolors = 'k', linewidths=3, zorder = 4)
            else:
                plt.scatter(x_val, data, marker = mkr_dict[mdf.iloc[i].Model], s =350, c = col_dict[mdf.iloc[i].Scenario], zorder = 4)
        for x, i in zip([1,5,9],[0,4,8]):
            for n, col in enumerate([0, 1.5, 2.0, 4.0]):
                ax.vlines(x=x+n, ymin=mins.iloc[i+n].value, ymax=maxs.iloc[i+n].value, colors = col_dict[col], linestyle='--', linewidth=3, zorder = 1)
        # Draw lines
        MMMs = mdf.loc[mdf['Model'] == 'MMM']
        ax.plot((1, 2, 3, 4), (MMMs.iloc[0].value, MMMs.iloc[1].value, MMMs.iloc[2].value, MMMs.iloc[3].value), color='dimgrey', linewidth=2, zorder=3)
        ax.plot((5, 6, 7, 8), (MMMs.iloc[4].value, MMMs.iloc[5].value, MMMs.iloc[6].value, MMMs.iloc[7].value), color='dimgrey', linewidth=2, zorder=3)
        ax.plot((9, 10, 11, 12), (MMMs.iloc[8].value, MMMs.iloc[9].value, MMMs.iloc[10].value, MMMs.iloc[11].value), color='dimgrey', linewidth=2, zorder=3)
        # Create legends
        mkrs = [matplotlib.patches.Patch(facecolor=col_dict[0], edgecolor=None, label='Historical'),
                matplotlib.patches.Patch(facecolor=col_dict[1.5], edgecolor=None, label='1.5$^{\circ}$C'),
                matplotlib.patches.Patch(facecolor=col_dict[2.0], edgecolor=None, label='2.0$^{\circ}$C'),
                matplotlib.patches.Patch(facecolor=col_dict[4.0], edgecolor=None, label='4.0$^{\circ}$C')]
        lgd1 = plt.legend(handles=mkrs, bbox_to_anchor=(1.0, 0.5), loc=2, fontsize=20, title='Scenario')
        ax.add_artist(lgd1)
        frame = lgd1.get_frame()
        frame.set_facecolor('white')
        for ln in lgd1.get_texts():
            plt.setp(ln, color='dimgrey', fontsize=18)
        frame.set_linewidth(2.0)
        frame.set_edgecolor('dimgrey')
        ln = lgd1.get_title()
        plt.setp(ln, color='#222222', fontsize=20)
        cols = [Line2D([0], [0], marker='x', color='w', label='ACCESS1.3', markeredgecolor='#222222',
                       markerfacecolor='#222222', markersize=15),
                Line2D([0], [0], marker='P', color='w', label='CESM2', markerfacecolor='#222222', markersize=15),
                Line2D([0], [0], marker='^', color='w', label='CNRM-CM6-1', markerfacecolor='#222222', markersize=15),
                Line2D([0], [0], marker='*', color='w', label='NorESM1-M', markerfacecolor='#222222', markersize=15),
                Line2D([0], [0], marker='o', color='w', label='MMM', markeredgecolor='#222222', markerfacecolor='w',
                       markersize=15)]
        lgd2 = plt.legend(handles=cols, bbox_to_anchor=(1.0, 1.), loc=2, fontsize=20, title="Model")
        ax.add_artist(lgd2)
        frame = lgd2.get_frame()
        frame.set_facecolor('white')
        for ln in lgd2.get_texts():
            plt.setp(ln, color='dimgrey', fontsize=18)
        ln = lgd2.get_title()
        plt.setp(ln, color='#222222', fontsize=20)
        frame.set_linewidth(2.0)
        frame.set_edgecolor('dimgrey')
    elif abs_or_difs == 'difs':
        cdf = pd.read_csv('D:\\Antarctic SMB\\Ice_shelf_difs_Gt_1p5_to_4_deg.csv', index_col = 0)
        cdf = cdf.transpose()
        cdf['Model'] = cdf.index
        mdf = pd.melt(cdf, id_vars=[ 'Model'], var_name=['Var'])
        plt.axhline(y = 0, linewidth = 2, color = 'dimgrey', linestyle = '--', zorder = 2)
        for i in range(15):
            data = mdf.iloc[i].value
            x_val = var_lookup[mdf.iloc[i].Var]
            if mdf.iloc[i].Model == 'MMM':
                plt.scatter(x_val, data, marker=mkr_dict[mdf.iloc[i].Model], s=550, c=col_dict[mdf.iloc[i].Var], edgecolors = 'k', linewidths=3, zorder = 4)
            else:
                plt.scatter(x_val, data, marker = mkr_dict[mdf.iloc[i].Model], s =550, c = col_dict[mdf.iloc[i].Var], zorder = 4)
        for n, col in enumerate(['ME', 'RU', 'SMB']):
            ax.vlines(x=var_lookup[col], ymin=mdf.loc[mdf['Var'] == col].min().value, ymax=mdf.loc[mdf['Var'] == col].max().value, colors = col_dict[col], linestyle='--', linewidth=3, zorder = 1)
        plt.subplots_adjust(left = 0.3, right = 0.98, )
    plt.savefig(filepath + 'Model_scatter_scenarios_components_'+abs_or_difs+'.png')
    plt.savefig('C:\\Users\\Ella\\OneDrive - University of Reading\\Antarctic SMB\\Model_scatter_scenarios_components_'+abs_or_difs+'_'+ reg + '.png')
    plt.savefig('C:\\Users\\Ella\\OneDrive - University of Reading\\Antarctic SMB\\Model_scatter_scenarios_components_'+abs_or_difs+'_'+ reg + '.eps')
    plt.show()

#scatter_scen('abs','EA')

# ==================================================================================================================== #

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
