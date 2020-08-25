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


# Create domain constraints.
Antarctic_peninsula = iris.Constraint(longitude=lambda v: -80 <= v <= -55,
                                latitude=lambda v: -75 <= v <= -55)
East_Antarctica = iris.Constraint(longitude=lambda v: -40 <= v <= 180,# & longitude=lambda v: 320 <= v <=360,
                                latitude=lambda v: -90 <= v <= -60)
West_Antarctica = iris.Constraint(longitude=lambda v: -179 <= v <= -40,
                                  latitude=lambda v: -90 <= v <= -72)


# Convert units
for v in [pr, evap, mrro, mrros, snm]:
    v.convert_units('kg/m2/d')
    iris.coord_categorisation.add_year(v, 'time', name='year')

pr_annual_tot = pr.aggregated_by(['year'],iris.analysis.SUM)
evap_annual_tot = evap.aggregated_by(['year'], iris.analysis.SUM)
mrro_annual_tot = mrro.aggregated_by(['year'], iris.analysis.SUM)
mrros_annual_tot = mrros.aggregated_by(['year'], iris.analysis.SUM)
snm_annual_tot = snm.aggregated_by(['year'], iris.analysis.SUM)


def load_model_data(var_name, region_name, model_name, scenarios):
    #  Load data for each scenario.
historical = iris.load_cube(filepath + var_name + '_ccam_'+ model_name + '_ant-44i_50km_day.historical.nc', region_name)
iris.coord_categorisation.add_season(historical, 'time', name='clim_season')
iris.coord_categorisation.add_season_year(historical, 'time', name='season_year')
iris.coord_categorisation.add_year(historical, 'time', name='year')
historical.convert_units('kg/m2/d')
historical_annual_seasonal_mean = historical.aggregated_by(['clim_season', 'season_year'],iris.analysis.MEAN)
historical_annual_tot = historical.aggregated_by(['year'],iris.analysis.SUM) # time series of n=36 yrs, annual tot
    # do this just once and re-use. This method requires bounds on lat/lon
    # coords, so let's add some in sensible locations using the "guess_bounds"
    # method.
    historical_annual_seasonal_mean.coord('latitude').guess_bounds()
    historical_annual_seasonal_mean.coord('longitude').guess_bounds()
    historical_grid_areas = iris.analysis.cartography.area_weights(historical_annual_seasonal_mean)
    # Perform the area-weighted mean for each of the datasets using the
    # computed grid-box areas.
    historical_mean = historical_annual_seasonal_mean.collapsed(['latitude', 'longitude'],
                                                                iris.analysis.MEAN,
                                                                weights=historical_grid_areas)
    #historical_annual_mean = iris.analysis.maths.multiply(historical_annual_mean, 31579200.)  # kg/m2/yr
    historical_annual_tot.coord('latitude').guess_bounds()
    historical_annual_tot.coord('longitude').guess_bounds()
    historical_grid_areas = iris.analysis.cartography.area_weights(historical_annual_mean)
    historical_annual_mean = historical_annual_mean.collapsed(['latitude', 'longitude'],
                                                   iris.analysis.MEAN,
                                                   weights=historical_grid_areas)
    if historical_annual_mean.units == 'kg/m2/s':
        historical_annual_mean.convert_units('kg/m2/yr')
        #historical_annual_mean = iris.analysis.maths.multiply(historical_annual_mean, 31579200.) # multiply by number of seconds per year to get units of kg/m2/yr == mm w.e. / yr
    # *2000 (number of gridboxes)
    series_labs = zip(historical_annual_seasonal_mean.coord('clim_season').points,
                      historical_annual_seasonal_mean.coord('season_year').points)
    # update labels of x axis to be seasonal (i.e. construct strings from series_labs[0]+series_labs[1])
    if scenarios == 'yes' or scenarios == 'y' or scenarios == 1:
        rcp45 = iris.load_cube(var_name + '_ccam_' + model_name + '_44i_50km_day.2000-2100.nc', region_name)
        rcp85 = iris.load_cube(var_name + '_ccam_' + model_name + '_44i_50km_day.2000-2100.nc', region_name)
        rcp45_grid_areas = iris.analysis.cartography.area_weights(historical) # on the same grid, so re-use
        rcp85_grid_areas = iris.analysis.cartography.area_weights(historical) # on the same grid, so re-use
        rcp45_mean = rcp45.collapsed(['latitude', 'longitude'],
                               iris.analysis.MEAN,
                               weights=rcp45_grid_areas)
        rcp85_mean = rcp85.collapsed(['latitude', 'longitude'],
                                 iris.analysis.MEAN,
                                 weights=rcp85_grid_areas)
    if scenarios=='y' or scenarios=='yes':
        return historical_annual_mean, historical_annual_seasonal_mean, rcp45_mean, rcp85_mean
    else:
        return historical_annual_mean, historical_annual_seasonal_mean

for v in ['snw', 'hfls', 'hfss',]: #'pr', 'evspsbl', 'mrros', 'prsn'
    for r in [Antarctic_peninsula, West_Antarctica, East_Antarctica]:
        fig, ax = plt.subplots(1,1)
        #ax = ax.flatten()
        gfdl_an, gfdl_seas = load_model_data(var_name=v, model_name='gfdl-esm2m', scenarios='n', region_name=r)
        era_an, era_seas = load_model_data(var_name=v, model_name='eraint', scenarios='n', region_name=r)
        access_an, access_seas = load_model_data(var_name=v, model_name='access1-0', scenarios='n', region_name=r)
        an_model_list = [gfdl_an.data,  access_an.data]
        labels_list = ['GFDL-esm2m', 'ACCESS1-0']
        MMM = np.mean(an_model_list, axis = 0)
        max_an = np.max(an_model_list, axis=0)
        min_an = np.min(an_model_list, axis = 0)
        p95 = np.percentile(an_model_list, 95, axis = 0)
        p5 = np.percentile(an_model_list, 5, axis=0)
        plt.plot(gfdl_an.coord('year').points, MMM, lw = 3, color = 'k', label = 'multi-model mean' )
        plt.fill_between(gfdl_an.coord('year').points, p5, p95, color = 'lightgrey', zorder = 1)
        plt.ylabel('Annual mean ' + v + ' (kg/m2/yr)', rotation=90)
        plt.plot(era_an.coord('year').points, era_an.data, label='historical ERA-Interim', lw=1, color='royalblue')
        #ax[1].plot(np.arange(1980, 2016, 0.25), era_seas.data, label='historical ERA-Interim', lw=1, color='royalblue')
        for m, l in zip(an_model_list, labels_list):
            plt.plot(gfdl_an.coord('year').points, m, label='historical ' + l , lw=0.5, color='dimgrey', linestyle='--', zorder = 10)
            #ax[1].plot(np.arange(1960, 2000.25, 0.25), m, label='historical ' + l, lw=1.5)
        # Plot the datasets
        #qplt.plot(rcp45_mean, label='RCP 4.5 scenario' + model_name, lw=1.5, color='blue')
        #qplt.plot(rcp85_mean, label='RCP 8.5 scenario' + model_name, lw=1.5, color='red')
        plt.legend()
        if r == Antarctic_peninsula:
            plt.title('Annual mean ' + v + ' ' + 'AP' )
            #ax[1].set_title('Seasonal mean ' + v + ' ' + 'AP' )
            plt.savefig(filepath + v + '_AP_historical_rough.png')
        elif r == East_Antarctica:
            plt.title('Annual mean ' + v + ' ' + 'EA' )
            #ax[1].set_title('Seasonal mean ' + v + ' ' + 'EA' )
            plt.savefig(filepath + v + '_EA_historical_rough.png')
        elif r == West_Antarctica:
            plt.title('Annual mean ' + v + ' ' + 'WA')
            #ax[1].set_title('Seasonal mean ' + v + ' ' + 'WA')
            plt.savefig(filepath + v + '_WA_historical_rough.png')
    #plt.show()


    # Draw a horizontal line showing the pre-industrial mean
    plt.axhline(y=pre_industrial_mean.data, color='gray', linestyle='dashed',
                label='pre-industrial', lw=1.5)

    # Constrain the period 1860-1999 and extract the observed data from a1b
    constraint = iris.Constraint(time=lambda
                                 cell: 1860 <= cell.point.year <= 1999)
    observed = a1b_mean.extract(constraint)
    # Assert that this data set is the same as the e1 scenario:
    # they share data up to the 1999 cut off.
    assert np.all(np.isclose(observed.data,
                             e1_mean.extract(constraint).data))

    # Plot the observed data
    qplt.plot(observed, label='observed', color='black', lw=1.5)

    # Add a legend and title
    plt.legend(loc="upper left")
    plt.title('North American mean air temperature', fontsize=18)

    plt.xlabel('Time / year')

    plt.grid()

    iplt.show()


if __name__ == '__main__':
    main()

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
