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
region = 'AIS'

# Load orography and coastlines
orog_full = iris.load_cube(filepath + 'orog.nc')
lsm_full = iris.load_cube(filepath + 'sftlf.nc')

# Load relevant SMB files
if region == 'AIS':
    pr = iris.load_cube(filepath + 'pr_ccam_eraint_ant-44i_50km_day.historical.nc')
    evap = iris.load_cube(filepath + 'evspsbl_ccam_eraint_ant-44i_50km_day.historical.nc')
    mrro = iris.load_cube(filepath + 'mrro_ccam_eraint_ant-44i_50km_day.historical.nc')
    #mrros = iris.load_cube(filepath + 'mrros_ccam_eraint_ant-44i_50km_day.historical.nc', region)
    #Load orography and coastlines
    orog = iris.load_cube(filepath + 'orog.nc')
    lsm = iris.load_cube(filepath + 'sftlf.nc')
    grid_area = iris.load_cube(filepath + 'grid_area.nc')
else:
    pr = iris.load_cube(filepath + 'pr_ccam_eraint_ant-44i_50km_day.historical.nc', region)
    evap = iris.load_cube(filepath + 'evspsbl_ccam_eraint_ant-44i_50km_day.historical.nc', region)
    mrro = iris.load_cube(filepath + 'mrro_ccam_eraint_ant-44i_50km_day.historical.nc', region)
    #mrros = iris.load_cube(filepath + 'mrros_ccam_eraint_ant-44i_50km_day.historical.nc', region)
    #Load orography and coastlines
    orog = iris.load_cube(filepath + 'orog.nc', region)
    lsm = iris.load_cube(filepath + 'sftlf.nc', region)
    grid_area = iris.load_cube(filepath + 'grid_area.nc', region)

# Convert units
for v in [pr, evap, mrro, mrro]:
    v.convert_units('kg/m2/d')
    iris.coord_categorisation.add_year(v, 'time', name='year')

pr_annual_tot = pr.aggregated_by(['year'],iris.analysis.SUM)
evap_annual_tot = evap.aggregated_by(['year'], iris.analysis.SUM)
mrro_annual_tot = mrro.aggregated_by(['year'], iris.analysis.SUM)
mrros_annual_tot = mrros.aggregated_by(['year'], iris.analysis.SUM)

grid_area_masked = grid_area * lsm
pr_an_Gt = (pr_annual_tot.data.mean(axis= 0)*grid_area_masked.data)/1e14
tot_pr = pr_an_Gt.sum()
evap_an_tot = (evap_annual_tot.data.mean(axis= 0)*grid_area_masked.data)/1e14

## Calculate SMB
# SMB = precip - evap - sublim - runoff
SMB = pr - evap - mrro # no sublimation file
SMB_annual_tot = pr_annual_tot - evap_annual_tot - mrro_annual_tot
SMB_an_Gt = (SMB_annual_tot.data.mean(axis = 0) * grid_area_masked.data)/10e12


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
    squished_cmap = shiftedColorMap(cmap=matplotlib.cm.coolwarm_r, min_val=-200., max_val=300., name='squished_cmap', var=mean_data,
                                    start=0.3, stop=0.7)
    f = ax.pcolormesh(gridlons, gridlats, np.ma.masked_where(lsm.data == 0, mean_data), transform = ccrs.PlateCarree(),
                    cmap=squished_cmap, vmin = -170, vmax = 250)
    ax.contour(lsm_full.coord('longitude').points, lsm_full.coord('latitude').points, lsm_full.data>0, levels = [0], lw = 2, color = 'k', transform = ccrs.PlateCarree())
    plt.colorbar(f)
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

def reproject(latitude, longitude):
    """Returns the x & y coordinates in meters using a sinusoidal projection"""
    from math import pi, cos, radians
    earth_radius = 6371009 # in meters
    lat_dist = pi * earth_radius / 180.0
    y = [lat * lat_dist for lat in latitude]
    x = [long * lat_dist * cos(radians(lat))
                for lat, long in zip(latitude, longitude)]
    return x, y

def area_of_polygon(x, y):
    """Calculates the area of an arbitrary polygon given its vertices"""
    area = 0.0
    for i in range(-1, len(x)-1):
        area += x[i] * (y[i+1] - y[i-1])
    return abs(area) / 2.0

def polygon_area(lats, lons, algorithm = 0, radius = 6378137):
    """
    Computes area of spherical polygon, assuming spherical Earth.
    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.
    lats and lons are in degrees.
    """
    from numpy import arctan2, cos, sin, sqrt, pi, power, append, diff, deg2rad
    lats = np.deg2rad(lats)
    lons = np.deg2rad(lons)
    # Line integral based on Green's Theorem, assumes spherical Earth
    #close polygon
    if lats[0]!=lats[-1]:
        lats = append(lats, lats[0])
        lons = append(lons, lons[0])
    #colatitudes relative to (0,0)
    a = sin(lats/2)**2 + cos(lats)* sin(lons/2)**2
    colat = 2*arctan2( sqrt(a), sqrt(1-a) )
    #azimuths relative to (0,0)
    az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2*pi)
   # Calculate diffs
    # daz = diff(az) % (2*pi)
    daz = diff(az)
    daz = (daz + pi) % (2 * pi) - pi
    deltas=diff(colat)/2
    colat=colat[0:-1]+deltas
    # Perform integral
    integrands = (1-cos(colat)) * daz
    # Integrate
    area = abs(sum(integrands))/(4*pi)
    area = min(area,1-area)
    if radius is not None: #return in units of radius
        return area * 4*pi*radius**2
    else: #return in ratio of sphere total area
        return area

# Find area of Antarctica
lsm.coord('latitude').guess_bounds()
lsm.coord('longitude').guess_bounds()
cell_areas = iris.analysis.cartography.area_weights(lsm)


co = {"type": "Polygon", "coordinates": [
    [(max(lsm.coord('longitude').points), max(lsm.coord('latitude').points)),
     (max(lsm.coord('longitude').points), min(lsm.coord('latitude').points)),
     (min(lsm.coord('longitude').points), min(lsm.coord('latitude').points)),
     (min(lsm.coord('longitude').points), max(lsm.coord('latitude').points))]]}
lon, lat = zip(*co['coordinates'][0])
from pyproj import Proj
lat_1 = min(lsm.coord('latitude').points)
lat_2 = max(lsm.coord('latitude').points)
lat_0= max(lsm.coord('latitude').points)-(max(lsm.coord('latitude').points)-min(lsm.coord('latitude').points))
lon_0 =max(lsm.coord('longitude').points)-(max(lsm.coord('longitude').points)-min(lsm.coord('longitude').points))

pa = Proj("+proj=aea +lat_1=lat_1 +lat_2=lat_2 +lat_0=lat_0 +lon_0=lon_0")
pa = Proj("+proj=aea +lat_1=-90 +lat_2=-55 +lat_0=-74 +lon_0=lon_0-79.75")
x, y = pa(lon, lat)
cop = {"type": "Polygon", "coordinates": [zip(x, y)]}
from shapely.geometry import shape
shape(cop).area  # 268952044107.43506




## Calculate SMB
# SMB = precip - evap - sublim - runoff

os.chdir('AntSMB')
pr_list = iris.load('pr_ccam*-1960-1999.nc') # Load all available historical simulations
era_int = iris.load_cube('pr_ccam_eraint_44i_50km_day.1980-2015.nc') # Full ERA historical series


## Validation? - Do I need to do this or will Chris's script do it?
# After Mottram et al. (2020): SMB values compared with observations in three steps:
# 1. modelled SMB interpolated onto observation location
# 2. interpolated SMB values from the same grid cell averaged (same for obs)
# 3. produces 923 comparison pairs (923 grid cells with averaged obs and model values)
#