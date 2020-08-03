## Load in data files
# Precipitation, evaporation, sublimation (?), runoff, snowmelt?, land-sea mask

## Do some processing to unify units and wrangle data if necessary

## Mask non-land points

## Calculate SMB
# SMB = precip - evap - sublim - runoff



## Validation? - Do I need to do this or will Chris's script do it?
# After Mottram et al. (2020): SMB values compared with observations in three steps:
# 1. modelled SMB interpolated onto observation location
# 2. interpolated SMB values from the same grid cell averaged (same for obs)
# 3. produces 923 comparison pairs (923 grid cells with averaged obs and model values)
#