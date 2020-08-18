#!/bin/bash

vars="pr" # evspsbl mrro tas prsn" #snm tas mrros snw prsn ts

models="access1-0 gfdl-esm2m" # eraint" #hadgem2 noresm1-m

simulations="historical" #rcp85 rcp45"

#cd /data/mac/ellgil82/AntSMB/
for m in $models
	for s in $simulations
		for v in $vars
		do
			echo downloading "$v" from server 
			wget https://data-cbr.csiro.au/thredds/fileServer/catch_all/oa-ccam/cordex_antarctica/ccam_"$m$"_ant-44i_50km/"$s"/"$v"_ccam_"$m"_ant-44i_50km_day.2010-2015.nc
			wget https://data-cbr.csiro.au/thredds/fileServer/catch_all/oa-ccam/cordex_antarctica/ccam_"$m$"_ant-44i_50km/"$s"/"$v"_ccam_"$m"_ant-44i_50km_day.2000-2009.nc
			wget https://data-cbr.csiro.au/thredds/fileServer/catch_all/oa-ccam/cordex_antarctica/ccam_"$m$"_ant-44i_50km/"$s"/"$v"_ccam_"$m"_ant-44i_50km_day.1990-1999.nc
			wget https://data-cbr.csiro.au/thredds/fileServer/catch_all/oa-ccam/cordex_antarctica/ccam_"$m$"_ant-44i_50km/"$s"/"$v"_ccam_"$m"_ant-44i_50km_day.1980-1989.nc
			infile="$v"_ccam_"$m"_ant-44i_50km_day.????-????.nc
			outfile="$v"_ccam_"$m"_ant-44i_50km_day.1980-2015.nc
			cdo mergetime $infile $outfile
		done

