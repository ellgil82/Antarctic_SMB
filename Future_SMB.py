import iris
import numpy as np
import matplotlib.pyplot as plt
import PyMC3 as pm

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
