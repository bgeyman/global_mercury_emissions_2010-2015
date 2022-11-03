import numpy as np
import pandas as pd
import xarray as xr
from helper_functions import *

def check_gridded_emission_magnitudes():
    '''Check that gridded emission magnitudes match emissions from original paper.'''
    area = xr.open_mfdataset('./output/area_1x1.nc')
    ds = xr.open_mfdataset('./output/Streets_2019_Anthro_Hg_Emissions.nc')
    input_table = pd.read_csv('./data/Seventeen_World_Region_Total_Hg_Emissions_2000-2015.csv')
    input_table = input_table[input_table['World region']=='Global Total']
    for year in [2000, 2010, 2011, 2012, 2013, 2014, 2015]:
        input_value = input_table[str(year)].values.item()
        comparison_value = 0
        for species in ['Hg0','Hg2','HgP']:
            comparison_value+=get_emission_sum(ds.isel(time=(ds.time.dt.year == year))[species], area['area'])
        # assert gridded emission magnitude within 0.1% of table value
        assert np.abs((input_value - comparison_value)/input_value) < 1e-3 
    
    return print('PASSED EMISSION CHECK')

def get_emission_totals():
    area = xr.open_mfdataset('./output/area_1x1.nc')
    ds = xr.open_mfdataset('./output/Streets_2019_Anthro_Hg_Emissions.nc')
    E = []
    for year in ds.time.dt.year.values:
        E_tmp = 0
        for species in ['Hg0','Hg2','HgP']:
            E_tmp+=get_emission_sum(ds.isel(time=(ds.time.dt.year == year))[species], area['area'])
        E.append(E_tmp)
    return E

check_gridded_emission_magnitudes()
E = get_emission_totals()