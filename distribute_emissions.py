import geopandas
import numpy as np
import pandas as pd
import xarray as xr
import json
import time
import matplotlib.pyplot as plt

from helper_functions import *

# ----------------------------------------
# Description:
# This script takes mercury emission magnitudes described for 17 world 
# regions in Streets et al. (2019) and distributes them using intra-region 
# emission patterns from the WHET inventory (Zhang et al., 2016)
# ----------------------------------------

# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------
output_path = './output/'
lat_res = 1 #0.5 # latitude grid resolution in degrees
lon_res = 1 #0.5 # longitude grid resolution in degrees
lats = np.arange(start=   90-lat_res/2, stop=-90, step= -lat_res)
lons = np.arange(start= -180+lon_res/2, stop=180, step=  lon_res)

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
sec_per_year = 60*60*24*365.25
kg_to_Mg = 1e-3

# ------------------------------------------------------------------------
# Make 17 world region masks
# ------------------------------------------------------------------------
# initialize empty dataset with correct resolution
data = np.zeros((len(lats), len(lons)))
ds = make_ds('empty_ds', data=data, lat=lats, lon=lons, attrs={})

# Get shapefiles of all countries from natural earth data
# http://www.naturalearthdata.com/downloads/10m-cultural-vectors/ne_10m_admin_0_countries/
countries = geopandas.read_file('./data/ne_10m_admin_0_countries/ne_10m_admin_0_countries.shp')
country_ids = {k: i for i, k in enumerate(countries.ADMIN)}
shapes = zip(countries.geometry, range(len(countries)))

# make DataArray (`countries`) with grid cells labeled with the corresponding 
# country/territory it is part of
ds['countries'] = rasterize(shapes, ds.coords, longitude='lon', latitude='lat')

# load territories/regions corresponding to each of the 17 world regions in 
# Streets et al. (2019). Dictionary structure is {'region_name':[list of countries in region]}
with open('./data/17_region_dict.json', 'r') as openfile:
    region_dict = json.load(openfile)

# construct a mask for each of the 17 world regions
for region in region_dict.keys():
    indices = [country_ids[country] for country in region_dict[region]]
    ds[region] = ds['countries'].isin(indices)
    ds[region] = ds[region].where(ds[region] != False, drop=False)
    ds[region] = ds[region].assign_attrs({'Region Name': region,
                                          'territories included': region_dict[region],
                                          '_FillValue': np.nan, 'units': 'mask'})

ds['lat'] = ds['lat'].assign_attrs({'units':'degrees_north','long_name':'latitude','axis':'Y'})
ds['lon'] = ds['lon'].assign_attrs({'units':'degrees_east','long_name':'longitude','axis':'X'}) 
ds = ds.drop(['countries','empty_ds'])

# now save masks to ./output
mask_compression_dict = {}
masks_output = ds.copy()
for v in masks_output.data_vars:
    mask_compression_dict[v] = {"zlib":True, "complevel":5}

ds = ds.assign_attrs({'Description':f'17 world region masks for {lat_res} x {lon_res} degree future emission scenarios',
                      'history':f'Created {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}',
                      'source':'Biogeochemistry of Global Contaminants Group',})

ds.to_netcdf('./output/17_region_masks.nc', engine="h5netcdf", encoding=mask_compression_dict)

# re-order longitude coordinates from [-180,180] --> [0,360]
# for alignment with WHET
mask_ds = reorder_lon_coords(ds)

# ------------------------------------------------------------------------
# Read Streets et al. (2019) emission magnitudes
# ------------------------------------------------------------------------
regional_total_emissions     = pd.read_csv('./data/Seventeen_World_Region_Total_Hg_Emissions_2000-2015.csv')
regional_Hg0_emissions = pd.read_csv('./data/Seventeen_World_Region_Hg0_Emissions_2000-2015.csv')
# calculate f_Hg0 for each region
regional_f_Hg0 = regional_Hg0_emissions.iloc[:,1:]/regional_total_emissions.iloc[:,1:]
regional_f_Hg0['World region'] = regional_total_emissions['World region']
regional_f_Hg0[regional_total_emissions.columns]

# ------------------------------------------------------------------------
# Read WHET inventory emission patterns (Zhang et al., 2016)
# ------------------------------------------------------------------------
WHET = xr.open_mfdataset('./data/WHET_compiled_1x1_final.nc')
area = do_grid(lat_resolution=1, lon_resolution=1)
area = make_ds(var_name='area', data=area, lat=WHET.lat.values, lon=WHET.lon.values, attrs={'units':'m2','long_name':'area',})
WHET = xr.merge((WHET, area))

#ds_out = ds_out.assign_coords({'area':WHET['area']})
WHET[['area']].to_netcdf('./output/area_1x1.nc')

first=True
for year in [2000, 2010, 2011, 2012, 2013, 2014, 2015]:
    
    # initialize temporary DataArray
    ds_tmp = WHET[['Hg0','Hg2','HgP']].isel(year=0).drop('year').copy()*0.
    
    for region in region_dict.keys():        
        # --- Hg0 --- 
        # get new emission magnitudes for (region, time)
        total_new = subset_df(df=regional_total_emissions, col_name='World region', value=region)[str(year)].values.item()
        Hg0_new   = subset_df(df=regional_Hg0_emissions, col_name='World region', value=region)[str(year)].values.item()
        Hg2_new   = total_new - Hg0_new
        # get old emission magnitudes for (region, time)
        Hg0_old = get_masked_emission_magnitude(emission_field = WHET['Hg0'].sel(year=year, method='nearest').drop('year'), mask = mask_ds[region], area = WHET['area'])
        sf_Hg0 = Hg0_new/Hg0_old # get scale factor
        masked_emission = mask_for_aggregation(emission_field=WHET['Hg0'].sel(year=year, method='nearest').drop('year'), mask=mask_ds[region])
        ds_tmp['Hg0'] += masked_emission*sf_Hg0
        
        # --- Hg2, HgP --- 
        Hg2_old = get_masked_emission_magnitude(emission_field = WHET['Hg2'].sel(year=year, method='nearest'), mask = mask_ds[region], area = WHET['area'])
        HgP_old = get_masked_emission_magnitude(emission_field = WHET['HgP'].sel(year=year, method='nearest'), mask = mask_ds[region], area = WHET['area'])
        f_HgP_old = HgP_old/(Hg2_old+HgP_old) # get particulate fraction of oxidized Hg emissions from WHET inventory
        
        sf_Hg2_all = Hg2_new/(Hg2_old+HgP_old)
        sf_Hg2 = sf_Hg2_all*(1-f_HgP_old)
        sf_HgP = sf_Hg2_all*(f_HgP_old)
        
        masked_emission = (mask_for_aggregation(emission_field=WHET['Hg2'].sel(year=year, method='nearest').drop('year'), mask=mask_ds[region]) + 
                           mask_for_aggregation(emission_field=WHET['HgP'].sel(year=year, method='nearest').drop('year'), mask=mask_ds[region]) )
        
        ds_tmp['Hg2'] += masked_emission*sf_Hg2
        ds_tmp['HgP'] += masked_emission*sf_HgP   
        
    ds_tmp = ds_tmp.expand_dims({'time':[year]})
    
    if first==True:
        ds_out = ds_tmp.copy()
    else:
        ds_out = xr.concat((ds_out, ds_tmp), dim='time')
    
    first=False

# now fill emissions between 2000 and 2010 by linear interpolation
for year in np.arange(2001, 2010):
    df_tmp = xr.Dataset()
    for species in ['Hg0','Hg2','HgP']:
        span = (2010-2000)
        f_lo = 1-((year-2000)/span)
        f_hi = 1-(np.abs(year-2010)/span)
        assert (f_lo + f_hi) == 1       
        
        df_tmp[species] = (f_lo*ds_out[species].sel(time=2000))+(f_hi*ds_out[species].sel(time=2010))
        
    df_tmp = df_tmp.expand_dims({'time':[year]})
    ds_out = xr.merge((ds_out, df_tmp))
    
datetime_list = []
for year in ds_out.time.values:
    year_datetime = get_days_since_epoch(year=year)
    datetime_list.append(year_datetime)
    
ds_out['time'] = datetime_list
    
# assign time attributes
ds_out['time'] = ds_out['time'].assign_attrs({'units':'days since 1970-01-01 00:00:00',
                                              'long_name':'time', 'calendar':'standard',
                                              'delta_t':'0000-00-01 00:00:00',
                                              'begin_date':'19700101',
                                              'begin_time':'000000',
                                              'axis':'T'})

ds_out = ds_out.assign_attrs({'title':f'2000 - 2015 Global Anthropogenic Mercury Emissions',
                             'history':f'Created {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}',
                             'source':'Biogeochemistry of Global Contaminants Group (BMG)',
                             'source script': 'distribute_emissions.py',
                             'full_reference':'D.G. Streets, H.M. Horowitz, Z. Lu, L. Levin, C.P. Thackray, E.M. Sunderland. 2019. Global and regional trends in mercury emissions and concentrations, 2010-2015. Atmospheric Environment. 201: 417-427.',
                              'description':''})

for v in ['lat','lon','Hg0','Hg2','HgP']:
    ds_out[v] = ds_out[v].assign_attrs(WHET[v].attrs)
    
ds_out['lat'] = ds_out['lat'].assign_attrs({'units':'degrees_north','long_name':'latitude','axis':'Y'})
ds_out['lon'] = ds_out['lon'].assign_attrs({'units':'degrees_east','long_name':'longitude','axis':'X'}) 
ds_out['Hg0'] = ds_out['Hg0'].assign_attrs({'units':'kg m-2 s-1', 'long_name':'gaseous elemental mercury'})
ds_out['Hg2'] = ds_out['Hg2'].assign_attrs({'units':'kg m-2 s-1', 'long_name':'gaseous oxidized mercury'})
ds_out['HgP'] = ds_out['HgP'].assign_attrs({'units':'kg m-2 s-1', 'long_name':'particulate mercury'})

compression_dict = {}
for v in ds_out.data_vars: # compression args for data vars
    compression_dict[v] = {"zlib":True, "dtype":"float32", "complevel":5}

for v in ds_out.coords: # compression args for coordinates
    compression_dict[v] = {"zlib":True, "complevel":5}

# save output
ds_out.to_netcdf('./output/Streets_2019_Anthro_Hg_Emissions.nc',
                 engine="h5netcdf", encoding=compression_dict)