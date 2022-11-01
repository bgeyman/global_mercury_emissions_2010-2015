import numpy as np
import pandas as pd
import xarray as xr
from affine import Affine
import geopandas
from rasterio import features

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
sec_per_year = 60*60*24*365.25
kg_to_Mg = 1e-3

# ---------------------------------------------------------------
# Functions
# ---------------------------------------------------------------
def make_ds(var_name, data, lat, lon, attrs={}):
    ''' Initialize an xarray dataset with coordinates (lat, lon)'''
    ds = xr.Dataset({var_name: xr.DataArray(
                     data   = data,
                     dims   = ['lat','lon'],
                     coords = {'lat':lat, 'lon':lon},
                     attrs  = attrs),
                    })
    return ds

def transform_from_latlon(lat, lon):
    ## https://gist.github.com/shoyer/0eb96fa8ab683ef078eb
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def rasterize(shapes, coords, latitude='latitude', longitude='longitude',
              fill=np.nan, **kwargs):
    ## https://gist.github.com/shoyer/0eb96fa8ab683ef078eb
    '''Rasterize a list of (geometry, fill_value) tuples onto the given
    xarray coordinates. This only works for 1d latitude and longitude
    arrays.
    '''
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))

def do_grid (lat_resolution=1, lon_resolution=1):
    '''Calculate the area of each grid cell for a user-provided
    grid cell resolution. Area is in square meters, but resolution
    is given in decimal degrees.'''
    # Calculations needs to be in radians
    lats = np.deg2rad(np.arange(-90, 91, lat_resolution))
    r_sq = 6371000**2
    n_lats = int(360./lon_resolution) 
    area = r_sq*np.ones(n_lats)[:, None]*np.deg2rad(lon_resolution)*(
                np.sin(lats[1:]) - np.sin(lats[:-1]))
    # confirm that area is within 1% of approximate global area (5.10e14 m2)
    assert np.abs( ( (np.sum(area) - 510e12) / 510e12 ) < 0.01)
    return area.T

def reorder_lon_coords(ds):
    '''Usage: takes an xarray DataArray or Dataset and converts longitude 
              coordinates from [-180, 180] --> [0, 360] or [0, 360] --> [-180, 180]'''
    
    ## convert [-180, 180] --> [0, 360]
    if ( (ds.lon.units == 'degrees_east') & (np.min(ds.lon)<0) ):
        EH = ds.sel(lon=slice(0,180))
        WH = ds.sel(lon=slice(-180,0))
        WH['lon'] = WH['lon']+360
        ds = xr.merge((WH, EH))
        
    ## convert [0, 360] --> [-180, 180]
    elif ( (ds.lon.units == 'degrees_east') & (np.max(ds.lon)>180) ):
        EH = ds.sel(lon=slice(0,180))
        WH = ds.sel(lon=slice(180,360))
        WH['lon'] = WH['lon']-360
        ds = xr.merge((WH, EH))
        
    # make sure that we retain attributes on lon
    ds['lon'] = ds['lon'].assign_attrs(EH['lon'].attrs)
        
    return ds

def get_emission_sum(ds, area, units=''):   
    '''
    Args:
        ds : xarray DataArray containing emissions
        units          : (str) emission units, to be used if emission_field attributes are empty
        
    Example Usage: 
        get_emission_sum(ds=WHET['Hg0'].isel(year=2000), area=WHET['area'])
    '''

    if (ds.units=='kg m-2 s-1') or (units=='kg m-2 s-1'):
        emission_sum = (ds*area*sec_per_year*kg_to_Mg).sum().values.item()

    return emission_sum

def get_masked_emission_magnitude(emission_field, mask, area, units=''):   
    '''
    Args:
        emission_field : xarray DataArray containing emissions
        mask           : xarray DataArray with region mask
        units          : (str) emission units, to be used if emission_field attributes are empty
        
    Example Usage: 
        get_masked_emission_magnitude(emission_field=WHET['Hg0'].isel(year=2000), mask=mask_ds['Canada'], area=WHET['area'])
    '''
    # ---------------------------------------------------
    # make sure lon coordinates match in `emission_field` and `mask`
    L1 = emission_field.lon.values
    L2 = mask.lon.values
    assert len(L1) == len(L2) and sorted(L1) == sorted(L2)
    # ---------------------------------------------------
    if (emission_field.units=='kg m-2 s-1') or (units=='kg m-2 s-1'):
        emission_sum = (emission_field*mask*area*sec_per_year*kg_to_Mg).sum().values.item()

    return emission_sum

def mask_for_aggregation(emission_field, mask, units=''):   
    '''
    '''
    # ---------------------------------------------------
    # make sure lon coordinates match in `emission_field` and `mask`
    L1 = emission_field.lon.values
    L2 = mask.lon.values
    assert len(L1) == len(L2) and sorted(L1) == sorted(L2)
    # ---------------------------------------------------
    if (emission_field.units=='kg m-2 s-1') or (units=='kg m-2 s-1'):
        emission_field = (emission_field*mask)
        
        # now mask to replace nan with 0 -- so DataArrays can be added
        emission_field = emission_field.where(~emission_field.isnull(), 0)
    else:
        print('units not recognized - input emission field returned')
        
    return emission_field

def subset_df(df, col_name=str, value=str):
    df = df[df[col_name]==value]
    return df

def get_days_since_epoch(year=int):
    ### calculate days since 1970-01-01 00:00:00
    dt64 = np.datetime64(f'{year}-01-01 00:00:00')
    unix_epoch = np.datetime64(0, 'D')
    one_day = np.timedelta64(1, 'D')
    days_since_epoch = (dt64 - unix_epoch) / one_day
    return days_since_epoch