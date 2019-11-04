"""
Esta rutina construye la tabla 5 del paper
Dani Risaro
Octubre 2019
"""

import warnings
warnings.filterwarnings('ignore')
import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir('/home/daniu/Documentos/rutinas')
#os.chdir('/media/nico/Seagate Expansion Drive/Documentos_DELL_home/rutinas/')
from edof_mpc_py import meanun

## calculo dof
def dof(x, dt=1):
    N = len(x)
    y = meanun(x)
    dof = (N*dt)/(y[3])
    dof = int(dof)
    return dof


time_a = pd.date_range('1983-06-01','2007-12-01', freq='MS')
time_b = pd.date_range('2008-01-01','2016-06-01', freq='MS')
time_c = pd.date_range('1983-06-01','2016-06-01', freq='MS')

time = [time_a, time_b, time_c]

lat_north_n = -42.125;      lat_north_c = -47.375;          lat_north_s = -49.625;
lat_south_n = -45.875;      lat_south_c = -48.125;          lat_south_s = -52.625;
lon_west_n = 360-65.125;    lon_west_c = 360-63.875;        lon_west_s = 360-66.375;
lon_east_n = 360-60.125;    lon_east_c = 360-62.125;        lon_east_s = 360-61.875;

lat_north = np.stack((lat_north_n, lat_north_c, lat_north_s), axis=0)
lat_south = np.stack((lat_south_n, lat_south_c, lat_south_s), axis=0)
lon_west = np.stack((lon_west_n, lon_west_c, lon_west_s), axis=0)
lon_east = np.stack((lon_east_n, lon_east_c, lon_east_s), axis=0)


directory = ['/media/nico/Seagate Expansion Drive/Documentos_DELL_home/datos_ERA_interim_u_v_SLP/output/',\
            '/media/nico/Seagate Expansion Drive/Documentos_DELL_home/datos_ERA_interim_u_v_SLP/output/',\
            '/media/nico/Seagate Expansion Drive/Documentos_DELL_home/datos_ERA_interim_u_v_SLP/output/',\
            '/media/nico/Seagate Expansion Drive/Documentos_DELL_home/datos_ERA_interim_u_v_SLP/output/',\
            '/media/nico/Seagate Expansion Drive/Documentos_DELL_home/datos_ERA_interim_u_v_SLP/output/',\
            '/media/nico/Seagate Expansion Drive/Documentos_DELL_home/datos_ERA_interim_hf/output/']

directory = ['/home/daniu/Documentos/datos_ERA_interim_u_v_SLP/output/',\
            '/home/daniu/Documentos/datos_ERA_interim_u_v_SLP/output/',\
            '/home/daniu/Documentos/datos_ERA_interim_u_v_SLP/output/',\
            '/home/daniu/Documentos/datos_ERA_interim_u_v_SLP/output/',\
            '/home/daniu/Documentos/datos_ERA_interim_u_v_SLP/output/',\
            '/home/daniu/Documentos/datos_ERA_interim_hf/output/']

filenames = ['filt36_anom_erai_1982-2017_swa.nc',\
            'filt36_anom_erai_1982-2017_swa.nc',\
            'filt36_ws_anom_erai_1982-2017_swa_sep.nc',\
            'filt36_dv_vor_anom_erai_1982-2017_gaussiangrid_swa_sep.nc',\
            'filt36_anom_erai_1982-2017_swa.nc',\
            'filt36_hf_anom_erai_1982-2017_swa_sep.nc']

variables = ['u10','v10','ws','svo','msl','nhf']

boxes = ['NPS','CPS','SPS']
idx = pd.MultiIndex.from_product([variables,boxes])

sst_df = pd.DataFrame(index=time_c, columns=boxes)
wind_df = pd.DataFrame(index=time_c, columns=idx)
wind_df_dof = pd.DataFrame(index=boxes, columns=variables)

#filename_sst = '/media/nico/Seagate Expansion Drive/Documentos_DELL_home/datos_reynolds/output/filt36_anom_sst_monthly_reynolds_1982-2017_swa.nc'
filename_sst = '/home/daniu/Documentos/datos_reynolds/output/filt36_anom_sst_monthly_reynolds_1982-2017_swa.nc'
data_sst = xr.open_dataset(filename_sst)

for i, ifile in enumerate(filenames):
    data_wind = xr.open_dataset(directory[i] + ifile)
    for j, jbox in enumerate(boxes):
        s = data_sst.sel(lon=slice(lon_west[j], lon_east[j]), lat=slice(lat_south[j],lat_north[j])).mean(dim=('lat','lon'))
        sst_df[jbox] = np.append(s.sst.values, s.sst.values[-1])
        d = data_wind.sel(longitude=slice(lon_west[j], lon_east[j]), latitude=slice(lat_north[j], lat_south[j])).mean(dim=('latitude','longitude'))
        var = d[variables[i]].values
        wind_df[variables[i],jbox] = var
        wind_df_dof[variables[i]][jbox] = dof(wind_df[variables[i]][jbox])

periods = ['1982-2007','2008-2017','1982-2017']
idx = pd.MultiIndex.from_product([boxes,periods])
wind_corr = pd.DataFrame(index=variables, columns=idx)
wind_df_dof = pd.DataFrame(index=variables, columns=idx)

for i, ivar in enumerate(variables):
    for j, jbox in enumerate(boxes):
        corr = (wind_df[ivar][jbox].loc['1982':'2007']).corr((sst_df[jbox]).loc['1982':'2007'])
        wind_corr.iloc[i,3*j] = corr
        wind_df_dof.iloc[i,3*j] = dof(wind_df[ivar][jbox].loc['1982':'2007'])

        corr = (wind_df[ivar][jbox].loc['2008':'2017']).corr((sst_df[jbox]).loc['2008':'2017'])
        wind_corr.iloc[i,3*j+1] = corr
        wind_df_dof.iloc[i,3*j+1] = dof(wind_df[ivar][jbox].loc['2008':'2017'])

        corr = wind_df[ivar][jbox].corr(sst_df[jbox])
        wind_corr.iloc[i,3*j+2] = corr
        wind_df_dof.iloc[i,3*j+2] = dof(wind_df[ivar][jbox])

wind_corr = wind_corr.astype('float').round(2)
export_csv = wind_corr.to_csv(r'/home/daniu/Documentos/tablas/tabla5paper.csv', index=None, header=True)
print(wind_corr)
print(wind_df_dof)
