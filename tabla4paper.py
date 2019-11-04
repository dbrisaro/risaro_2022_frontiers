"""
Esta rutina construye la tabla 4 del paper
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
from edof_mpc_py import meanun

## calculo dof
def dof(x, dt=1):
    N = len(x)
    y = meanun(x)
    dof = (N*dt)/(y[3])
    dof = int(dof)
    return dof

## funcion que genera la tabla
def table_generator(directory, filenames, time, tablename):

    nfiles = len(filenames)

    lat_north_n = -42.125;      lat_north_c = -47.375 +(0.75);          lat_north_s = -49.625;
    lat_south_n = -45.875;      lat_south_c = -48.125;          lat_south_s = -52.625;
    lon_west_n = 360-65.125;    lon_west_c = 360-63.875;        lon_west_s = 360-66.375;
    lon_east_n = 360-60.125;    lon_east_c = 360-62.125;        lon_east_s = 360-61.875;

    lat_north = np.stack((lat_north_n, lat_north_c, lat_north_s), axis=0)
    lat_south = np.stack((lat_south_n, lat_south_c, lat_south_s), axis=0)
    lon_west = np.stack((lon_west_n, lon_west_c, lon_west_s), axis=0)
    lon_east = np.stack((lon_east_n, lon_east_c, lon_east_s), axis=0)

    datasets = ['NCEPR1','CFSR','ERAI','CCMPv2']
    boxes = ['NPS','CPS','SPS']
    idx = pd.MultiIndex.from_product([datasets,boxes])

    ws_df = pd.DataFrame(index=time, columns=idx)
    ws_df_dof = pd.DataFrame(index=boxes, columns=datasets)
    ws_rmse = pd.DataFrame(index=boxes, columns=datasets)
    ws_corr = pd.DataFrame(index=boxes, columns=datasets)

    for i, ifile, idataset in zip(range(nfiles), filenames, datasets):
        data = xr.open_dataset(directory + ifile)
        data = data['ws']
        for j, jbox in enumerate(boxes):
            ws = data.sel(longitude=slice(lon_west[j], lon_east[j]), latitude=slice(lat_north[j], lat_south[j])).mean(dim=('latitude','longitude'))
            ws = ws.values
            ws_df[idataset,jbox] = ws
            ws_df_dof[idataset][jbox] = dof(ws_df[idataset][jbox])

    for i, ifile, idataset in zip(range(nfiles), filenames, datasets):
        for j, jbox in enumerate(boxes):
            corr = ws_df[idataset][jbox].corr(ws_df['CCMPv2'][jbox])
            rmse = ((ws_df[idataset][jbox] - ws_df['CCMPv2'][jbox])**2).mean()**.5
            ws_rmse[idataset][jbox] = rmse
            ws_corr[idataset][jbox] = corr

    vars = ['R','RMSE']
    idx_table = pd.MultiIndex.from_product([datasets[0:3], vars])
    ws_table = pd.DataFrame(index=boxes, columns=idx_table)

    ws_table.loc[:,::2] = ws_corr[datasets[0:3]].values
    ws_table.loc[:,1::2] = ws_rmse[datasets[0:3]].values

    ws_table = ws_table.round(2)

    export_csv = ws_table.to_csv(r'/home/daniu/Documentos/tablas/' + tablename + '.csv', index=None, header=True)
    export_csv = ws_df_dof.to_csv(r'/home/daniu/Documentos/tablas/' + tablename + '_dof.csv', index=None, header=True)
    print(ws_table)
    print(ws_df_dof)
    return ws_df

##------------

print('sin filtro')
directory = '/home/daniu/Documentos/datos_ws_reanalisis_CCMPv2_mensuales/'
filenames = ['NCEPR1.ws.10m.mon.mean.1988-2017.nc',\
            'CFSR.ws.10m.mon.mean.1988-2017.nc',\
            'ERAI.ws.10m.mon.mean.1988-2017.nc',\
            'CCMPv2.ws.10m.mon.mean.1988-2017.nc']
time = pd.date_range('1988-01-01','2017-12-31', freq='MS')
tablename = 'tabla4paper-sinfiltrado'
ws_df_sinfilt = table_generator(directory, filenames, time, tablename)

print('con filtro')
directory = '/home/daniu/Documentos/datos_ws_reanalisis_CCMPv2_mensuales/'
filenames = ['NCEPR1.filt36.ws.10m.mon.mean.1988-2017.nc',\
            'CFSR.filt36.ws.10m.mon.mean.1988-2017.nc',\
            'ERAI.filt36.ws.10m.mon.mean.1988-2017.nc',\
            'CCMPv2.filt36.ws.10m.mon.mean.1988-2017.nc']
time = pd.date_range('1989-07-01','2016-07-01', freq='MS')
tablename = 'tabla4paper-filtrado'
ws_df_filt = table_generator(directory, filenames, time, tablename)
