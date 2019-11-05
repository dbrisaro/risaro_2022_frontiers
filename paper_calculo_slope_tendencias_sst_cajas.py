"""
En esta rutina calculo las tendencias de las series filtradas de
SST a en NPS, CPS y SPS con un breakpoint en el medio


Dani Risaro
Noviembre 2019
"""

import numpy as np
import pandas as pd
import xarray as xr

archivo = '/media/daniu/Seagate Expansion Drive/Documentos_DELL_home/datos_reynolds/output/anom_sst_monthly_reynolds_1982-2017_swa.nc'

data = xr.open_dataset(archivo)

lat_north_n = -42.125;       lat_north_c = -47.375;           lat_north_s = -49.625;
lat_south_n = -45.875;       lat_south_c = -48.125;           lat_south_s = -52.625;
lon_west_n = 360-65.125;     lon_west_c = 360-(64.875-1);         lon_west_s = 360-66.375;
lon_east_n = 360-60.125;     lon_east_c = 360-(63.125-1);         lon_east_s = 360-61.875;

lat_north = np.stack((lat_north_n, lat_north_c, lat_north_s), axis=0)
lat_south = np.stack((lat_south_n, lat_south_c, lat_south_s), axis=0)
lon_west = np.stack((lon_west_n, lon_west_c, lon_west_s), axis=0)
lon_east = np.stack((lon_east_n, lon_east_c, lon_east_s), axis=0)

cajas = ['NPS', 'CPS', 'SPS']
window = 36

periods = ['1982-2007','2008-2017','1982-2017']
df = pd.DataFrame(index=cajas, columns=periods)
new_df = pd.DataFrame(index=cajas, columns=periods)

for i in range(3):
    time_series_sst = data.sel(lat=slice(lat_south[i], lat_north[i]),
                    lon=slice(lon_west[i], lon_east[i])).mean(dim=('lat','lon')).rolling(time=window,
                    center='True').mean().dropna(dim='time',how='any')
    sst_df = time_series_sst.to_dataframe()

    xo = sst_df.loc['1982':'2007']
    xo = xo.to_numpy().squeeze()
    t = np.arange(len(xo))
    p = np.polyfit(t, xo, 1)
    df.loc[cajas[i], periods[0]] =  p[0]*120

    xo = sst_df.loc['2008':'2017']
    xo = xo.to_numpy().squeeze()
    t = np.arange(len(xo))
    p = np.polyfit(t, xo, 1)
    df.loc[cajas[i], periods[1]] =  p[0]*120

    xo = sst_df.loc['1982':'2017']
    xo = xo.to_numpy().squeeze()
    t = np.arange(len(xo))
    p = np.polyfit(t, xo, 1)
    df.loc[cajas[i], periods[2]] =  p[0]*120

# en este df esta guardado la tendencia de sst en cada periodo
df = df.astype('float').round(2)
print(df)

# en este df esta guardado el aumento o disminucion neta de sst en cada periodo
new_df.loc[:,'1982-2007'] = df.loc[:,'1982-2007']*294/120
new_df.loc[:,'2008-2017'] = df.loc[:,'2008-2017']*102/120
new_df.loc[:,'1982-2017'] = df.loc[:,'1982-2017']*396/120
new_df = new_df.round(2)
print(new_df)
