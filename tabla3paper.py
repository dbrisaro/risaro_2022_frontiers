"""
Calculo las caracteristicas climatologicas de las cajas
NPS, CPS, SPS para la SST
Dani Risaro
Marzo 2019
"""
import numpy as np
import pandas as pd
import xarray as xr

archivo = '/home/daniu/Documentos/datos_reynolds/output/sst_monthly_reynolds_1982-2018_corregida_climatologia.nc'

time = pd.date_range('1982-01-01', '2017-12-31', freq='MS')
clim_sst = xr.open_dataset(archivo)

lat_north_n = -42.125;      lat_north_c = -47.375;          lat_north_s = -49.625;
lat_south_n = -45.875;      lat_south_c = -48.125;          lat_south_s = -52.625;
lon_west_n = 360-65.125;    lon_west_c = 360-(64.875-1);        lon_west_s = 360-66.375;
lon_east_n = 360-60.125;    lon_east_c = 360-(63.125-1);        lon_east_s = 360-61.875;

lat_north = np.stack((lat_north_n, lat_north_c, lat_north_s), axis=0)
lat_south = np.stack((lat_south_n, lat_south_c, lat_south_s), axis=0)
lon_west = np.stack((lon_west_n, lon_west_c, lon_west_s), axis=0)
lon_east = np.stack((lon_east_n, lon_east_c, lon_east_s), axis=0)

tabla_estadisticos = pd.DataFrame(index=['NPS', 'CPS', 'SPS'],
        columns=['Annual mean SST (ºC)',
                'Maximum climatological SST (ºC)',
                'Minimum climatological SST (ºC)',
                'Amplitude SST (ºC)'])

for i in range(len(lat_north)):

    sst = clim_sst.sel(lat=slice(lat_south[i], lat_north[i]),
                    lon=slice(lon_west[i], lon_east[i])).mean(dim=('lat','lon'))

    mean_sst = sst.mean().sst.values
    sst_max = sst.max().sst.values
    sst_min = sst.min().sst.values

    amplitud = sst_max - sst_min

    tabla_estadisticos.iloc[i,0] = mean_sst
    tabla_estadisticos.iloc[i,1] = sst_max
    tabla_estadisticos.iloc[i,2] = sst_min
    tabla_estadisticos.iloc[i,3] = amplitud

tabla_final = tabla_estadisticos.astype('float').round(1)
tabla_tex = tabla_final.to_latex()

print(tabla_final)
tabla_final.to_csv('/home/daniu/Documentos/tablas/tabla_estadisticos_SST_patagonia_1982_2017.csv', sep=',')
