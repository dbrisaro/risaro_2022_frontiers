""""
Esta rutina construye la tabla 6 del paper
Dani Risaro
Octubre 2019
"""

import warnings
warnings.filterwarnings('ignore')
import xarray as xr
import numpy as np
import pandas as pd
import os
from scipy import signal
import matplotlib.pyplot as plt

os.chdir('/home/daniu/Documentos/rutinas')

#os.chdir('/media/nico/Seagate Expansion Drive/Documentos_DELL_home/rutinas/')
from edof_mpc_py import meanun
from indices_climaticos_mensuales import carga_indices

## calculo dof
def dof(x, dt=1):
    N = len(x)
    y = meanun(x)
    dof = (N*dt)/(y[3])
    dof = int(dof)
    return dof

time_c = pd.date_range('1983-07-01','2016-07-01', freq='MS')
time_tot = pd.date_range('1982-01-01','2017-12-31', freq='MS')
time = time_c

lat_north_n = -42.125;      lat_north_c = -47.375;          lat_north_s = -49.625;
lat_south_n = -45.875;      lat_south_c = -48.125;          lat_south_s = -52.625;
lon_west_n = 360-65.125;    lon_west_c = 360-63.875;        lon_west_s = 360-66.375;
lon_east_n = 360-60.125;    lon_east_c = 360-62.125;        lon_east_s = 360-61.875;

lat_north = np.stack((lat_north_n, lat_north_c, lat_north_s), axis=0)
lat_south = np.stack((lat_south_n, lat_south_c, lat_south_s), axis=0)
lon_west = np.stack((lon_west_n, lon_west_c, lon_west_s), axis=0)
lon_east = np.stack((lon_east_n, lon_east_c, lon_east_s), axis=0)

## cargo SST!!
boxes = ['NPS','CPS','SPS']

sst_df = pd.DataFrame(index=time_c, columns=boxes)

#filename_sst = '/media/nico/Seagate Expansion Drive/Documentos_DELL_home/datos_reynolds/output/filt36_anom_sst_monthly_reynolds_1982-2017_swa.nc'
filename_sst = '/home/daniu/Documentos/datos_reynolds/output/filt36_anom_sst_monthly_reynolds_1982-2017_swa.nc'
data_sst = xr.open_dataset(filename_sst)

for j, jbox in enumerate(boxes):
    s = data_sst.sel(lon=slice(lon_west[j], lon_east[j]), lat=slice(lat_south[j],lat_north[j])).mean(dim=('lat','lon'))
    sst_df[jbox] = np.append(s.sst.values, s.sst.values[-1])

## cargo indices!!
ind = ['AAO','SAM','ENSO 3.4','SOI','PDO','IPO','NAO']

ind_df = pd.DataFrame(index=time_tot, columns=ind)

AAO, SAM, NINO34, SOI, PDO, IPO, NAO = carga_indices()

ind_df['AAO'] = AAO.loc['1982':'2017']
ind_df['SAM'] = SAM.loc['1982':'2017']
ind_df['ENSO 3.4'] = NINO34.loc['1982':'2017']
ind_df['SOI'] = SOI.loc['1982':'2017']
ind_df['PDO'] = PDO.loc['1982':'2017']
ind_df['IPO'] = IPO.loc['1982':'2017']
ind_df['NAO'] = NAO.loc['1982':'2017']

ind_df = ind_df.rolling(window=36, center=True).mean().dropna()

## Cargo PCs!!
pcs = ['PC1','PC2','PC3']

pc_df = pd.read_csv('/home/daniu/Documentos/tablas/eof_pcs_110W10W60S10S.csv', sep=',', index_col=[0])
pc_df.loc['2016-07-01'] = pc_df.loc['2016-06-01']

# -- correlaciones
pcs_corr = pd.DataFrame(index=pcs, columns=ind)
ssta_corr = pd.DataFrame(index=boxes, columns=ind)

for i, iind in enumerate(ind):
    for j in range(3):
        jbox = boxes[j]
        corr = ind_df[iind].corr(sst_df[jbox])
        ssta_corr[iind][jbox] = corr

        jpc = pcs[j]
        corr = ind_df[iind].corr(pc_df.iloc[:,j])
        pcs_corr[iind][jpc] = corr

pcs_corr = pcs_corr.astype('float').round(2)
ssta_corr = ssta_corr.astype('float').round(2)

export_csv = pcs_corr.to_csv(r'/home/daniu/Documentos/tablas/tabla6paper.csv', index=None, header=True)
print(pcs_corr)
print(ssta_corr)

# -- correlacion SST en SPS, CPS y NPS con PC
pcs_ssta_corr = pd.DataFrame(index=pcs, columns=boxes)
for j in range(3):
    for i in range(3):
        jbox = boxes[j]
        jpc = pcs[i]
        corr = sst_df[jbox].corr(pc_df.iloc[:,i])
        pcs_ssta_corr[jbox][jpc] = corr

pcs_ssta_corr = pcs_ssta_corr.astype('float').round(2)
print(pcs_ssta_corr)


# -- correlacion SST en SPS, CPS y NPS con PC
pcs_detrended_ssta_corr = pd.DataFrame(index=pcs, columns=boxes)
for j in range(3):
    for i in range(3):
        jbox = boxes[j]
        jpc = pcs[i]
        jsst = pd.Series(signal.detrend(sst_df[jbox]), index=time_c)
        corr = jsst.corr(pc_df.iloc[:,i])
        pcs_detrended_ssta_corr[jbox][jpc] = corr

pcs_detrended_ssta_corr = pcs_detrended_ssta_corr.astype('float').round(2)
print(pcs_detrended_ssta_corr)
