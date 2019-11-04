"""
En esta rutina calculo las series filtradas de
SST a en NPS y CPS y la serie de correlacion entre ellas

Dani Risaro
Noviembre 2019
"""

import numpy as np
import pandas as pd
import xarray as xr
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

def sig_r(dof, r):
    """
    Calcula los coeficientes de correlacion (R) significativos
    a partir de los grados de libertad.

    Parámetros de entrada
    dof: float o int. Grados de libertad de la serie
    r: coeficiente de correlacion

    Output:
    d: 1 o 0. Indica si el R es significativo o no
    """

    tabla = np.loadtxt('/home/daniu/Documentos/tablas/dof.txt', delimiter=',')
    indice = np.argmin(np.abs(tabla[:,0]-dof))
    r_corte = tabla[indice,1]
    if np.abs(r) > r_corte:
        d = 1
    else:
        d = np.nan
    return d

#-----------------------------------

archivo = '/home/daniu/Documentos/datos_reynolds/output/anom_sst_monthly_reynolds_1982-2017_swa.nc'

time = pd.date_range('1982-01-01', '2017-12-31', freq='MS')
data = xr.open_dataset(archivo)

lat_north_n = -42.125;       lat_north_s = -49.625;
lat_south_n = -45.875;       lat_south_s = -52.625;
lon_west_n = 360-65.125;     lon_west_s = 360-66.375;
lon_east_n = 360-60.125;     lon_east_s = 360-61.875;

lat_north = np.stack((lat_north_n, lat_north_s), axis=0)
lat_south = np.stack((lat_south_n, lat_south_s), axis=0)
lon_west = np.stack((lon_west_n, lon_west_s), axis=0)
lon_east = np.stack((lon_east_n, lon_east_s), axis=0)

cajas = ['NPS', 'SPS']
window = 36
color = ['red', 'blue']

time_series_sst = []

for i in range(2):      # extract temporal series
    a = data.sel(lat=slice(lat_south[i], lat_north[i]),
                    lon=slice(lon_west[i], lon_east[i])).mean(dim=('lat','lon')).rolling(time=window,
                    center='True').mean().dropna(dim='time',how='any')
    time_series_sst.append(a)

    ntime = a.time.size


r = [np.nan, np.nan]
r_sig = [np.nan, np.nan]
for itime in range(2,ntime):
    xo = time_series_sst[0].to_dataframe().iloc[0:itime]
    yo = time_series_sst[1].to_dataframe().iloc[0:itime]
    c = xo.corrwith(yo)
    r.append(c.values)

    degf = dof(xo.to_numpy())
    is_sig = sig_r(degf, c.values)
    r_sig.append(c*is_sig)

# I remove the first two years of r data, as it's too noisy
r[0:24] = [i * np.nan for i in r[0:24]]

time = time_series_sst[0].time.values
nombre_figura = '/home/daniu/Documentos/figuras/figura_paper_correlacion_NPS_SPS'

plt.close('all')
figprops = dict(figsize=(5, 3.5), dpi=72)
fig = plt.figure(**figprops)
ax = plt.axes([0.1, 0.5, 0.85, 0.4])
bx = plt.axes([0.1, 0.05, 0.85, 0.4])
ax.plot(time, time_series_sst[0].sst.values, 'r', label='NPS', lw=0.5)
ax.plot(time, time_series_sst[1].sst.values, 'b', label='SPS', lw=0.5)
ax.legend(fontsize=6)
ax.set_xticklabels([])
ax.axhline(y=0, color='k', lw=0.5)
ax.set_ylim([-.5, .5])
ax.set_ylabel('SST anomalies [$^{\circ}$C]', fontsize=6)
ax.set_title('a)', fontsize=6, loc='left')
ax.tick_params('both', labelsize=6)

bx.plot(time, r, 'grey', lw=.75, alpha=0.5)
bx.plot(time, r_sig, color='k', marker='*',
    markersize=1, linestyle='', markerfacecolor='k')
bx.axhline(y=0, color='k', linewidth=0.5)
bx.set_ylabel('Correlation', fontsize=6)
bx.set_ylim([-1, 1])
bx.set_yticks([-1,-.5,0,.5,1])
bx.set_xlabel('Time [Years]', fontsize=6)
bx.set_title('b)', fontsize=6, loc='left')
bx.tick_params('both', labelsize=6)

fig.savefig(nombre_figura, dpi=300, bbox_inches='tight')
fig.savefig(nombre_figura + '.pdf', bbox_inches='tight')
