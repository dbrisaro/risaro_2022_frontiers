"""
En esta rutina calculo las series filtradas de
SST a en NPS, CPS y SPS y la evolucion temporal de


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

time_tot = pd.date_range('1982-01-01', '2017-12-31', freq='MS')
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
color_a = ['red', 'grey', 'blue']
color_b = ['lightcoral', 'lightgrey', 'lightblue']
title_a = ['a)', 'b)', 'c)']
title_b = ['d)', 'e)', 'f)']
ylabel_a = ['SST anomalies [$^{\circ}$C]','','']
ylabel_b = ['Linear trend [$^{\circ}$C dec$^{-1}$]','','']
yticks = [-.4, -.2, 0, .2, .4]
yticklabels_a = [yticks, [], []]
yticklabels_b = [yticks, [], []]
nombre_figura = '/home/daniu/Documentos/figuras/figura_paper_series_NPS_CPS_SPS-evolucion_tendencias'

plt.close('all')
figprops = dict(figsize=(10, 3.5), dpi=72)
fig = plt.figure(**figprops)

for i in range(3):
    time_series_sst = data.sel(lat=slice(lat_south[i], lat_north[i]),
                    lon=slice(lon_west[i], lon_east[i])).mean(dim=('lat','lon')).rolling(time=window,
                    center='True').mean().dropna(dim='time',how='any')

    ntime = time_series_sst.time.size
    time = time_series_sst.time.values

    info_trend = pd.DataFrame(index=time, columns=['a', 'b', 'b_sig', 'r_sig'])

    for itime in range(24,ntime):
        xo = time_series_sst.to_dataframe().iloc[0:itime+1]
        xo = xo.to_numpy().squeeze()
        t = np.arange(len(xo))
        p = np.polyfit(t, xo, 1)
        r = np.corrcoef(t, xo)[0,1]
        degf = dof(xo)
        is_sig = sig_r(degf, r)
        info_trend.iloc[itime,0] = p[1]
        info_trend.iloc[itime,1] = p[0]*120
        info_trend.iloc[itime,2] = p[0]*120*is_sig
        info_trend.iloc[itime,3] = r*is_sig

    pp = np.poly1d(p)
    ajuste = pp(t)

#    str_fit = 'y={:.2f}$^{{\circ}}$C dec$^{{-1}} {c}$ + {:.2f}'.format(p[0]*120, p[1], c='t')
    str_fit = 'y={:.2f}$^{{\circ}}$C/dec ${c}$ + {:.2f}'.format(p[0]*120, p[1], c='t')

    # let the plot begin
    no = 0.05 + 0.3*i
    splt_a = [no, 0.55, 0.27, 0.4]
    splt_b = [no, 0.05, 0.27, 0.4]

    ax = plt.axes(splt_a)
    bx = plt.axes(splt_b)

    ax.plot(time, time_series_sst.sst.values, color=color_a[i], label=cajas[i], lw=.5)
    ax.plot(time, ajuste, color=color_a[i], alpha=0.5, linestyle='--', lw=.5, label=str_fit)
    ax.legend(fontsize=5, loc='upper left')
    ax.set_xticklabels([])
    ax.axhline(y=0, color='k', lw=0.5)
    ax.set_ylim([-.5, .5])
    ax.set_xlim([time_tot[0], time_tot[-1]])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels_a[i])
    ax.set_ylabel(ylabel_a[i], fontsize=6)
    ax.tick_params('both', labelsize=6)
    ax.set_title(title_a[i], loc='left', fontsize=6)

    bx.plot(time, info_trend['b'], color=color_b[i], lw=.75)
    bx.plot(time, info_trend['b_sig'], color='k', marker='*',
        markersize=1, linestyle='', markerfacecolor='k')
    bx.axhline(y=0, color='k', linewidth=0.5)
    bx.set_ylabel(ylabel_b[i], fontsize=6)
    bx.set_ylim([-.5, .5])
    bx.set_xlim([time_tot[0], time_tot[-1]])
    bx.set_yticks(yticks)
    bx.set_yticklabels(yticklabels_b[i])
    bx.set_xlabel('Time [Years]', fontsize=6)
    bx.set_title(title_b[i], loc='left', fontsize=6)
    bx.tick_params('both', labelsize=6)

fig.savefig(nombre_figura, dpi=300, bbox_inches='tight')
fig.savefig(nombre_figura + '.pdf', bbox_inches='tight')


### complementary figure


cajas = ['NPS', 'CPS', 'SPS']
window = 36
color_a = ['red', 'grey', 'blue']
color_b = ['lightcoral', 'lightgrey', 'lightblue']
title_a = ['a)', 'b)', 'c)']
title_b = ['d)', 'e)', 'f)']
ylabel_a = ['SST anomalies [$^{\circ}$C]','','']
ylabel_b = ['Linear trend [$^{\circ}$C dec$^{-1}$]','','']
yticks = [-.6, -.4, -.2, 0, .2, .4, .6]
yticklabels_a = [yticks, [], []]
yticklabels_b = [yticks, [], []]
nombre_figura = '/home/daniu/Documentos/figuras/figura_paper_series_NPS_CPS_SPS-con std'

plt.close('all')
figprops = dict(figsize=(10, 3.5), dpi=72)
fig = plt.figure(**figprops)

for i in range(3):
    time_series_sst = data.sel(lat=slice(lat_south[i], lat_north[i]),
                    lon=slice(lon_west[i], lon_east[i])).mean(dim=('lat','lon')).rolling(time=window,
                    center='True').mean().dropna(dim='time',how='any')

    std_sst = data.sel(lat=slice(lat_south[i], lat_north[i]),
                    lon=slice(lon_west[i], lon_east[i])).std(dim=('lat','lon')).rolling(time=window,
                    center='True').mean().dropna(dim='time',how='any')

    err_temp = std_sst.sst.values
    temp = time_series_sst.sst.values
    ntime = time_series_sst.time.size
    time = time_series_sst.time.values

    info_trend = pd.DataFrame(index=time, columns=['a', 'b', 'b_sig', 'r_sig'])

    for itime in range(24,ntime):
        xo = time_series_sst.to_dataframe().iloc[0:itime+1]
        xo = xo.to_numpy().squeeze()
        t = np.arange(len(xo))
        p = np.polyfit(t, xo, 1)
        r = np.corrcoef(t, xo)[0,1]
        degf = dof(xo)
        is_sig = sig_r(degf, r)
        info_trend.iloc[itime,0] = p[1]
        info_trend.iloc[itime,1] = p[0]*120
        info_trend.iloc[itime,2] = p[0]*120*is_sig
        info_trend.iloc[itime,3] = r*is_sig

    pp = np.poly1d(p)
    ajuste = pp(t)

#    str_fit = 'y={:.2f}$^{{\circ}}$C dec$^{{-1}} {c}$ + {:.2f}'.format(p[0]*120, p[1], c='t')
    str_fit = 'y={:.2f}$^{{\circ}}$C/dec ${c}$ + {:.2f}'.format(p[0]*120, p[1], c='t')

    # let the plot begin
    no = 0.05 + 0.3*i
    splt_a = [no, 0.05, 0.27, 0.5]

    ax = plt.axes(splt_a)

    ax.plot(time, time_series_sst.sst.values, color=color_a[i], label=cajas[i], lw=.5)
    ax.fill_between(time, temp-err_temp, temp+err_temp, color=color_a[i], alpha=.15, edgecolor='white', lw=0)
    ax.plot(time, ajuste, color=color_a[i], alpha=0.5, linestyle='--', lw=.5, label=str_fit)
    ax.legend(fontsize=5, loc='upper left')
    ax.set_xticklabels([])
    ax.axhline(y=0, color='k', lw=0.5)
    ax.set_ylim([-.75, .75])
    ax.set_xlim([time_tot[0], time_tot[-1]])
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels_a[i])
    ax.set_ylabel(ylabel_a[i], fontsize=6)
    ax.tick_params('both', labelsize=6)
    ax.set_title(title_a[i], loc='left', fontsize=6)

fig.savefig(nombre_figura, dpi=300, bbox_inches='tight')
fig.savefig(nombre_figura + '.pdf', bbox_inches='tight')
