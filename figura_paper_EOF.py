"""
Calculo los EOFs de SSTa del paper en el dominio
110-10W 60-10S

Dani Risaro
Noviembre 2019
"""

import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import warnings
import xarray as xr
from eofs.standard import Eof
warnings.filterwarnings('ignore')
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# area for the plot
lonlatbox_list = [250, 350, -60, -10]
lon_w, lon_e, lat_s, lat_n = lonlatbox_list

# load SST date already filtered
archivo = '/home/daniu/Documentos/datos_reynolds/output/filt36_anom_sst_monthly_reynolds_1982-2017_swa_sep_detrended.nc'
data = xr.open_dataset(archivo)
lon_sst = data.lon.values
lat_sst = data.lat.values
time_sst = data.time.values

# load topo data
data_bati = xr.open_dataset('/home/daniu/Documentos/batimetria/ETOPO1_Bed_g_gmt4.grd')
data_bati = data_bati.sel(x=slice(-(360-lon_w), -(360-lon_e)), \
                    y=slice(lat_s, lat_n))
blon = data_bati.x.values
blat = data_bati.y.values
data_bati = data_bati.z.values

# load SAF data
saf = np.loadtxt('/home/daniu/Documentos/frentes/saf_orsi.csv', delimiter=',')
saf_lon = saf[:,0]
saf_lat = saf[:,1]
ind_lon = np.where(((saf_lon >= lon_w) & (saf_lon <= lon_e)))
ind_lat = np.where(((saf_lat >= lat_s) & (saf_lat <= lat_n)))
ind = np.intersect1d(ind_lon, ind_lat)
saf_x = saf_lon[ind]
saf_y = saf_lat[ind]

# EOF calculator
coslat = np.cos(np.deg2rad(data.coords['lat'].values))
wgts = np.sqrt(coslat)[..., np.newaxis]
solver_sst = Eof(data.sst.values, weights=wgts)

cant_modos = 3
scaling_pc = 1      #          * *0* : Un-scaled EOFs (default).
scaling_eof = 2     #          * *1* : EOFs are divided by the square-root of their eigenvalues.
                    #          * *2* : EOFs are multiplied by the square-root of their eigenvalues.

eof_sst = solver_sst.eofs(neofs=cant_modos, eofscaling=scaling_eof)
pc_sst = solver_sst.pcs(npcs=cant_modos, pcscaling=scaling_pc)
varfrac_sst = solver_sst.varianceFraction()
lambdas_sst = solver_sst.eigenvalues()

# details for the figure
pos = []
for i in range(3):
    xo = 0.32*(i+1)

    splt_pos = [
                [0.05, 1-xo, 0.4, 0.33],
                [0.53, 1-xo+0.165, 0.45, 0.13],
                ]

    pos.append(splt_pos)

lon_ticks = np.arange(-110, 0, 20)
lat_ticks = np.arange(-60, 0, 10)

clevs_sst = np.linspace(-.3, .3, 13)
clevs_label_sst = np.linspace(-.3, .3, 7)
cmap = plt.cm.RdBu_r

cbar_label_sst = 'SST anomalies [$^{\circ}$C]'
title = ['a) EOF 1 (var={0:.0f}%)'.format(varfrac_sst[0]*100),\
            'b) EOF 2 (var={0:.0f}%)'.format(varfrac_sst[1]*100),\
            'c) EOF 3 (var={0:.0f}%)'.format(varfrac_sst[2]*100)]

fontsize = 6
figsize = (7, 7)

# figure
plt.close('all')
fig = plt.figure(figsize=figsize)
plt.clf()

for i in range(3):

    matriz_eof = eof_sst[i,:,:]
    ax = plt.axes(pos[i][0], projection=ccrs.Mercator())
    ax.set_extent(lonlatbox_list, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='black', linewidths=0.2, zorder=5)
    ax.add_feature(ccrs.cartopy.feature.LAND, edgecolor='k', color='white', zorder=4)

    if i<2:
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.set_xticklabels([])
        lat_formatter = LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)

    else:
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

    ax.tick_params('both', labelsize=fontsize)
    ax.set_aspect('equal', 'box')
    ax.set_title(title[i], fontsize=fontsize, loc='left')
    ceof = plt.contourf(lon_sst, lat_sst, matriz_eof, clevs_sst, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
    cbati = plt.contour(blon, blat, data_bati, [-200], colors='gray', linewidths=.25, linestyles='solid', transform=ccrs.PlateCarree())
    ceof_line = plt.contour(lon_sst, lat_sst, matriz_eof, clevs_label_sst, colors='k',
           linewidths=.25, transform=ccrs.PlateCarree(), zorder=1, alpha=0.8)
    plt.clabel(ceof_line, inline=True, inline_spacing=3, fmt='%2.1f', fontsize=3, colors='k')

    ax.plot(saf_x, saf_y, color='darkblue', linestyle='-', linewidth=0.4, transform=ccrs.PlateCarree())

cax = fig.add_axes([0.05, 0.02, 0.4, 0.01])
cb = fig.colorbar(ceof, orientation='horizontal', cax=cax)
cb.ax.set_xlabel(cbar_label_sst, fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize)

for i in range(3):
    pc_eof = pc_sst[:,i]
    ax = plt.axes(pos[i][1])
    if i==2:
        ax.set_xlabel('Time [years]',fontsize=6)

    ax.tick_params('both', labelsize=fontsize)
    ax.plot(time_sst, pc_eof, lw=0.5, color='b')
    ax.axhline(y=0, linewidth=0.35, color='k')
    ax.set_ylabel('PC ' + str(i+1),fontsize=6)
    ax.set_ylim([-2.5, 2.5])

fig.savefig('/home/daniu/Documentos/figuras/fig_paper_eof', dpi=300, bbox_inches='tight')
fig.savefig('/home/daniu/Documentos/figuras/fig_paper_eof' + '.pdf', bbox_inches='tight')
