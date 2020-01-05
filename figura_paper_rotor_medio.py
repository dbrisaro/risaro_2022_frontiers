"""
En esta rutina hago la Figura 2 del paper,
con los rotores medios de NCEP, CFSR, ERA y CCMPv2
Dani Risaro
Enero 2020
"""

import warnings
warnings.filterwarnings('ignore')
import cmocean as cm
import numpy as np
import pandas as pd
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.ndimage.filters import gaussian_filter

# curl and wind data
directory = '/home/daniu/Documentos/datos_rotor_reanalisis_CCMPv2_mensuales/'
datasets = ['CCMPv2','ERAI','CFSR','NCEPR1']
windfiles = '.u-v.mean.gaussiangrid.nc'
curlfiles = '.dv.vor.gaussiangrid.nc'

# area for the plot
lonlatbox_list = [250, 350, -60, -10]
lon_w, lon_e, lat_s, lat_n = lonlatbox_list

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

# figure settings
figsize = (6,4)
splt_pos = [                # the figure has 4 subplots (2x2). Here we set the position of each of them
            [0.05, 0.50, 0.38, 0.38],
            [0.50, 0.50, 0.38, 0.38],
            [0.05, 0.05, 0.38, 0.38],
            [0.50, 0.05, 0.38, 0.38],
            ]
cmap = plt.get_cmap(cm.cm.curl)
lon_ticks = np.arange(-110, 0, 20)
lat_ticks = np.arange(-60, 0, 10)
title = ['a) CCMPv2','b) ERA-Interim','c) CFSR','d) NCEPR1']
fontsize = 6
clevs_curl = np.linspace(-1, 1, 21)
clevs_curl_label = np.linspace(-1.5, 1.5, 7)
nodes = [20, 20, 10, 3]

# figure
plt.close('all')
fig = plt.figure(figsize=figsize)
plt.clf()

for i in range(4):

    # load curl data
    data = xr.open_dataset(directory + datasets[i] + curlfiles).sel(lon=slice(lon_w,lon_e),lat=slice(lat_n,lat_s))
    data_curl = data.svo.mean('time').values*1e5
    lon_curl = data.lon.values
    lat_curl = data.lat.values
    sigma = 1.25 # this depends on how noisy your data is, play with it!
    mean_curl = gaussian_filter(data_curl, sigma)

    # load u.v data
    data = xr.open_dataset(directory + datasets[i] + windfiles).sel(lon=slice(lon_w,lon_e),lat=slice(lat_n,lat_s))
    data_wind = data.mean('time')
    mean_u = data_wind.u.values
    mean_v = data_wind.v.values
    lon_wind = data.lon.values
    lat_wind = data.lat.values

    ax = plt.axes(splt_pos[i], projection=ccrs.Mercator())
    ax.set_extent(lonlatbox_list, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='black', linewidths=0.2, zorder=5)
    ax.add_feature(ccrs.cartopy.feature.LAND, edgecolor='k', color='white', zorder=4)

    if i==0:
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.set_xticklabels([])
        lat_formatter = LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)
    elif i==1:
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    elif i==3:
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.set_yticklabels([])
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        ax.xaxis.set_major_formatter(lon_formatter)
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
    ccurl = plt.contourf(lon_curl, lat_curl, mean_curl, clevs_curl, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
    cbati = plt.contour(blon, blat, data_bati, [-200], colors='gray', linewidths=.25, linestyles='solid',
        transform=ccrs.PlateCarree())

    ccurl_line = plt.contour(lon_curl, lat_curl, mean_curl, clevs_curl_label, colors='k',
        linewidths=.25, transform=ccrs.PlateCarree(), zorder=1, alpha=0.8)
    plt.clabel(ccurl_line, inline=1, inline_spacing=-2, fmt='%2.1f', fontsize=4, colors='k')

    ax.plot(saf_x, saf_y, color='darkblue', linestyle='-', linewidth=0.4, transform=ccrs.PlateCarree())

    n = nodes[i]
    qvr = ax.quiver(lon_wind[::1*n], lat_wind[::1*n], mean_u[::1*n,::1*n], mean_v[::1*n,::1*n], units='xy',
	scale=7e-6, headaxislength=3.5, transform=ccrs.PlateCarree(), color='k', alpha=0.6)
    ax.quiverkey(qvr, 0.75, 1.05, 5, '5 m s$^{-1}$', labelpos='E', coordinates='axes', color='k', fontproperties={'size':6})


cax = fig.add_axes([0.9, 0.07, 0.016, 0.8])
cb = fig.colorbar(ccurl, orientation='vertical', cax=cax)
cb.ax.set_ylabel('Wind stress curl [N m$^{-2}$ km$^{-1}$ *10$^{4}$]', fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize)

fig.savefig('/home/daniu/Documentos/figuras/fig2paper', dpi=300, bbox_inches='tight')
fig.savefig('/home/daniu/Documentos/figuras/fig2paper' + '.pdf', bbox_inches='tight')


# figure
plt.close('all')
fig = plt.figure(figsize=figsize)
plt.clf()

for i in range(4):

    # load curl data
    data = xr.open_dataset(directory + datasets[i] + curlfiles).sel(lon=slice(lon_w,lon_e),lat=slice(lat_n,lat_s))
    data_curl = data.svo.mean('time').values*1e5
    lon_curl = data.lon.values
    lat_curl = data.lat.values
    sigma = 1.25 # this depends on how noisy your data is, play with it!
    mean_curl = gaussian_filter(data_curl, sigma)

    # load u.v data
    data = xr.open_dataset(directory + datasets[i] + windfiles).sel(lon=slice(lon_w,lon_e),lat=slice(lat_n,lat_s))
    data_wind = data.mean('time')
    mean_u = data_wind.u.values
    mean_v = data_wind.v.values
    lon_wind = data.lon.values
    lat_wind = data.lat.values

    ax = plt.axes(splt_pos[i], projection=ccrs.Mercator())
    ax.set_extent(lonlatbox_list, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', color='black', linewidths=0.2, zorder=5)
    ax.add_feature(ccrs.cartopy.feature.LAND, edgecolor='k', color='white', zorder=4)

    if i==0:
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.set_xticklabels([])
        lat_formatter = LatitudeFormatter()
        ax.yaxis.set_major_formatter(lat_formatter)
        mean_curl = mean_curl[::2,::2]
        lon_curl = lon_curl[::2]
        lat_curl = lat_curl[::2]

    elif i==1:
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        mean_curl = mean_curl[::2,::2]
        lon_curl = lon_curl[::2]
        lat_curl = lat_curl[::2]

    elif i==3:
        ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
        ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
        ax.set_yticklabels([])
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        ax.xaxis.set_major_formatter(lon_formatter)
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
    ccurl = plt.contourf(lon_curl, lat_curl, mean_curl, clevs_curl, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
    cbati = plt.contour(blon, blat, data_bati, [-200], colors='gray', linewidths=.25, linestyles='solid',
        transform=ccrs.PlateCarree())

    ccurl_line = plt.contour(lon_curl, lat_curl, mean_curl, clevs_curl_label, colors='k',
        linewidths=.25, transform=ccrs.PlateCarree(), zorder=1, alpha=0.8)
    plt.clabel(ccurl_line, inline=1, inline_spacing=-2, fmt='%2.1f', fontsize=4, colors='k')

    ax.plot(saf_x, saf_y, color='darkblue', linestyle='-', linewidth=0.4, transform=ccrs.PlateCarree())

    n = nodes[i]
    qvr = ax.quiver(lon_wind[::1*n], lat_wind[::1*n], mean_u[::1*n,::1*n], mean_v[::1*n,::1*n], units='xy',
	scale=7e-6, headaxislength=3.5, transform=ccrs.PlateCarree(), color='k', alpha=0.6)
    ax.quiverkey(qvr, 0.75, 1.05, 5, '5 m s$^{-1}$', labelpos='E', coordinates='axes', color='k', fontproperties={'size':6})


cax = fig.add_axes([0.9, 0.07, 0.016, 0.8])
cb = fig.colorbar(ccurl, orientation='vertical', cax=cax)
cb.ax.set_ylabel('Wind stress curl [N m$^{-2}$ km$^{-1}$ *10$^{4}$]', fontsize=fontsize)
cb.ax.tick_params(labelsize=fontsize)

fig.savefig('/home/daniu/Documentos/figuras/fig2paper_prueba', dpi=300, bbox_inches='tight')
fig.savefig('/home/daniu/Documentos/figuras/fig2paper_prueba' + '.pdf', bbox_inches='tight')
