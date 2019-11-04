"""
En esta rutina hago la Figura 8 o 9 del paper,
con las tendencias de u, v, int, curl, msl y hf
para distintos periodos
Dani Risaro
Octubre 2019
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
directory = '/home/daniu/Documentos/breakpoint-2008/'
meanfiles = ['sst_mean_swa_sep.nc', 'erai_mean_swa_sep.nc',
            'erai_mean_swa_sep.nc', 'ws_mean_swa_sep.nc',
            'dvor_mean_swa_sep.nc', 'erai_mean_swa_sep.nc',
            'hf_mean_swa_sep.nc']

meanfiles_uv = ['', 'erai_mean_swa_sep_1982-2007.nc', 'erai_mean_swa_sep_2008-2017.nc']

trendfiles_1 = ['trend_b_filt36_anom_sst_monthly_reynolds_1982-2007_swa_sep.nc',
                'trend_b_filt36_anom_erai_1982-2007_swa_sep.nc',
                'trend_b_filt36_anom_erai_1982-2007_swa_sep.nc',
                'trend_b_filt36_ws_anom_erai_1982-2007_swa_sep.nc',
                'trend_b_filt36_dv_vor_anom_erai_1982-2007_gaussiangrid_swa_sep.nc',
                'trend_b_filt36_anom_erai_1982-2007_swa_sep.nc',
                'trend_b_filt36_hf_anom_erai_1982-2007_swa_sep.nc']

trendfiles_2 = ['trend_b_filt36_anom_sst_monthly_reynolds_2008-2017_swa_sep.nc',
                'trend_b_filt36_anom_erai_2008-2017_swa_sep.nc',
                'trend_b_filt36_anom_erai_2008-2017_swa_sep.nc',
                'trend_b_filt36_ws_anom_erai_2008-2017_swa_sep.nc',
                'trend_b_filt36_dv_vor_anom_erai_2008-2017_gaussiangrid_swa_sep.nc',
                'trend_b_filt36_anom_erai_2008-2017_swa_sep.nc',
                'trend_b_filt36_hf_anom_erai_2008-2017_swa_sep.nc']

files = [meanfiles, trendfiles_1, trendfiles_2]


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
figsize = (8.27, 11.69)

#[left, bottom, width, height]
# the figure has 21 subplots (7x3). Here we set the position of each of them

pos = []
pos_cb = []
for i in range(7):
    xo = 0.13*(i+1)

    splt_pos = [
                [0.05, 1-xo, 0.30, 0.1],
                [0.37, 1-xo, 0.30, 0.1],
                [0.67, 1-xo, 0.30, 0.1],
                ]

    splt_cb_pos = [
                [0.33, 1-xo, 0.01, 0.1],
                [0.65, 1-xo, 0.01, 0.1],
                [0.96, 1-xo, 0.01, 0.1],
                ]

    pos.append(splt_pos)
    pos_cb.append(splt_cb_pos)


lon_ticks = np.arange(-110, 0, 20)
lat_ticks = np.arange(-60, 0, 10)
title = ['a) sst','b) u-wnd','c) v-wnd','d) ws', 'e) curl', 'f) SLP', 'g) nhf']
fontsize = 6

clevs_curl = np.linspace(-1, 1, 21)
clevs_curl_label = np.linspace(-1.5, 1.5, 7)

variables = ['sst','u10','v10','ws','svo','msl','nhf']

cblabel = ['sst','u10','v10','ws','svo','msl','nhf']

mean_levels = [
                np.linspace(0,24,25), np.linspace(-6,6,13),
                np.linspace(-6,6,13), np.linspace(0,8,9),
                np.linspace(-1,1,21), np.linspace(990,1020,16),
                np.linspace(-60,60,13)
                ]

trend_levels1 = [np.linspace(-0.8,0.8,9), np.linspace(-0.8,0.8,9),
                np.linspace(-0.8,0.8,9), np.linspace(-0.8,0.8,9),
                np.linspace(-.5,.5,11), np.linspace(-2,2,9),
                np.linspace(-15,15,7)
                ]

trend_levels2 = [np.linspace(-1.6,1.6,17), np.linspace(-1.6,1.6,17),
                np.linspace(-1.6,1.6,17), np.linspace(-1.6,1.6,17),
                np.linspace(-1,1,21), np.linspace(-4,4,17),
                np.linspace(-30,30,13)
                ]

levels = [mean_levels, trend_levels1, trend_levels2]

cmap_mean = [cm.cm.thermal, plt.cm.Spectral_r, plt.cm.Spectral_r,
            plt.cm.Spectral_r, plt.cm.Spectral_r, plt.cm.Spectral_r,
            plt.cm.Spectral_r]

cmap_trend = [plt.cm.RdBu_r, plt.cm.PuOr_r, plt.cm.PuOr_r,
            plt.cm.PuOr_r, plt.cm.PuOr_r, plt.cm.PuOr_r,
            plt.cm.PuOr_r]

cmap = [cmap_mean, cmap_trend, cmap_trend]

# figure
plt.close('all')
fig = plt.figure(figsize=figsize)
plt.clf()

for i in range(len(variables)):                  # esto recorre las filas, es decir las variables

    for j in range(3):                          # esto recore las columnas, es decir i=0 media, i=1 tend primer per, i=2 tend seg periodo

        # load data
        if j==0:
            data = xr.open_dataset(directory + files[j][i])
            data_var = data[variables[i]].values.squeeze()
            lon = data.longitude.values
            lat = data.latitude.values

        else:
            data = xr.open_dataset(directory + files[j][i])
            data_var = data[variables[i]].values.squeeze()*120
            lon = data.longitude.values
            lat = data.latitude.values

        ax = plt.axes(pos[i][j], projection=ccrs.Mercator())
        ax.set_extent(lonlatbox_list, crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', color='black', linewidths=0.2, zorder=5)
        ax.add_feature(ccrs.cartopy.feature.LAND, edgecolor='k', color='white', zorder=4)

        if j==0 and i<len(variables)-1:         # only ylabels
            ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
            ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
            ax.set_xticklabels([])
            lat_formatter = LatitudeFormatter()
            ax.yaxis.set_major_formatter(lat_formatter)

        elif j>0 and i==len(variables)-1:         # only xlabels
            ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
            ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
            ax.set_yticklabels([])
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            ax.xaxis.set_major_formatter(lon_formatter)

        elif j==0 and i==len(variables)-1:         # xlabels and ylabels
            ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
            ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
            lon_formatter = LongitudeFormatter(zero_direction_label=True)
            lat_formatter = LatitudeFormatter()
            ax.xaxis.set_major_formatter(lon_formatter)
            ax.yaxis.set_major_formatter(lat_formatter)

        else:
            ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
            ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        ax.tick_params('both', labelsize=fontsize)
        ax.set_aspect('equal', 'box')

        if j==0:
            ax.set_title(title[i], fontsize=fontsize, loc='left')

        if variables[i] == 'svo':
            sigma = 1.5 # this depends on how noisy your data is, play with it!
            mean_curl = gaussian_filter(data_var, sigma)
            cvar = plt.contourf(lon, lat, mean_curl*1e5, levels[j][i], cmap=cmap[j][i], transform=ccrs.PlateCarree(), extend='both')

        elif variables[i] == 'msl':
            cvar = plt.contourf(lon, lat, data_var/100, levels[j][i], cmap=cmap[j][i], transform=ccrs.PlateCarree(), extend='both')

        else:
            cvar = plt.contourf(lon, lat, data_var, levels[j][i], cmap=cmap[j][i], transform=ccrs.PlateCarree(), extend='both')

        cbati = plt.contour(blon, blat, data_bati, [-200], colors='gray', linewidths=.25, linestyles='solid',
                transform=ccrs.PlateCarree())

        # cvar_line = plt.contour(lon, lat, data_var, 5,colors='k',
        #     linewidths=.25, transform=ccrs.PlateCarree(), zorder=1, alpha=0.8)
        # plt.clabel(cvar_line, inline=1, inline_spacing=-2, fmt='%2.1f', fontsize=4, colors='k')

        # ax.plot(saf_x, saf_y, color='darkblue', linestyle='-', linewidth=0.4, transform=ccrs.PlateCarree())

        if (variables[i]=='u10' or variables[i]=='v10' or variables[i]=='ws') and j!=0:

            # load u.v data
            data = xr.open_dataset(directory + meanfiles_uv[j])
            mean_u = data.u10.values.squeeze()
            mean_v = data.v10.values.squeeze()
            lon_wind = data.longitude.values
            lat_wind = data.latitude.values

            n = 20
            qvr = ax.quiver(lon_wind[::1*n], lat_wind[::1*n], mean_u[::1*n,::1*n], mean_v[::1*n,::1*n], units='xy',
    	           scale=7e-6, headaxislength=3.5, transform=ccrs.PlateCarree(), color='k', alpha=0.6)
            ax.quiverkey(qvr, 0.75, 1.05, 5, '5 m s$^{-1}$', labelpos='E', coordinates='axes', color='k', fontproperties={'size':6})

        #if j==0 or j==2:
        cax = fig.add_axes(pos_cb[i][j])
        cb = fig.colorbar(cvar, orientation='vertical', cax=cax)

        if j==0 or j==2:                        # only labels in first and last column
            cb.ax.set_ylabel('var [ ]', fontsize=fontsize)

        cb.ax.tick_params(labelsize=fontsize)

fig.savefig('/home/daniu/Documentos/figuras/fig_paper_breakpoint', dpi=300, bbox_inches='tight')
fig.savefig('/home/daniu/Documentos/figuras/fig_paper_breakpoint' + '.pdf', bbox_inches='tight')
