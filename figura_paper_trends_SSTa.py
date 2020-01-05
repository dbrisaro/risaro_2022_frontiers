"""
Figure 4 paper
Período 1982-2017, OISSTv2 Reynolds

Autora: Daniela Risaro
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
from matplotlib.patches import Polygon
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.ndimage.filters import gaussian_filter

#----------------------------------------------------------------

archivo = '/home/daniu/Documentos/datos_reynolds/output/trend_b_filt36_anom_sst_monthly_reynolds_1982-2017_swa_sep.nc'
data_sst = xr.open_dataset(archivo)

#-- hago un subset de datos tendencias de sst
lat_s = -60
lat_n = -10
lon_w = 250
lon_e = 350

data_trend = data_sst.sel(lon=slice(lon_w, lon_e), lat=slice(lat_s, lat_n))

# area for the plot
lonlatbox_list = [lon_w, lon_e, lat_s, lat_n]
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

#-- detalles del map
cmap = plt.get_cmap(plt.cm.RdBu_r)
nombre_salida = '/home/daniu/Documentos/figuras/figura_paper_linear_trends_ssta_sin_cajas'
clevs_slope = np.linspace(-0.4, 0.4, 33)
clevs_slope_label = np.array([-.4, -.3, -.2, -.1, .1, .2, .3, .4])

data = data_trend.sst.squeeze().values*120
lon = data_trend.lon.values
lat = data_trend.lat.values

umbral = 0.04

data[np.abs(data)<umbral] = np.nan

lon_ticks = np.arange(-110, 0, 20)
lat_ticks = np.arange(-60, 0, 10)

plt.close('all')
fig = plt.figure(figsize=(4,3))
plt.clf()

ax = plt.axes([0.08, 0.05, 0.85, 0.85], projection=ccrs.Mercator())

ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', color='black', linewidths=0.2)
ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params('both', labelsize=6)
ax.set_aspect('equal', 'box')

cslope = plt.contourf(lon, lat, data, clevs_slope, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
cbati = plt.contour(blon, blat, data_bati, [-200], colors='gray', linewidths=.25, linestyles='solid',
    transform=ccrs.PlateCarree())
cslope_line = plt.contour(lon, lat, data, clevs_slope_label, colors='k',
    linewidths=.25, transform=ccrs.PlateCarree())
plt.clabel(cslope_line, inline=1, inline_spacing=-2, fmt='%2.1f', fontsize=4, colors='k')

ax.plot(saf_x, saf_y, color='darkblue', linestyle='-', linewidth=0.65, transform=ccrs.PlateCarree())

cax = fig.add_axes([0.96, 0.13, 0.01, 0.68])

cb = fig.colorbar(cslope, orientation='vertical', cax=cax)
cb.ax.set_ylabel('Linear trend [$^{\circ}$C dec$^{-1}$]', fontsize=6)
cb.ax.tick_params(labelsize=6)


fig.savefig(nombre_salida, dpi=300, bbox_inches='tight')
fig.savefig(nombre_salida + '.pdf', bbox_inches='tight')


lat_north_n = -42.125;      lat_north_c = -47.375;          lat_north_s = -49.625;
lat_south_n = -45.875;      lat_south_c = -48.125;          lat_south_s = -52.625;
lon_west_n = 360-65.125;    lon_west_c = 360-(64.875-1);        lon_west_s = 360-66.375;
lon_east_n = 360-60.125;    lon_east_c = 360-(63.125-1);        lon_east_s = 360-61.875;

lat_north = np.stack((lat_north_n, lat_north_c, lat_north_s), axis=0)
lat_south = np.stack((lat_south_n, lat_south_c, lat_south_s), axis=0)
lon_west = np.stack((lon_west_n, lon_west_c, lon_west_s), axis=0)
lon_east = np.stack((lon_east_n, lon_east_c, lon_east_s), axis=0)

nombre_salida = '/home/daniu/Documentos/figuras/figura_paper_linear_trends_ssta_cajas'

plt.close('all')
fig = plt.figure(figsize=(4,3))
plt.clf()
ax = plt.axes([0.08, 0.05, 0.85, 0.85], projection=ccrs.Mercator())
ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', color='black', linewidths=0.2)

ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())

lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params('both', labelsize=6)

ax.set_aspect('equal', 'box')

cslope = plt.contourf(lon, lat, data, clevs_slope, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
cbati = plt.contour(blon, blat, data_bati, [-200], colors='gray', linewidths=.25, linestyles='solid',
    transform=ccrs.PlateCarree())
cslope_line = plt.contour(lon, lat, data, clevs_slope_label, colors='k',
    linewidths=.25, transform=ccrs.PlateCarree())
plt.clabel(cslope_line, inline=1, inline_spacing=-2, fmt='%2.1f', fontsize=4, colors='k')

ax.plot(saf_x, saf_y, color='darkblue', linestyle='-', linewidth=0.65, transform=ccrs.PlateCarree())

for i in range(3):
    x1, y1 = ccrs.Mercator().transform_point(lon_west[i],lat_south[i], ccrs.PlateCarree())
    x2, y2 = ccrs.Mercator().transform_point(lon_west[i],lat_north[i], ccrs.PlateCarree())
    x3, y3 = ccrs.Mercator().transform_point(lon_east[i],lat_north[i], ccrs.PlateCarree())
    x4, y4 = ccrs.Mercator().transform_point(lon_east[i],lat_south[i], ccrs.PlateCarree())
    poly_n = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none', edgecolor='k', linewidth=.5)
    ax.add_patch(poly_n)

cax = fig.add_axes([0.96, 0.13, 0.01, 0.68])
cb = fig.colorbar(cslope, orientation='vertical', cax=cax)
cb.ax.set_ylabel('Linear trend [$^{\circ}$C dec$^{-1}$]', fontsize=6)
cb.ax.tick_params(labelsize=6)

fig.savefig(nombre_salida, dpi=300, bbox_inches='tight')
fig.savefig(nombre_salida + '.pdf', bbox_inches='tight')

## generamos el histograma
data = data_trend.sst.squeeze().values*120
data[data==0] = np.nan

data[np.abs(data)<umbral] = 0

print(data.shape)
data = data.reshape(80000)
data = data[~np.isnan(data)]

ndat = len(data)

zeros,  = np.where(data==0)
positive, = np.where(data>0)
negative, = np.where(data<0)

warming = positive.size/ndat
cooling = negative.size/ndat
neutral = zeros.size/ndat

print('Warming: ', int(warming*100), '%')
print('Cooling: ', int(cooling*100), '%')
print('Neutral: ', int(neutral*100), '%')
