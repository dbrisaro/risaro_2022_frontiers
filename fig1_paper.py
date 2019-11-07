"""
Hago nuevamente la figura 1 del paper siguiendo las sugerencias
del Lic Lois

Dani Risaro
Noviembre 2019
"""
import sys
import warnings
import cmocean as cm
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import xarray as xr
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.patches import Polygon
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

sys.path.insert(0, '/home/daniu/Documentos/tesis_daniu_modulo')
import cargar_datos

lat_s = -60
lat_n = -10
lon_w = 270-360
lon_e = 330-360

oct = pd.date_range('01-01-1998','12-01-2017', freq='AS-OCT')
feb = pd.date_range('01-01-1998','12-01-2017', freq='AS-JAN')

#-- cargo datos de chlorofila
archivo = '/home/daniu/Documentos/datos_chl_calibrados_9km/chl_calibrada_interpolada_70-50W_55-23S_1998_2017_gridded.nc'
chl_ocx = xr.open_dataset(archivo)
chl_subset = chl_ocx.sel(latitude=slice(lat_n,lat_s), longitude=slice(lon_w,lon_e), time=oct).mean(dim='time')

#-- cargo datos de mdt
archivo = '/home/daniu/Documentos/datos_MDT/dataset-mdt-cnes-cls13-global.nc'
mdt = xr.open_dataset(archivo)
mdt_subset = mdt.sel(lat=slice(lat_s,lat_n), lon=slice(lon_w,lon_e))

print('Datos de chl y mdt cargados')

#-- cargo datos de SST
archivo = '/home/daniu/Documentos/datos_reynolds/output/sst_mean.nc'

lat_s = -60
lat_n = -10
lon_w = 270
lon_e = 330

time = pd.date_range('1982-01-01', '2017-12-31', freq='MS')
#-- sst media
data_sst = xr.open_dataset(archivo)
data_sst = data_sst.sel(lon=slice(lon_w, lon_e), lat=slice(lat_s, lat_n))
mean_sst = data_sst.sst.squeeze()
del data_sst

print('Datos de SST cargados')

#-- nombre del area y el periodo
area = cargar_datos.name(lat_s, lat_n, lon_w, lon_e)
anio = cargar_datos.periodo(1982, 2017)

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


#-- cargo posicion de los fondeos
lat = []
lon = []

archivo = '/home/daniu/Documentos/datos_boya/boya_2005/datos_boya_1hora.csv'
boya_hor = pd.read_csv(archivo, header=[0],index_col=0, delimiter='\t')

lat.append(boya_hor.lat.values[0])
lon.append(boya_hor.lon.values[0])

archivo = '/home/daniu/Documentos/datos_boya/boya_2006_corto/datos_boya_1hora.csv'
boya_hor = pd.read_csv(archivo, header=[0],index_col=0, delimiter='\t')

lat.append(boya_hor.lat.values[0])
lon.append(boya_hor.lon.values[0])

archivo = '/home/daniu/Documentos/datos_boya/boya_2006/datos_boya_1hora.csv'
boya_hor = pd.read_csv(archivo, header=[0],index_col=0, delimiter='\t')

lat.append(boya_hor.lat.values[0])
lon.append(boya_hor.lon.values[0])

archivo = '/home/daniu/Documentos/datos_boya/boya_2015/datos_boya_1hora.csv'
boya_hor = pd.read_csv(archivo, header=[0],index_col=0, delimiter='\t')

lat.append(boya_hor.lat.values[0])
lon.append(boya_hor.lon.values[0])

archivo = '/home/daniu/Documentos/datos_boya/boya_2016/datos_boya_1hora.csv'
boya_hor = pd.read_csv(archivo, header=[0],index_col=0, delimiter='\t')

lat.append(boya_hor.lat.values[0])
lon.append(boya_hor.lon.values[0])

lat_xy = []
lon_xy = []
color_xy = 'rrrrr'
color_xy = 'rgbkm'
color_xy = 'ggggg'
label = ['','C','B','A','D']
for ilat, ilon in zip(lat,lon):
    x, y = ccrs.Mercator().transform_point(ilon, ilat, ccrs.PlateCarree())
    lon_xy.append(x)
    lat_xy.append(y)

xo = [-1.5, -1.5, -1.5, -1.5, -2]
yo = [0, 0, 0, 0, -2]

#-- detalles del mapa
nombre_salida = '/home/daniu/Documentos/figuras/figura_1_paper'
clevs_chl = np.linspace(-1.5, 1.0, 41)
clevs_chl = np.linspace(-1.0, 1.0, 41)
clevels_chl = [-1.5, -1., -.5, 0, .5, 1.]
clevels_chl = [-1., -.5, 0, .5, 1.]
ticks_wanted_chl = [0.03, 0.1, 0.3, 1, 3, 10]
ticks_wanted_chl = [0.1, 0.3, 1, 3, 10]

clevels_mdt = np.linspace(-1.2, 1.2, 25)
clevels_mdt = np.array([-1.2, -1.0, -0.8, -0.7, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0, 1.2])

from matplotlib.colors import LinearSegmentedColormap
colors = [(0.33,0.33,0.33)] + [(plt.cm.jet(i)) for i in range(1,256)]
new_map = LinearSegmentedColormap.from_list('new_map', colors, N=256)

lon_chl = chl_subset.longitude.values
lat_chl = chl_subset.latitude.values

lon_mdt = mdt_subset.lon.values
lat_mdt = mdt_subset.lat.values

cmap = plt.get_cmap(cm.cm.thermal)
clevs_sst = np.linspace(0, 25, 26)
clevs_sst_label = np.linspace(0, 25, 6)

lat_sst = mean_sst.lat.values
lon_sst = mean_sst.lon.values

lon_w_zoom = 290
lon_e_zoom = 305
lat_s_zoom = -55
lat_n_zoom = -38

#-- cargo coordenadas de las cajas
lat_north_n = -42.125;      lat_north_c = -47.375;          lat_north_s = -49.625;
lat_south_n = -45.875;      lat_south_c = -48.125;          lat_south_s = -52.625;
lon_west_n = 360-65.125;    lon_west_c = 360-(64.875-1);        lon_west_s = 360-66.375;
lon_east_n = 360-60.125;    lon_east_c = 360-(63.125-1);        lon_east_s = 360-61.875;

lat_north = np.stack((lat_north_n, lat_north_c, lat_north_s), axis=0)
lat_south = np.stack((lat_south_n, lat_south_c, lat_south_s), axis=0)
lon_west = np.stack((lon_west_n, lon_west_c, lon_west_s), axis=0)
lon_east = np.stack((lon_east_n, lon_east_c, lon_east_s), axis=0)

# -- armo la figura
plt.close('all')
figprops = dict(figsize=(4, 3), dpi=72)
fig = plt.figure(**figprops)
plt.clf()
ax = plt.axes([0.04, 0.3, 0.4, 0.5], projection=ccrs.Mercator())
ax.set_extent([lon_w, lon_e, lat_s, lat_n], crs=ccrs.PlateCarree())
ax.coastlines(resolution='50m', color='black', linewidths=0.2)
ax.set_aspect('equal', 'box')
ax.add_feature(ccrs.cartopy.feature.LAND, edgecolor='k', color='white', zorder=2)

csst = ax.contourf(lon_sst, lat_sst, mean_sst.values, clevs_sst, cmap=cmap, transform=ccrs.PlateCarree(), extend='both')
cmdt_line = ax.contour(lon_mdt, lat_mdt, mdt_subset.mdt.values.squeeze(), clevels_mdt, colors='k',
    zorder=10, linewidths=.15, transform=ccrs.PlateCarree())
#plt.clabel(cmdt_line, inline=True, inline_spacing=0.4, fmt = '%2.1f', fontsize=5, colors='k')

cbati = ax.contour(blon, blat, data_bati, [-200], colors='gray', linestyles='solid', linewidths=.15, transform=ccrs.PlateCarree())
ax.set_title('a)',loc='left',fontsize=5)

ax.set_xticks([-90, -70, -50, -30], crs=ccrs.PlateCarree())
ax.set_yticks([-60, -50, -40, -30, -20, -10], crs=ccrs.PlateCarree())

lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.tick_params('both', labelsize=5)

cax = fig.add_axes([0.06, 0.2, 0.35, 0.015])
cb = fig.colorbar(csst, orientation='horizontal', cax=cax)
cb.ax.set_xlabel('Temperature [$^{\circ}$C]', fontsize=5)
cb.ax.tick_params(labelsize=5)

x1, y1 = ccrs.Mercator().transform_point(lon_w_zoom, lat_s_zoom, ccrs.PlateCarree())
x2, y2 = ccrs.Mercator().transform_point(lon_w_zoom, lat_n_zoom, ccrs.PlateCarree())
x3, y3 = ccrs.Mercator().transform_point(lon_e_zoom, lat_n_zoom, ccrs.PlateCarree())
x4, y4 = ccrs.Mercator().transform_point(lon_e_zoom, lat_s_zoom, ccrs.PlateCarree())
poly_n = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none', edgecolor='red', linewidth=.5)
ax.add_patch(poly_n)

bx = plt.axes([0.45, 0.15, 0.47, 0.8], projection=ccrs.Mercator())
bx.set_extent([lon_w_zoom, lon_e_zoom, lat_s_zoom, lat_n_zoom], crs=ccrs.PlateCarree())
bx.coastlines(resolution='10m', color='black', linewidths=0.4)
bx.set_aspect('equal', 'box')

for spine in bx.spines.values():
    spine.set_edgecolor('red')
    
cchl = bx.contourf(lon_chl, lat_chl, np.log10(chl_subset.chl.values), clevs_chl, cmap=new_map, transform=ccrs.PlateCarree(), extend='both')
cchl.cmap.set_under('darkblue')

cbati = bx.contour(blon, blat, data_bati, [-200], colors='gray',linestyles='solid', linewidths=.15, transform=ccrs.PlateCarree())
bx.set_title('b)',loc='left',fontsize=5)
bx.set_xticks([-70, -65, -60, -55], crs=ccrs.PlateCarree())
bx.set_yticks([-55, -50, -45, -40], crs=ccrs.PlateCarree())

lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
bx.xaxis.set_major_formatter(lon_formatter)
bx.yaxis.set_major_formatter(lat_formatter)
bx.tick_params('both', labelsize=5)

for ilabel, x, y, xx, yy in zip(label, lon_xy, lat_xy, xo, yo):
    plt.annotate(ilabel, xy=(x, y), xytext = (xx, yy),
    textcoords='offset points', ha='right', va='bottom', fontsize=5, color='navy')
bx.scatter(lon_xy, lat_xy, s=7, color=color_xy, zorder=6)
#
# for ii in range(0,3):
#     x1, y1 = ccrs.Mercator().transform_point(lon_west[ii],lat_south[ii], ccrs.PlateCarree())
#     x2, y2 = ccrs.Mercator().transform_point(lon_west[ii],lat_north[ii], ccrs.PlateCarree())
#     x3, y3 = ccrs.Mercator().transform_point(lon_east[ii],lat_north[ii], ccrs.PlateCarree())
#     x4, y4 = ccrs.Mercator().transform_point(lon_east[ii],lat_south[ii], ccrs.PlateCarree())
#     poly_n = Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)], facecolor='none', edgecolor='k', linewidth=.5)
#     bx.add_patch(poly_n)


cbx = fig.add_axes([0.49, 0.04, 0.37, 0.015])
cb = fig.colorbar(cchl, orientation='horizontal', cax=cbx)
cb.ax.set_xlabel('Chlorophyll a [mg m$^{-3}$]', fontsize=5)
cb.set_ticks(clevels_chl)
cb.set_ticklabels(ticks_wanted_chl)
cb.ax.tick_params(labelsize=5)

fig.savefig(nombre_salida, dpi=300, bbox_inches='tight')
fig.savefig(nombre_salida + '.pdf', bbox_inches='tight')
plt.close('all')
