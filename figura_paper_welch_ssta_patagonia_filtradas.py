"""
En esta rutina calculo los espectros de welch de
las cajas de SSTa
Dani Risaro
Octubre 2019
"""
import warnings
import os
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy import signal                                                        # Libreria de señales
from scipy.stats import chi2, f                                                 # Test de ruido y barra de error espectro
from scipy import stats
from scipy.integrate import simps

os.chdir('/home/daniu/Documentos/rutinas/')
import PSD_rednoise

archivo = '/home/daniu/Documentos/datos_reynolds/output/anom_sst_monthly_reynolds_1982-2017_swa.nc'

time = pd.date_range('1982-01-01', '2017-12-31', freq='MS')
data = xr.open_dataset(archivo)

lat_north_n = -42.125;       lat_north_c = -47.375;           lat_north_s = -49.625;
lat_south_n = -45.875;       lat_south_c = -48.125;           lat_south_s = -52.625;
lon_west_n = 360-65.125;     lon_west_c = 360-(64.875-1);         lon_west_s = 360-66.375;
lon_east_n = 360-60.125;     lon_east_c = 360-(63.125-1);         lon_east_s = 360-61.875;

lat_north = np.stack((lat_north_n, lat_north_c, lat_north_s), axis=0)
lat_south = np.stack((lat_south_n, lat_south_c, lat_south_s), axis=0)
lon_west = np.stack((lon_west_n, lon_west_c, lon_west_s), axis=0)
lon_east = np.stack((lon_east_n, lon_east_c, lon_east_s), axis=0)

cajas = ['NPS','CPS','SPS']
window = 36

figname = 'welch_variance_preserving_PSD_ssta_PS'
plt.close('all')
figprops = dict(figsize=(3, 2.7), dpi=72)
fig = plt.figure(**figprops)
color = ['red','grey','blue']
ax = plt.axes([0.1, 0.1, 0.85, 0.8])
bx = ax.twiny()

for i in range(3):

    a = data.sel(lat=slice(lat_south[i], lat_north[i]),
                    lon=slice(lon_west[i], lon_east[i])).mean(dim=('lat','lon')).rolling(time=window,
                    center='True').mean().dropna(dim='time',how='any')

    dat = a.sst.values
    N = dat.size
    desvio = dat.std()

    nperseg = N/2                                         # Longitud del segmento
    noverlap = nperseg*(3/4)                              # Solapamiento
    S = nperseg - noverlap                                # Desplazamiento entre solapamientos
    P = int((N - nperseg) / S + 1)                        # Cantidad de muestras - segmentos
    probability = 0.975                                   # Probabilidad
    alfa = 1-probability                                  # Significancia
    v = 2*P                                               # Grados de libertad
    c = chi2.ppf([1-alfa/2, alfa/2], v)
    c = v/c

    freq, Pxx = signal.welch(dat, fs=1, window="hanning",
                nperseg=int(nperseg), noverlap=int(noverlap),
                scaling='density')

    freq_noise, Pxx_noise = PSD_rednoise.psd_rednoise(dat)

    Pxxc_lower = Pxx*c[0]
    Pxxc_upper = Pxx*c[1]

    f = freq[1::]
    Pxx = Pxx[1::]

    periods = np.array([16,8,4,2,1,0.5])
    ticks = np.log10(1/periods)

    f_n = freq_noise[1::]
    Pxx_n = Pxx_noise[1::]
    ax.semilogx(f, Pxx*f, color=color[i], label=cajas[i], lw=.5)
    ax.semilogx(f_n, Pxx_n*f_n, color=color[i], linestyle='--', lw=.5)
#    ax.fill_between(freq, Pxxc_lower*freq, Pxxc_upper*freq, color=color[i], alpha=0.05)
    ax.set_xlim([1/192, 1/2])

    ax.legend(loc='upper right', fontsize=6)
    ax.set_xlabel('freq [cycles mo$^{-1}$]', fontsize=6)
    ax.set_ylabel('f*PSD [$^{\circ}$C$^2$]', fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_ylim([0,0.035])

    periods = np.array([16,8,4,2,1,0.5])
    ticks = np.log10(1/periods)
    bx.set_xlim(ax.get_xlim())
    bx.set_xticks(ticks)
    bx.set_xticklabels(periods)
    bx.set_xlabel('Period [years]', fontsize=6)
    bx.tick_params(labelsize=6)
    area = simps((Pxx)[0:6], dx=f[1]-f[0])
    print('Var tot', np.round(desvio**2,3))
    print('Var first peak:', np.round(area,3))
    print('% first peak on total:', area/(desvio**2)*100 )


fig.savefig('/home/daniu/Documentos/figuras/' + figname + '.pdf', bbox_inches='tight')
fig.savefig('/home/daniu/Documentos/figuras/' + figname, dpi=300, bbox_inches='tight')
