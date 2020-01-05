"""
En esta rutina calculo los espectros de welch de
las PC de los EOF del paper

Dani Risaro
Noviembre 2019
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
from eofs.standard import Eof
warnings.filterwarnings('ignore')

os.chdir('/home/daniu/Documentos/rutinas/')
import PSD_rednoise

# load SST date already filtered
archivo = '/home/daniu/Documentos/datos_reynolds/output/filt36_anom_sst_monthly_reynolds_1982-2017_swa_sep_detrended.nc'
data = xr.open_dataset(archivo)
lon_sst = data.lon.values
lat_sst = data.lat.values
time_sst = data.time.values

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

time = data.time.values

PCs = ['PC 1','PC 2','PC 3']

figname = 'welch_variance_preserving_PSD_ssta_PC'
plt.close('all')
figprops = dict(figsize=(3, 2.7), dpi=72)
fig = plt.figure(**figprops)
color = ['indigo','orange','forestgreen']
ax = plt.axes([0.1, 0.1, 0.85, 0.8])
bx = ax.twiny()

for i in range(3):

    dat = pc_sst[:,i]
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
    ax.semilogx(f, Pxx*f, color=color[i], label=PCs[i], lw=.5, marker='.', markersize=2)
    ax.semilogx(f_n, Pxx_n*f_n, color=color[i], linestyle='--', lw=.5)
#    ax.fill_between(freq, Pxxc_lower*freq, Pxxc_upper*freq, color=color[i], alpha=0.05)
    ax.set_xlim([1/192, 1/2])

    ax.legend(loc='upper right', fontsize=6)
    ax.set_xlabel('freq [cycles per month]', fontsize=6)
    ax.set_ylabel('Temperature variance [$^{\circ}$C$^2$]', fontsize=6)
    ax.tick_params(labelsize=6)
    ax.set_ylim([0, .6])

    bx.set_xlim(ax.get_xlim())
    bx.set_xscale('log')
    periods = np.array([.008, .009, .01, .1, .2, .3, .4, .5])
    periods = 1/periods
    bx.set_xticklabels([100, 50, 25, 10, 5, 2])
    bx.set_xlabel('Period [months]', fontsize=6)
    bx.tick_params(labelsize=6)
    area = simps((Pxx)[0:6], dx=f[1]-f[0])
    print('Var tot', np.round(desvio**2,3))
    print('Var first peak:', np.round(area,3))
    print('% first peak on total:', area/(desvio**2)*100 )

    area = simps((Pxx)[5:9], dx=f[1]-f[0])
    print('Var second peak:', np.round(area,3))
    print('% second peak on total:', area/(desvio**2)*100 )
    ax.grid(which='both', axis='x', linestyle='--', lw=.25)


fig.savefig('/home/daniu/Documentos/figuras/' + figname + '.pdf', bbox_inches='tight')
fig.savefig('/home/daniu/Documentos/figuras/' + figname, dpi=300, bbox_inches='tight')
