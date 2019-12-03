"""
En esta rutina ploteo las series temporales de PC2 de SSTa,
ENSO 3.4 y PDO en el SWA y SEOP

Dani Risaro
Diciembre 2019
"""

import warnings
import os
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from eofs.standard import Eof
warnings.filterwarnings('ignore')

os.chdir('/media/daniu/Seagate Expansion Drive/Documentos_DELL_home/rutinas/')
from edof_mpc_py import meanun
from indices_climaticos_mensuales import carga_indices


time_tot = pd.date_range('1982-01-01','2017-12-31', freq='MS')

# load index data
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

ind_df_filt = ind_df.rolling(window=36, center=True).mean().dropna()
ONI = ind_df['ENSO 3.4'].rolling(window=3, center=True).mean().dropna()

# load SST date already filtered
archivo = '/media/daniu/Seagate Expansion Drive/Documentos_DELL_home/datos_reynolds/output/filt36_anom_sst_monthly_reynolds_1982-2017_swa_sep_detrended.nc'
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

PC_df = pd.DataFrame(pc_sst, index=ind_df_filt.index[0:-1], columns=PCs)
PC_df.loc['2016-07-01'] = PC_df.loc['2016-06-01']

oni = ONI.loc['1983-07-01':'2016-07-01']
oni = (oni - oni.mean())/oni.std()
pdo = (ind_df_filt['PDO'] - ind_df_filt['PDO'].mean())/ind_df_filt['PDO'].std()
enso = (ind_df_filt['ENSO 3.4'] - ind_df_filt['ENSO 3.4'].mean())/ind_df_filt['ENSO 3.4'].std()
pc2 = (PC_df['PC 2'] - PC_df['PC 2'].mean())/PC_df['PC 2'].std()

time = enso.index.values
idx, = np.where(np.abs(oni)>1)
idx = list(idx)
from operator import itemgetter
from itertools import *
groups = []
for k, g in groupby(enumerate(idx), lambda x: x[0]-x[1]):
    groups.append(list(map(itemgetter(1), g)))

figname = 'pdo-pc2-enso'
figprops = dict(figsize=(5, 2.7), dpi=72)
fig = plt.figure(**figprops)
ax = plt.axes([0.05, 0.05, 0.9, 0.9])
ax.plot(pdo, label='PDO', lw=.5, color='k', linestyle='--')
ax.plot(enso, label='ENSO 3.4', lw=.5, color='saddlebrown', linestyle='--')
ax.plot(pc2, label='PC2', lw=.5, color='b')
ax.axhline(y=0, lw=.5, color='k')
for g in groups:
    if len(g)>3:
        xi = g[0]
        xf = g[-1]
        ti = time[xi]
        tf = time[xf]
        print(ti, tf)
        if oni[g[0]]>0:
            ax.axvspan(ti, tf, facecolor='lightsalmon', alpha=0.15)
        else:
            ax.axvspan(ti, tf, facecolor='lightblue', alpha=0.15)

ax.legend(fontsize=6)
ax.tick_params('both', labelsize=6)
ax.set_xlabel('Time [Years]', fontsize=6)
ax.set_ylim([-2.5, 2.5])
fig.savefig('/media/daniu/Seagate Expansion Drive/Documentos_DELL_home/figuras/' + figname + '.pdf', bbox_inches='tight')
fig.savefig('/media/daniu/Seagate Expansion Drive/Documentos_DELL_home/figuras/' + figname, dpi=300, bbox_inches='tight')
