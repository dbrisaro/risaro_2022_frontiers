"""
En esta rutina comparamos los datos de CCMPv2 y era interim
con la boya GEF del año 2006
Dani Risaro
Abril 2019
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%m-%Y')
import sys
import string

sys.path.insert(0, '/home/daniu/Documentos/tesis_daniu_modulo')
import analisis_series_temporales

##-----------------------------------------------------------------
time = pd.date_range('2006-09-25','2007-03-08',freq='D')
time_h = pd.date_range('2006-09-25','2007-03-08 23:30:00',freq='H')
time_d = pd.date_range('2006-09-25','2007-03-08',freq='D')
time_6h = pd.date_range('2006-09-25','2007-03-08 19:00:00',freq='6h')

archivo = '/home/daniu/Documentos/datos_boya/boya_2006/datos_ccmp_6horas.csv'
ccmpv2 = pd.read_csv(archivo, index_col=0)

archivo = '/home/daniu/Documentos/datos_boya/boya_2006/datos_era_interim_6horas.csv'
era_interim = pd.read_csv(archivo, index_col=0)

archivo = '/home/daniu/Documentos/datos_boya/boya_2006/datos_boya_1hora.csv'
boya_hor = pd.read_csv(archivo, header=[0],index_col=0, delimiter='\t')
boya = boya_hor.iloc[::6]

time = pd.date_range('2006-09-25','2007-03-08 19:00:00',freq='6h')

#-- remuevo outliers
nombre = '/home/daniu/Documentos/figuras/figura_paper_comparacion boya_reanalisis'

gs = gridspec.GridSpec(2, 5)
fig = plt.figure(figsize=(7,3.5),dpi=72)


ax1 = plt.axes([0.05, 0.57, 0.65, 0.45])
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax1.get_yticklabels(), fontsize=7)

ax2 = plt.axes([0.05, 0.05, 0.65, 0.45])
plt.setp(ax2.get_xticklabels(), fontsize=7)
plt.setp(ax2.get_yticklabels(), fontsize=7)

ax3 = plt.axes([0.75, 0.57, 0.3, 0.45])
plt.setp(ax3.get_xticklabels(), visible=False)
plt.setp(ax3.get_yticklabels(), visible=False)

ax4 = plt.axes([0.75, 0.05, 0.3, 0.45])
plt.setp(ax4.get_xticklabels(), fontsize=7)
plt.setp(ax4.get_yticklabels(), visible=False)

ax1.plot(time, ccmpv2['speed'], color='darkgrey', label='CCMPv2', lw=0.5)
ax1.plot(time, boya['int'], color='seagreen', label='In-situ data', lw=0.5)
ax1.legend(fontsize=6)
ax1.set_ylim([0,20])
ax1.yaxis.set_ticks_position('both')
ax1.set_title('a)',loc='left',fontsize=7)

ax2.plot(time, era_interim['speed'], color='orangered', label='Era-Interim', lw=0.5)
ax2.plot(time, boya['int'], color='seagreen', label='In-situ data', lw=0.5)
ax2.set_ylim([0,20])
ax2.legend(fontsize=6)
ax2.xaxis.set_major_formatter(myFmt)
ax2.set_xlabel('Time [months]', fontsize=7)
ax2.set_title('b)',loc='left',fontsize=7)


xx = boya['int']
yy = ccmpv2['speed'];   yy.index = xx.index
dif = xx- yy

outliers_dif_ccmpv2 = analisis_series_temporales.reject_outliers(dif, m=2)

yy.index=xx.index

xxx = xx[outliers_dif_ccmpv2.index]
yyy = yy[outliers_dif_ccmpv2.index]

slope, intercept = analisis_series_temporales.regresion_lineal(xxx, yyy)

a = np.round(np.corrcoef(xxx, yyy)[0,1],2)

ax3.scatter(xxx, yyy, color='blue', alpha=0.25, marker='.', lw=0.5, s=20, edgecolors='b')
ax3.set_ylim([0,20])
ax3.set_xlim([0,20])
ax3.set_xticks([0,5,10,15,20])
ax3.annotate('R: {:.2f}'.format(a), xy=(0.07,0.85), xycoords='axes fraction', fontsize=7)
ax3.plot([0, 20], [0, 20], 'k--', zorder=1, color='orange', lw=0.6)
ax3.plot(xxx, slope*xxx + intercept, 'k-', zorder=1, color='m', lw=0.6)
ax3.set_ylabel('CCMPv2', fontsize=7)
ax3.set_title('c)',loc='left',fontsize=7)

xx = boya['int']
yy = era_interim['speed'];   yy.index = xx.index

dif = xx- yy

outliers_dif_era = analisis_series_temporales.reject_outliers(dif, m=2)

xxx = xx[outliers_dif_era.index]
yyy = yy[outliers_dif_era.index]

slope, intercept = analisis_series_temporales.regresion_lineal(xxx, yyy)

b = np.round(np.corrcoef(xxx, yyy)[0,1],2)

ax4.scatter(xxx, yyy, color='blue', alpha=0.25, marker='.', lw=0.5, s=20, edgecolors='b')
ax4.annotate('R: {:.2f}'.format(b), xy=(0.07,0.85), xycoords='axes fraction', fontsize=7)
ax4.set_ylim([0,20])
ax4.set_xticks([0,5,10,15,20])
ax4.set_xlim([0,20])
ax4.plot([0, 20], [0, 20], 'k--', zorder=1, color='orange', lw=0.6)
ax4.plot(xxx, slope*xxx + intercept, 'k-', zorder=1, color='m', lw=0.6)
ax4.set_xlabel('In-situ', fontsize=7)
ax4.set_ylabel('Era-Interim', fontsize=7)
ax4.set_title('d)',loc='left',fontsize=7)

ax1.grid(linestyle='--', lw=0.3)
ax2.grid(linestyle='--', lw=0.3)
ax3.grid(linestyle='--', lw=0.3)
ax4.grid(linestyle='--', lw=0.3)

ax = plt.axes([0.05, 0.05, 0.95, 0.95], frameon=False)
ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
ax.grid(False)
ax.set_ylabel('Wind speed [m s$^{-1}$]', fontsize=7)
fig.tight_layout()

fig.savefig(nombre, dpi=300, bbox_inches='tight')
fig.savefig(nombre + '.pdf', bbox_inches='tight')
