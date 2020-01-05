"""
En esta rutina armamos la tabla 2 del paper donde se comparan los
datos de viento de la boya contra los datos de CCMPv2 y ERA- interim.
En este caso recalibramos los datos de la boya a 10m

Dani Risaro
Diciembre 2019
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def reject_outliers(data, m=2):
    """
    Función que remueve outliers del conjunto de datos

    Parámetros de entrada:
    data: dataframe de diferencias
    m: int. Cantidad de desvíos std

    Salida:
    dataframe without outliers
    """
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def remove_outliers_diference(data_x, data_y):
    """
    Esta funcion remueve los outliers de un par de set de datos
    a partir de las diferencias entre ellos (entre dfA y dfB).
    Cuando las difs exceden 2 std, esos datos son removidos

    Parámetros de entrada:
    data_x: dataframe A
    data_y: dataframe B

    Salida:

    """
    dif = data_x - data_y
    outliers = reject_outliers(dif, m=2)
    data_x_without_out = data_x[outliers.index]
    data_y_without_out = data_y[outliers.index]

    return data_x_without_out, data_y_without_out

def ref_10m(wnd_speed, z):              # ver supplement de atlas 2011 para la parametrizacion de esta funcion
    c0 = 3.7
    c1 = 1.165
    a = 0.032
    k = 0.40
    g = 9.81
    zeta_cero = z * np.exp(-c0 + c1*np.log(a*k**2*wnd_speed**2/g*z))
    wnd_speed_10m = (np.log(10/zeta_cero)/np.log(z/zeta_cero)) * wnd_speed

    return wnd_speed_10m

z = 4       # la boya mide a 4m


# table to fill
sites = ['A' ,'B', 'C1', 'C2', 'D']
parameters = ['Latitude (S)',
            'Longitude (W)',
            'Date',
            'Period of data (days)',
            'Mean wsp buoy (m/s)',
            'Std wsp buoy (m/s)',
            'R (CCMPv2, buoy)',
            'R (ERA-Interim, buoy)',
            'R (CCMPv2, buoy) without outliers',
            'R (ERA-Interim, buoy) without outliers']

df = pd.DataFrame(index=sites, columns=parameters)

# organizo la iteracion

directory = '/home/daniu/Documentos/datos_boya/'

buoyfiles = ['boya_2015/datos_boya_1hora.csv',
            'boya_2006/datos_boya_1hora.csv',
            'boya_2006_corto/datos_boya_1hora.csv',
            'boya_2005/datos_boya_1hora.csv',
            'boya_2016/datos_boya_1hora.csv']

ccmpfiles = ['boya_2015/datos_ccmp_6horas.csv',
            'boya_2006/datos_ccmp_6horas.csv',
            'boya_2006_corto/datos_ccmp_6horas.csv',
            'boya_2005/datos_ccmp_6horas.csv',
            'boya_2016/datos_ccmp_6horas.csv']

eraifiles = ['boya_2015/datos_era_interim_6horas.csv',
            'boya_2006/datos_era_interim_6horas.csv',
            'boya_2006_corto/datos_era_interim_6horas.csv',
            'boya_2005/datos_era_interim_6horas.csv',
            'boya_2016/datos_era_interim_6horas.csv']


for i in range(5):

    ibuoyfile = directory + buoyfiles[i]
    iccmpfile = directory + ccmpfiles[i]
    ieraifile = directory + eraifiles[i]

    ccmpv2 = pd.read_csv(iccmpfile, index_col=0)
    erai = pd.read_csv(ieraifile, index_col=0)
    buoy_hor = pd.read_csv(ibuoyfile, header=[0], index_col=0, delimiter='\t')
    boya = buoy_hor.iloc[::6]

    if i==4:
        boya = buoy_hor.iloc[3::6]

    # comparison with CCMP
    x_boya = boya['int']
    x_boya = ref_10m(boya['int'], z)
    y_ccmp = ccmpv2['speed']
    x_boya.index = y_ccmp.index     # reindex just in case they are not in the same format

    x_boya_without_out, y_ccmp_without_out = remove_outliers_diference(x_boya, y_ccmp)

    R_with_out_ccmp = x_boya.corr(y_ccmp)
    R_without_out_ccmp = x_boya_without_out.corr(y_ccmp_without_out)

    # comparison with ERA i
    x_boya = boya['int']
    x_boya = ref_10m(boya['int'], z)
    y_erai = erai['speed']
    x_boya.index = y_erai.index     # reindex just in case they are not in the same format

    x_boya_without_out, y_erai_without_out = remove_outliers_diference(x_boya, y_erai)

    R_with_out_erai = x_boya.corr(y_erai)
    R_without_out_erai = x_boya_without_out.corr(y_erai_without_out)

    # lat - lon position and length
    pos_lat = boya.lat.values[0]
    pos_lon = boya.lon.values[0]
    ndays = len(buoy_hor)/24

    # date range
    date = buoy_hor.index[0] + ' to ' + buoy_hor.index[-1]

    # mean and std from buoy
    mean_buoy = x_boya.mean()
    std_buoy = x_boya.std()

    # fill the table
    df.loc[sites[i], 'Latitude (S)'] = pos_lat.round(2)
    df.loc[sites[i], 'Longitude (W)'] = pos_lon.round(2)
    df.loc[sites[i], 'Date'] = date
    df.loc[sites[i], 'Period of data (days)'] = int(ndays)
    df.loc[sites[i], 'Mean wsp buoy (m/s)'] = mean_buoy.round(2)
    df.loc[sites[i], 'Std wsp buoy (m/s)'] = std_buoy.round(2)
    df.loc[sites[i], 'R (CCMPv2, buoy)'] = R_with_out_ccmp.round(2)
    df.loc[sites[i], 'R (ERA-Interim, buoy)'] = R_with_out_erai.round(2)
    df.loc[sites[i], 'R (CCMPv2, buoy) without outliers'] = R_without_out_ccmp.round(2)
    df.loc[sites[i], 'R (ERA-Interim, buoy) without outliers'] = R_without_out_erai.round(2)


print(df)
df.to_csv('/home/daniu/Documentos/tablas/tabla_buoy_observations_ref.csv', sep=',')
