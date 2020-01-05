"""
carga de indices mensuales
AAO, SAM, ENSO3.4, SOI, PDO, IPO
Dani Risaro
Octubre 2019
"""


def carga_indices():

    import urllib.request
    import re
    from astropy.io import ascii
    import os
    import pandas as pd
    import numpy as np


    # -- AAO
    contents = urllib.request.urlopen('https://www.cpc.ncep.noaa.gov/products/precip/CWlink/daily_ao_index/aao/monthly.aao.index.b79.current.ascii').read()
    s = str(contents,'utf-8')
    data_AAO = ascii.read(s).to_pandas()

    anio_inicial = data_AAO['col1'][0]
    mes_inicial = data_AAO['col2'][0]

    anio_final = data_AAO['col1'][len(data_AAO)-1]
    mes_final = data_AAO['col2'][len(data_AAO)-1]

    fecha_AAO = pd.date_range(str(anio_inicial) + '-' +  str(mes_inicial) + '-01',str(anio_final) + '-' +  str(mes_final) + '-01',freq='MS')

    AAO = pd.DataFrame(data_AAO['col3'].values, index=fecha_AAO, columns=['AAO'])

    # -- NINO 3.4
    contents = urllib.request.urlopen('https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt').read()
    s = str(contents,'utf-8')
    data_NINO34 = ascii.read(s).to_pandas()

    anio_inicial = data_NINO34['YR'][0]
    mes_inicial = data_NINO34['MON'][0]

    anio_final = data_NINO34['YR'][len(data_NINO34)-1]
    mes_final = data_NINO34['MON'][len(data_NINO34)-1]

    fecha_NINO34 = pd.date_range(str(anio_inicial) + '-' +  str(mes_inicial) + '-01',str(anio_final) + '-' +  str(mes_final) + '-01',freq='MS')

    NINO34 = pd.DataFrame(data_NINO34['ANOM'].values, index=fecha_NINO34, columns=['NINO34'])

    # -- SOI
    contents = urllib.request.urlopen('https://climatedataguide.ucar.edu/sites/default/files/darwin.anom_.txt').read()
    s = str(contents,'utf-8')
    data_SOI = ascii.read(s).to_pandas()

    anio_inicial = data_SOI['col1'][0]
    anio_final = data_SOI['col1'][len(data_SOI)-1]

    fecha_SOI = pd.date_range(str(anio_inicial) + '-01-01', str(anio_final) + '-12-01',freq='MS')

    SOI = pd.DataFrame(np.reshape(data_SOI.values[:,1::].T,len(data_SOI)*12,1), index=fecha_SOI, columns=['SOI'])

    # -- PDO
    contents = urllib.request.urlopen('https://www.ncdc.noaa.gov/teleconnections/pdo/data.csv').read()
    s = str(contents,'utf-8')
    data_PDO = ascii.read(s,data_start=2).to_pandas()

    anio_inicial = data_PDO['col1'][0]
    anio_final = data_PDO['col1'][len(data_PDO)-1]

    fecha_PDO = pd.date_range(str(anio_inicial)[0:4] + '-' + str(anio_inicial)[4:6] +'-01', str(anio_final)[0:4] + '-' + str(anio_final)[4:6] +'-01', freq='MS')

    PDO = pd.DataFrame(data_PDO['col2'].values, index=fecha_PDO, columns=['PDO'])

    #-- IPO
    contents = urllib.request.urlopen('https://www.esrl.noaa.gov/psd/data/timeseries/IPOTPI/tpi.timeseries.ersstv5.data').read()
    s = str(contents,'utf-8')
    s = s[11:-473]
    data_IPO = ascii.read(s,delimiter='\s',guess=False).to_pandas()

    anio_inicial = data_IPO.iloc[0,0]
    anio_final = data_IPO.iloc[len(data_IPO)-1,0]

    fecha_IPO = pd.date_range(str(anio_inicial) + '-01-01', str(anio_final) + '-12-01',freq='MS')

    IPO = pd.DataFrame(np.reshape(data_IPO.values[:,1::].T,len(data_IPO)*12,1), index=fecha_IPO, columns=['IPO'])

    IPO[IPO==-99.000] = np.nan

    #-- SAM
    from datetime import datetime
    m = datetime.today().month
    contents = urllib.request.urlopen('http://www.nerc-bas.ac.uk/public/icd/gjma/newsam.1957.2007.txt').read()
    s = str(contents,'utf-8')
    ss = s[77:-72]
    ss = ss.split('\n')

    data = np.genfromtxt(ss)

    data_SAM = pd.DataFrame(data[:,1::], index=data[:,0], columns=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])

    anio_inicial = int(data_SAM.index[0])
    anio_final = int(data_SAM.index[len(data_SAM)-1])

    fecha_SAM = pd.date_range(str(anio_inicial) + '-01-01', str(anio_final) + '-12-01',freq='MS')

    SAM = pd.DataFrame(np.reshape(data_SAM.values.T,len(data_SAM)*12,1), index=fecha_SAM, columns=['SAM'])

    #--NAO
    contents = urllib.request.urlopen('https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/norm.nao.monthly.b5001.current.ascii').read()
    s = str(contents,'utf-8')
    data_NAO = ascii.read(s).to_pandas()

    anio_inicial = data_NAO['col1'][0]
    anio_final = data_NAO['col1'][len(data_NAO)-1]
    mes_final = data_NAO['col2'][len(data_NAO)-1]

    fecha_NAO = pd.date_range(str(anio_inicial) + '-01-01', str(anio_final) + '-' + str(mes_final) + '-01',freq='MS')

    NAO = pd.DataFrame(np.reshape(data_NAO['col3'].values,len(data_NAO),1), index=fecha_NAO, columns=['NAO'])

    return AAO, SAM, NINO34, SOI, PDO, IPO, NAO
