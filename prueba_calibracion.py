
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


wnd_speed = np.linspace(0,30,101)

wnd_speed_calibr = ref_10m(wnd_speed, z)

plt.plot(wnd_speed, wnd_speed_calibr)
plt.plot(wnd_speed, wnd_speed)
plt.show()
