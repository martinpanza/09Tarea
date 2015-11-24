# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 17:02:07 2015

@author: splatt
"""

from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from scipy.optimize import (leastsq, curve_fit)

def H(v, d):
    H = v / d
    return H

datos = np.loadtxt('hubble_original.dat')
distancia = datos[:, 0]
velocidad = datos[:, 1]

H0 = np.linspace(0, 24, 24)
fig1 = plt.figure(1)
fig1.clf()
plt.plot(H0, H(velocidad, distancia))
plt.xlabel('H0 (constante de Hubble) [km / s / Mpc]')
plt.ylabel('V/D (velocidad/distancia) [km / s / Mpc]')
plt.show()