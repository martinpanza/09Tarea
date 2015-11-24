# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:33:35 2015

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

datos = np.loadtxt('SNIa*.dat')

distancia = datos[:, 0]
velocidad = datos[:, 1]

H0 = np.linspace(0, 36, 36)

np.random.seed(111)
muestra = H(velocidad, distancia)

N = len(muestra)
Nboot = 100000
mean_values = np.zeros(Nboot)

for i in range(Nboot):
    s = np.random.randint(low=0, high=N, size=N)
    mean_values[i] = np.mean(muestra[s])

fig2 = plt.figure(2)
fig2.clf()
plt.hist(mean_values, bins=30)
plt.axvline(np.mean(muestra), color='r')
plt.xlabel('H0 (constante de Hubble) [km / s / Mpc]')
plt.ylabel('V/D (velocidad/distancia) [km / s / Mpc]')
plt.legend('H','r')
plt.show()

mean_values = np.sort(mean_values)
limite_bajo = mean_values[int(Nboot * 0.05)]
limite_alto = mean_values[int(Nboot * 0.95)]
print "El intervalo de confianza al 95% es: [{}:{}]".format(limite_bajo, limite_alto)