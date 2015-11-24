# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:53:52 2015

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

datos = np.loadtxt('DR9Q.dat',usecols = (79,80,81,82))
datos = datos / 3.631
banda_i = datos[:, 0]
error_banda_i = datos[:, 1]
banda_z = datos[:, 2]
error_banda_z = datos[:, 3]

np.random.seed(110)
Nmc = 100000
mean_values_i = np.zeros(Nmc)
for i in range(Nmc):
    r = np.random.normal(0, 1, size=len(banda_i))
    muestra_i = banda_i + error_banda_i * r
    mean_values_i[i] = np.mean(muestra_i)
    
Nmc = 100000
mean_values_z = np.zeros(Nmc)
for i in range(Nmc):
    r = np.random.normal(0, 1, size=len(banda_z))
    muestra_z = banda_z + error_banda_z * r
    mean_values_z[i] = np.mean(muestra_z)
    
fig1 = plt.figure(1)
fig1.clf()
plt.hist(mean_values_i, bins=30)
plt.axvline(np.mean(muestra_i), color='r')
plt.xlabel('Muestra z')
plt.ylabel('Flujo en banda i [1e-6 Jy]')
plt.legend('H','r')
plt.show()

mean_values_i2 = np.sort(mean_values_i)
limite_bajo = mean_values_i2[int(Nmc * 0.05)]
limite_alto = mean_values_i2[int(Nmc * 0.95)]
print "El intervalo de confianza al 95% para banda_i es: [{}:{}]".format(limite_bajo, limite_alto)

fig2 = plt.figure(2)
fig2.clf()
plt.hist(mean_values_z, bins=30)
plt.axvline(np.mean(muestra_z), color='r')
plt.xlabel('Muestra z')
plt.ylabel('Flujo en banda z [1e-6 Jy]')
plt.legend('H','r')
plt.show()

mean_values_z2 = np.sort(mean_values_z)
limite_bajo = mean_values_z2[int(Nmc * 0.05)]
limite_alto = mean_values_z2[int(Nmc * 0.95)]
print "El intervalo de confianza al 95% para banda_z es: [{}:{}]".format(limite_bajo, limite_alto)