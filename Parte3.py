# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:53:52 2015

@author: splatt
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def H(v, d):
    H = v / d
    return H

datos = np.loadtxt('DR9Q.dat', usecols = (79,80,81,82))
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
plt.axvline(np.mean(muestra_i), color='r', label='flujo i estimado')
plt.xlabel('Muestra i')
plt.ylabel('Flujo en banda i [1e-6 Jy]')
plt.legend(shadow=True, fancybox=True)
plt.show()

mean_values_i2 = np.sort(mean_values_i)
limite_bajo = mean_values_i2[int(Nmc * 0.025)]
limite_alto = mean_values_i2[int(Nmc * 0.975)]
print "Flujo i estimado = ", np.mean(muestra_i)
print "El intervalo de confianza al 95% para banda_i es: [{}:{}]".format(limite_bajo, limite_alto)

fig2 = plt.figure(2)
fig2.clf()
plt.hist(mean_values_z, bins=30)
plt.axvline(np.mean(muestra_z), color='r', label='flujo z estimado')
plt.xlabel('Muestra z')
plt.ylabel('Flujo en banda z [1e-6 Jy]')
plt.legend(shadow=True, fancybox=True)
plt.show()

mean_values_z2 = np.sort(mean_values_z)
limite_bajo = mean_values_z2[int(Nmc * 0.025)]
limite_alto = mean_values_z2[int(Nmc * 0.975)]
print "Flujo z estimado = ", np.mean(muestra_z)
print "El intervalo de confianza al 95% para banda_z es: [{}:{}]".format(limite_bajo, limite_alto)

fig2 = plt.figure(3)
fig2.clf()
plt.plot(np.polyfit(mean_values_i,mean_values_z,1))
plt.xlabel('Flujo en banda i [1e-6 Jy]')
plt.ylabel('Flujo en banda z [1e-6 Jy]')
plt.show()
coef=np.polyfit(mean_values_i,mean_values_z,1)
print "Coeficientes de polyfit = ", coef[0], coef [1]
