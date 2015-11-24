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
#datos=np.genfromtxt('hubble_original.dat')
distancia = datos[:, 0]
velocidad = datos[:, 1]