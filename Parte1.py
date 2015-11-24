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