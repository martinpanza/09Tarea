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