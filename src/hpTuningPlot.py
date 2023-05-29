#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:45:43 2023

@author: fergushathorn
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#%%

hp_data = pd.read_csv('hpTuning.csv', 
                      sep = '\t', 
                      names=['c1', 'c2', 'w', 'a', 'b', 'c'])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

axis_data = ['c1', 'c2', 'c']

# Make data.
X = np.array(hp_data[axis_data[0]])
Y = np.array(hp_data[axis_data[1]])
Z = np.array(hp_data[axis_data[2]])

# Plot the surface.
surf = ax.plot_trisurf(X, Y, Z)

ax.view_init(10, 90)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
# ax.zaxis.set_major_formatter('{x:.02f}')

ax.set_xlabel(axis_data[0], fontsize='12')
ax.set_ylabel(axis_data[1], fontsize='12')
ax.set_zlabel(axis_data[2], fontsize='12')


plt.show()
