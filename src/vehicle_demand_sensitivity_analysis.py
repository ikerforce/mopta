#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 29 10:25:03 2023

@author: fergushathorn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

fig_save_path = "VehicleNumberSensitivity/Figures/"

#%% best result cost

pctiles = [0.01, 0.1, 1, 5, 10, 20, 30, 40, 50]

sa = np.zeros(shape=(len(pctiles), 10))

for pt in range(len(pctiles)):
    
    res = np.loadtxt("VehicleNumberSensitivity/percentile_"+str(pctiles[pt])+"/best_scores.txt")
    
    sa[pt, :] = res

sa_sd = np.std(sa, axis=1, ddof=1)
sa_mean = np.mean(sa, axis=1)

cl = 0.95

sa_se = sa_sd / np.sqrt(len(pctiles))

sa_cv = np.abs(t.ppf((1 - cl) / 2, df=len(pctiles)-1))

sa_marg = sa_se * sa_cv

yerr=[[sa_mean - sa_marg], [sa_mean + sa_marg]]

plt.plot(sa_mean, 'o', color='blue', label='Mean')
plt.errorbar(range(len(pctiles)), 
             sa_mean, 
             yerr=sa_marg, 
             fmt='none', 
             ecolor='red', 
             capsize=5, 
             label='Confidence Interval')
plt.xticks(range(len(pctiles)), pctiles)
plt.title("Cost vs Service Level")
plt.xlabel('1 - Service Level (%)')
plt.ylabel('Best Score ($)')
plt.grid(True)
plt.legend()
plt.savefig(fig_save_path+"cost.png", dpi=500)
plt.show()

#%% number of chargers
sa_ch = np.zeros(shape=(len(pctiles), 9))

for pt in range(len(pctiles)):
    
    res = np.loadtxt("VehicleNumberSensitivity/percentile_"+str(pctiles[pt])+"/number_of_chargers.txt")
    
    sa_ch[pt, :] = res

sa_ch_sd = np.std(sa_ch, axis=1, ddof=1)
sa_ch_mean = np.mean(sa_ch, axis=1)

cl = 0.95

sa_ch_se = sa_ch_sd / np.sqrt(len(pctiles))

sa_ch_cv = np.abs(t.ppf((1 - cl) / 2, df=len(pctiles)-1))

sa_ch_marg = sa_ch_se * sa_ch_cv

yerr=[[sa_ch_mean - sa_ch_marg], [sa_ch_mean + sa_ch_marg]]

plt.plot(sa_ch_mean, 'o', color='blue', label='Mean')
plt.errorbar(range(len(pctiles)), 
             sa_ch_mean, 
             yerr=sa_ch_marg, 
             fmt='none', 
             ecolor='red', 
             capsize=5, 
             label='Confidence Interval')
plt.xticks(range(len(pctiles)), pctiles)
plt.title("Number of chargers vs Service Level (%)")
plt.xlabel('1 - Service Level')
plt.ylabel('Number of chargers')
plt.grid(True)
plt.legend()
plt.savefig(fig_save_path+"chargers.png", dpi=500)
plt.show()

#%% number of stations
sa_st = np.zeros(shape=(len(pctiles), 10))

for pt in range(len(pctiles)):
    
    res = np.loadtxt("VehicleNumberSensitivity/percentile_"+str(pctiles[pt])+"/number_of_stations.txt")
    
    sa_st[pt, :] = res

sa_st_sd = np.std(sa_st, axis=1, ddof=1)
sa_st_mean = np.mean(sa_st, axis=1)

cl = 0.95

sa_st_se = sa_st_sd / np.sqrt(len(pctiles))

sa_st_cv = np.abs(t.ppf((1 - cl) / 2, df=len(pctiles)-1))

sa_st_marg = sa_st_se * sa_st_cv

yerr=[[sa_st_mean - sa_st_marg], [sa_st_mean + sa_st_marg]]

plt.plot(sa_st_mean, 'o', color='blue', label='Mean')
plt.errorbar(range(len(pctiles)), 
             sa_st_mean, 
             yerr=sa_st_marg, 
             fmt='none', 
             ecolor='red', 
             capsize=5, 
             label='Confidence Interval')
plt.xticks(range(len(pctiles)), pctiles)
plt.title("Number of stations vs Service Level")
plt.xlabel('1 - Service Level (%)')
plt.ylabel('Number of stations')
plt.grid(True)
plt.legend()
plt.savefig(fig_save_path+"stations.png", dpi=500)
plt.show()

#%% chargers per station
c_p_s = np.mean(np.ceil(sa_ch / sa_st[:, :-1]), axis=1)

plt.bar(range(len(pctiles)), c_p_s, color='blue')
plt.xticks(range(len(pctiles)), pctiles)
plt.title("Station Chargers vs Service Level")
plt.xlabel('1 - Service Level (%)')
plt.ylabel('Chargers per Station')
plt.savefig(fig_save_path+"chargers_per_station.png", dpi=500)
plt.show()
