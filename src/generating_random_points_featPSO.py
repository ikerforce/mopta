#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 23:51:12 2023

@author: fergushathorn
"""

import sys, os
import numpy as np
import pandas as pd
import concurrent.futures
import multiprocessing
import time
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean

#%% Read in demand nodes

car_data = pd.read_csv('/Users/fergushathorn/Documents/MOPTA/MOPTA_2023_data_demo/MOPTA2023_car_locations.csv',
                       names = ['lat','lon'])

car_data = np.array(car_data)

#%%

n_stations = 600
pop_size = 100


def generate_random_stations(x_range=290, y_range=150, n=n_stations):
    x = np.random.uniform(low=0, high=x_range, size=n)
    y = np.random.uniform(low=0, high=y_range, size=n)
    ind = np.dstack([x,y])[0]
    return ind
    
def generate_population(pop_size):
    
    # generate individuals
    population = [generate_individual() for i in range(pop_size)]
    
    return population

population = generate_population(pop_size)

plt.scatter(car_data[:,0], car_data[:,1], label='Vehicles', color='gray')
plt.scatter(population[0][:,0], population[0][:,1], label='Stations', color='red')

#%%

# initial velocities
v = np.random.uniform(low=5, high=20, size=(pop_size,n_stations,2))
probd = truncnorm((20 - 100) / 50, (250 - 100) / 50, loc=100, scale=50)

def evaluate(population, car_data):
    distance_list = []
    distance_max = 50
    
    counter = 0
    
    for individual in population:
        # assign vehicles to stations
        samples = probd.rvs(size=len(car_data))
        
        assignments = {i:0 for i in range(individual.shape[0])} # {tuple(i):0 for i in individual}
        
        vehicle_assignments = {i:None for i in range(car_data.shape[0])}
        
        for vehicle in range(len(car_data)):
            distance_threshold = min(distance_max,samples[vehicle])
            tree = cKDTree(individual)
            indices = tree.query_ball_point(car_data[vehicle], distance_threshold)
            
            assignment_per_index = [assignments[i] for i in indices]
            
            l=min(assignment_per_index)
            u=max(assignment_per_index)
            
            assignment_per_index_norm = [(i-l)/(u-l) if u-l > 0 else 0 
                                         for i in assignment_per_index ]
            
            distance_per_index = [euclidean(car_data[vehicle], individual[i])/distance_max for i in indices]
            
            scores = [0.5*(1-assignment_per_index_norm[i]) + 0.5*distance_per_index[i] for i in range(len(indices))]
            
            best_station = np.argmax(scores)
            
            assignments[best_station] += 1
            
            vehicle_assignments[vehicle] = best_station
        
        # simple version
        '''
        tree = cKDTree(individual)
        distances, indices = tree.query(car_data)
        distance_list.append(sum(distances))
        '''
        
    best_individual = np.argmin(distance_list)    
        
    total_cost = sum(distance_list)
    return [total_cost, best_individual, distance_list]

evaluation_results = evaluate(population, car_data)

#%%

# initialize global best
global_best = [evaluation_results[0], population[evaluation_results[1]]]

orig_best = global_best[1].copy()

# initialize individual bests
individual_bests = [[1e10, ind] for ind in population]

#%%

# hyperparameters
w1 = 0.99
w2 = 1.5
w3 = 1.5

#%%

def apply_perturbation(population, velocities, global_best, individual_bests):
    
    for individual in range(len(population)):
        inertia = velocities[individual] * w1
        cognitive = np.multiply(w2*np.random.uniform(0.75,1), np.subtract(individual_bests[individual][1], population[individual]))
        social = np.multiply(w3*np.random.uniform(0.75, 1), np.subtract(global_best[1], population[individual]))
        
        new_v = np.add(inertia, np.add(cognitive, social))
                
        population[individual] = population[individual] + new_v

    return population, velocities
        
#%%

def update_population(velocities, population, global_best, individual_bests, car_data):
    
    population, velocities = apply_perturbation(population, velocities, global_best, individual_bests)
    
    evaluation_results = evaluate(population, car_data)
    
    if evaluation_results[0] < global_best[0]: # update global best
        global_best = [evaluation_results[0], population[evaluation_results[1]]]
    
    for individual in range(len(population)): # update local bests
        if evaluation_results[2][individual] < individual_bests[individual][0]:
            individual_bests[individual] = [evaluation_results[2][individual], population[individual]]
    
    return velocities, population, global_best, individual_bests
            
#%%

results = []

generations = 1000

for gen in range(generations):
    
    v, population, global_best, individual_bests = update_population(v, population, global_best, individual_bests, car_data)
    
    results.append(global_best[0])
    
#%%

plt.scatter(car_data[:,0], car_data[:,1], label='Vehicles', color='gray')
plt.scatter(population[0][:,0], population[0][:,1], label='Stations', color='red')

#%%

plt.scatter(car_data[:,0], car_data[:,1], label='Vehicles', color='gray')
plt.scatter(orig_best[:,0], orig_best[:,1], label='Original')
plt.scatter(global_best[1][:,0], global_best[1][:,1], label='Final')
plt.legend()

#%%

plt.plot(results)


