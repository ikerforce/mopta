#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 13 12:53:32 2023

@author: fergushathorn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import truncnorm
import time
from sklearn.neighbors import BallTree
import os

#%%

car_data = pd.read_csv('/Users/fergushathorn/Documents/MOPTA/mopta/mopta/data/MOPTA2023_car_locations.csv', header=None)
car_data = np.array(car_data)

solution = np.loadtxt('/Users/fergushathorn/Documents/MOPTA/mopta/mopta/data/best_solution.txt')

#%%

# Define individual structures

n_individual = solution.shape[0] # stations in an individual
pop_size = 1 # number of individuals
x_max = 290 # size of search space on x axis
y_max = 150 # size of search space on y axis
max_chargers = 8 # max number of chargers at a station

# car_data = car_data[car_data[:,0] < x_max]
# car_data = car_data[car_data[:,1] < y_max]

# cost parameters
driving_cost_per_mile = 0.041
charging_cost_per_mile = 0.0388
construction_cost_per_station = 5000
maintenance_fee_per_charger = 500
no_assignment_penalty = 5000
exceed_range_penalty = 5000

# define the population
options = {'c1': 0.05, 'c2': 0.14, 'w':0.83}

# uniform distribution for generating random points in the search space
ud_x = uniform(0, x_max)
ud_y = uniform(0, y_max)

# uniform distribution for generating random number between 0 and 1
ud = uniform(0,1)

class Ranges:
    def __init__(self, ranges):
        self.ranges = ranges

class Individual:

    def __init__(self, ident, positions):
        self.ident = ident
        self.positions = positions # numpy array of x,y positions of all stations
        self.cost = 0
        self.vehicle_assignments = None
        self.assignment_proportion = None
        self.distance_to_station = None
        self.total_cost_no_penalty = None


nd = norm(0.5,0.1) # normal distribution for sampling velocities

def gen_pop(pop_size, solution):
    population = {} # empty dictionary to fill with individuals to make population
    for i in range(pop_size):
        population.update({i : Individual(ident=i,
                                          positions=solution.copy())}
                          )
    return population


# get sample of vehicle ranges
min_range = 20
max_range = 250
mean_range = 100
sd_range = 50

# sample ranges for each vehicle
def get_samples(min_range=min_range,
                max_range=max_range,
                mean_range=mean_range,
                sd_range=sd_range):
    pd = truncnorm((min_range - mean_range) / sd_range, 
                   (max_range - mean_range) / sd_range, 
                   loc=mean_range, scale=sd_range)
    samples = pd.rvs(size=(car_data.shape[0], 10))
    samples_pct = np.percentile(samples, 20)
    samples_aggregated = np.tile(samples_pct, (car_data.shape[0],))

    return samples_aggregated

# ranges sampled
samples = get_samples()
rng = Ranges(ranges=samples)

def compute_visiting_probability(ranges:np.array, lam:float=0.012):
    return np.exp(-lam**2 * (ranges - 20)**2 )

# Calculate the pairwise distances between points in the two arrays
def calculate_distance(car_location, station_location):
    return np.linalg.norm(car_location - station_location, axis=1)

def KNN(population, K):
    S = 16
    for individual in range(pop_size):

        ranges = rng.ranges
      
        # get visiting probabilities
        c = compute_visiting_probability(ranges)
        # number of vehicles per batch
        v_b = np.round(c * 10, 0)
        # exclude vehicles with a low probability of charging
        exclusion = 0 # 0.2*10
      
        # vehicle batches which have a strong enough probability of visiting a charger
        applicable_vehicles = np.array([v for v in range(v_b.shape[0]) if v_b[v] > exclusion])
        applicable_ranges = ranges[applicable_vehicles]
      
        v_b = v_b[applicable_vehicles]
      
        vehicle_positions = car_data[applicable_vehicles]
      
        # station positions
        station_dict = {j : population[individual].positions[j] for j in range(n_individual)}
        # station position array
        station_positions = np.array([station_dict[i] for i in station_dict])
      
        # find the indices of applicable stations
        tree = BallTree(station_positions, leaf_size=2)
        # ind = tree.query_radius(vehicle_positions, r=applicable_ranges) # indices of stations that are in range of vehicles
        # ind_range = tree.query_radius(vehicle_positions, r=applicable_ranges) # indices of stations that are in range of vehicles
        dist, ind_closest = tree.query(vehicle_positions, k=K) # indices of closest stations
      
        # dictionary of how many chargers at each station
        station_counter = {i : 0 for i in range(station_positions.shape[0])}
        vehicle_assignments = {i : None for i in range(vehicle_positions.shape[0])}
        distance_to_station = []
        charging_costs = []
        driving_costs = []
      
        # shuffle_vehicle_indices = list(range(vehicle_positions.shape[0]))
        # random.shuffle(shuffle_vehicle_indices)
      
        # assign vehicles in order of demand, i.e. higher demand nodes get assigned first (means higher prob of charging)
        shuffle_vehicle_indices = np.argsort(v_b)[::-1]
      
        assigned = 0
      
        # for each vehicle, find a station with availability and add the vehicle batch to that station
        for v in shuffle_vehicle_indices:
          # first check if any of the nearby stations are already in use
          stations_in_use = [ind for ind in ind_closest[v] if (station_counter[ind] + v_b[v] <= S and station_counter[ind] > 0)]
          stations_not_in_use = [ind for ind in ind_closest[v] if ind not in stations_in_use]
          closest_stations = stations_in_use + stations_not_in_use
          for ind in range(len(closest_stations)):
            if station_counter[closest_stations[ind]] + v_b[v] <= S: 
              station_counter[closest_stations[ind]] += v_b[v]
              vehicle_assignments[v] = closest_stations[ind]
              distance_to_station.append(dist[v][ind])
              charging_costs.append(charging_cost_per_mile * v_b[v] * (250 - applicable_ranges[v] - dist[v][ind]))
              driving_costs.append(driving_cost_per_mile * v_b[v] * dist[v][ind])
              assigned += 1
              break
      
        # calculate the cost
        # driving cost
        driving_cost = sum(driving_costs)
        # charging cost
        charging_cost = sum(charging_costs)
        # station cost, charger cost
        station_cost, charger_cost = 0,0
        for s in station_counter:
          station_cost += construction_cost_per_station * (station_counter[s] > 0)
          charger_cost += maintenance_fee_per_charger * np.ceil(station_counter[s] / 2)
        # penalty cost (% assigned) (1-pct_assigned)*M
        pct_assigned = assigned/vehicle_positions.shape[0]
        penalty_cost_no_assignment_made = (vehicle_positions.shape[0] - assigned) * no_assignment_penalty # no_assignment_penalty * (vehicle_positions.shape[0] - assigned) # (1-pct_assigned) * 
        # penalty cost for exceeding range
        # ranges_resort = np.array([applicable_ranges[i] for i in shuffle_vehicle_indices])
        # distance_to_station_ar = np.array(distance_to_station)
        # penalty_cost_exceeding_range = exceed_range_penalty * sum(distance_to_station_ar > ranges_resort)
        
        # distance penalty for unassigned stations must be more than the distance cost
      
        # distance_cost = sum(np.array(distance_to_station)**2) * 10
        
        # penalty for vehicles that do not travel to a charger (to make sure the best global solution does not favour low demand scenarios)
        # non_travelling_vehicles = car_data.shape[0] * 10 - sum(v_b)
        # non_travelling_vehicle_cost = non_travelling_vehicles * maintenance_fee_per_charger
      
        total_cost_no_penalty = driving_cost + charging_cost + station_cost + charger_cost
        total_cost = total_cost_no_penalty + penalty_cost_no_assignment_made # + distance_cost # penalty_cost_exceeding_range
        
        population[individual].total_cost_no_penalty = total_cost_no_penalty
        
    return total_cost
      
    
def evaluate(population, vehicles, global_best_score, global_best_stations, local_best_scores, K, global_best_station_counter, global_best_index, global_best_assignments, gen):
    '''
    The vehicle-->station assignments are be made in this function and the cost is evaluated
    '''
    
    # print('Solving assignment problem...')
    KNN(population, K, gen, local_best_scores)
    # print('Assignment solved for population')
    
    population_fitness = []
    population_fitness_no_penalty = []
    for i in range(pop_size):
        population_fitness.append(population[i].cost)
        population_fitness_no_penalty.append(population[i].total_cost_no_penalty)
    
    if min(population_fitness) <= global_best_score:
        global_best_score = min(population_fitness)
        global_best_stations = population[np.argmin(population_fitness)].positions
        global_best_station_counter = population[np.argmin(population_fitness)].station_counter
        global_best_index = np.argmin(population_fitness)
        global_best_assignments = population[np.argmin(population_fitness)].vehicle_assignments
      # print('Updated global best')
    # print("Evaluation Complete\n----------------------------")
    
    return global_best_score, global_best_stations, global_best_station_counter, global_best_index, global_best_assignments, local_best_scores, population_fitness, population_fitness_no_penalty

#%%



np.random.seed(42)

noise_sd = 0.1

noise = np.random.normal(0, noise_sd, size=solution.shape)

population = gen_pop(1, solution)
total_cost = KNN(population, 200)

#%%

solution = np.loadtxt('/Users/fergushathorn/Documents/MOPTA/mopta/mopta/data/best_solution.txt')
deviations = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 5, 10, 20]
# costs = {i:[] for i in deviations}

number_of_samples = 100

costs = np.zeros(shape=(number_of_samples, len(deviations)))



for i in range(number_of_samples):
    for j in range(len(deviations)):
        #np.random.seed(42)
        noise = np.random.normal(0, deviations[j], size=solution.shape)
        population = gen_pop(1, solution+noise)
        total_cost = KNN(population, 200)
        costs[i,j] = total_cost

#%%
population = gen_pop(1, solution)
total_cost = KNN(population, 200)

plt.boxplot(costs)
plt.plot(range(1,len(deviations)+1), np.tile(total_cost, len(deviations)), color='red', label='PS-k-NN Solution')
plt.xticks(range(len(deviations)+1), [None] + deviations)
plt.xlabel('Standard deviation of noise')
plt.ylabel('Objective Value ($)')
plt.title('Sensitivity of the result to random perturbations')
plt.legend()
plt.savefig("/Users/fergushathorn/Documents/MOPTA/mopta/mopta/data/FinalResults/locationSensitivity.png", dpi=500)
plt.show()


