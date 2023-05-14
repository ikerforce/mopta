#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:01:59 2023

@author: fergushathorn
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import truncnorm
import time
from sklearn.neighbors import BallTree

# Define individual structures for small instance

n_individual = 600 // 4 # stations in an individual
pop_size = 100 // 4 # number of individuals
x_max = 290 / 4 # size of search space on x axis
y_max = 150 / 4 # size of search space on y axis
max_chargers = 8 # max number of chargers at a station
# get sample of vehicle ranges
min_range = 20
max_range = 250
mean_range = 100
sd_range = 50

# cost parameters
driving_cost_per_mile = 0.041
charging_cost_per_mile = 0.0388
construction_cost_per_station = 5000
maintenance_fee_per_charger = 500
no_assignment_penalty = 5000
exceed_range_penalty = 5000


car_data = pd.read_csv('MOPTA2023_car_locations.csv')
car_data = np.array(car_data)
car_data = car_data[car_data[:,0] < x_max]
car_data = car_data[car_data[:,1] < y_max]

# define the population
options = {'c1': 0.09, 'c2': 0.1, 'w':0.96}

# uniform distribution for generating random points in the search space
ud_x = uniform(0, x_max)
ud_y = uniform(0, y_max)

# uniform distribution for generating random number between 0 and 1
ud = uniform(0,1)

class Ranges:
    def __init__(self, ranges):
        self.ranges = ranges

class Individual:

    def __init__(self, ident, positions, chargers, velocities):
        self.ident = ident
        self.positions = positions # numpy array of x,y positions of all stations
        self.chargers = chargers # numpy array of number of chargers per station
        self.velocities = velocities # numpy array of x,y velocity of each station
        self.best_positions = positions # numpy array of best configuration for the station
        self.cost = 0
        self.vehicle_assignments = None
        self.assignment_proportion = None
        self.station_counter = None
        self.distance_to_station = None
        self.total_cost_no_penalty = None

    def update_positions(self, global_best_stations, g:list, options=options): # global best stations must be a numpy array with x,y coords of the best global configuration
        # c1 = (0.08 - 0.15) * g[0] / g[1] + 0.15
        # c2 = (0.15 - 0.08) * g[0] / g[1] + 0.08
        inertia_components = options['w'] * (self.velocities)
        cognitive_components = ud.rvs(1) * options['c1'] * (self.best_positions - self.positions)
        social_components = ud.rvs(1) * options['c2'] * (global_best_stations - self.positions)
        
        new_velocities = inertia_components + cognitive_components + social_components
        new_velocities[new_velocities[:,0] > 1, 0] = 1
        new_velocities[new_velocities[:,1] > 1, 1] = 1
        new_velocities[new_velocities[:,0] < -1, 0] = -1
        new_velocities[new_velocities[:,1] < -1, 1] = -1
        new_positions = new_velocities + self.positions
        
        new_positions.reshape(n_individual, 2)
        
        self.positions = new_positions
        self.velocities = new_velocities
        
        
nd = norm(0.5,0.1) # normal distribution for sampling velocities

def gen_pop(pop_size, car_data_anchor):
    population = {} # empty dictionary to fill with individuals to make population
    for i in range(pop_size):
        # generating random x,y positions for the individual stations
      
        # uniform distribution for generating random points in the search space
        spawn_radius = 10
        ud_x = uniform(-spawn_radius, spawn_radius)
        # ud_y = uniform(-spawn_radius, spawn_radius)
      
        noise = ud_x.rvs((n_individual,2))
      
        positions = np.zeros_like(noise)
      
        for s in range(car_data_anchor.shape[0]):
            positions[s] = noise[s] + car_data_anchor[s]
        velocities=nd.rvs(size=(n_individual, 2)) * np.random.choice([1,-1], size=(n_individual, 2)) # sampling velocities from random dist
        population.update({i : Individual(ident=i,
                                          positions=positions.copy(),
                                          chargers=np.random.choice(list(range(1,9))),
                                          velocities=velocities)}
                          )
    return population

# sample ranges for each vehicle
def get_samples(min_range=min_range,
                max_range=max_range,
                mean_range=mean_range,
                sd_range=sd_range):
    pd = truncnorm((min_range - mean_range) / sd_range, 
                   (max_range - mean_range) / sd_range, 
                   loc=mean_range, scale=sd_range)
    samples = pd.rvs(size=(car_data.shape[0], 10))
    samples_aggregated = np.mean(samples, axis=1)

    return samples_aggregated

# ranges sampled
samples = get_samples()
rng = Ranges(ranges=np.array([mean_range]*car_data.shape[0]))

def compute_visiting_probability(ranges:np.array, lam:float=0.012):
    return np.exp(-lam**2 * (ranges - 20)**2 )

# Calculate the pairwise distances between points in the two arrays
def calculate_distance(car_location, station_location):
    return np.linalg.norm(car_location - station_location, axis=1)

def KNN(population, K, gen, local_best_scores):
    S = 16
    for individual in range(pop_size):
          
        if gen < 0:
            ranges = get_samples() # rng.ranges#
            rng.ranges = ranges
        else:
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
          charger_cost += maintenance_fee_per_charger * station_counter[s]
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
        non_travelling_vehicles = car_data.shape[0] * 10 - sum(v_b)
        non_travelling_vehicle_cost = non_travelling_vehicles * maintenance_fee_per_charger
      
        total_cost_no_penalty = driving_cost + charging_cost + station_cost + charger_cost
        total_cost = total_cost_no_penalty + penalty_cost_no_assignment_made + non_travelling_vehicle_cost # + distance_cost # penalty_cost_exceeding_range
      
        population[individual].cost = total_cost
        population[individual].assignment_proportion = pct_assigned
        population[individual].station_counter = station_counter
        population[individual].vehicle_assignments = vehicle_assignments
        population[individual].distance_to_station = distance_to_station
        
      
        if total_cost < local_best_scores[individual]:
          local_best_scores[individual] = total_cost
          population[individual].best_positions = population[individual].positions
        population[individual].total_cost_no_penalty = total_cost_no_penalty
      
    
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

#%% Hyper-param tuning

# parameters = [{'c1' : 0.1, 'c2' : 0.1, 'w' : 0.97},
#               {'c1' : 0.5, 'c2' : 0.1, 'w' : 0.97},
#               {'c1' : 0.15, 'c2' : 0.1, 'w' : 0.97}]

# generate parameter combinations
c1_values = np.round(np.arange(0.05, 0.15, 0.05), decimals=2)
c2_values = c1_values.copy()
w_values = np.around(np.arange(0.9, 1, 0.05), decimals=2)
reps = 10

parameters = [
    {'c1' : i, 'c2' : j, 'w' : k}
    for i in c1_values
    for j in c2_values
    for k in w_values
    ]

global_bests_tuning = []
global_best_score_tuning = []
population_fitness_tuning = []
population_fitness_no_penalty_tuning = []

for params in parameters:
    global_best_stations_reps = []
    global_best_score_reps = []
    pop_fitness_reps = []
    pop_fitness_no_penalty_reps = []
    
    for r in range(reps):
        bests = []
        local_bests = []
        mean_prop = []
        pop_fitness = []
        pop_fitness_no_penalty = []
        # print("Generating population...")
        pop_size = 1
        car_anchors = np.random.choice(list(range(car_data.shape[0])), n_individual)
        car_data_anchor = car_data[car_anchors]
        population = gen_pop(pop_size, car_data_anchor)
        generations = 100
        K = 30
        global_best_stations = car_data_anchor #np.array([[ud_x.rvs(1)[0],ud_y.rvs(1)[0]] for i in range(n_individual)]) # position of every station in the global best
        global_best_score = 1000000000
        global_best_station_counter, global_best_index = None,None
        global_best_assignments = None
        local_best_scores = [1e9 for i in range(pop_size)] # storing local best scores (not positions)
        start = time.time()
        particle_movement = []
        for gen in range(generations):
            global_best_score, global_best_stations, global_best_station_counter, global_best_index, global_best_assignments, local_best_scores, population_fitness, population_fitness_no_penalty = evaluate(population, car_data, global_best_score, global_best_stations, local_best_scores, K, global_best_station_counter, global_best_index, global_best_assignments, gen)
            bests.append(global_best_score)
            local_bests.append(np.mean(local_best_scores))
            pop_fitness.append(population_fitness)
            pop_fitness_no_penalty.append(population_fitness_no_penalty)
              
            for ind in population:
                    population[ind].update_positions(global_best_stations, [gen, generations], params)
            
            props = []
            gen_movement = []
            for i in range(pop_size):
                props.append(population[i].assignment_proportion)
                gen_movement.append(population[i].positions)
            
            mean_prop.append(np.mean(props))
            particle_movement.append(gen_movement)
        
        global_best_stations_reps.append(global_best_stations)
        global_best_score_reps.append(global_best_score)
        pop_fitness_reps.append(pop_fitness)
        pop_fitness_no_penalty_reps.append(pop_fitness_no_penalty)
        
    global_bests_tuning.append(global_best_stations_reps)
    global_best_score_tuning.append(np.mean(global_best_score_reps))
    population_fitness_tuning.append(np.mean(pop_fitness_reps))
    population_fitness_no_penalty_tuning.append(np.mean(pop_fitness_no_penalty_reps))

