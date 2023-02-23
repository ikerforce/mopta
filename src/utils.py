import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_ranges(n_evs:int=10790, n_sims:int=1):
    pd = truncnorm((20 - 100) / 50, (250 - 100) / 50, loc=100, scale=50)
    samples = pd.rvs(size=(n_evs, n_sims))
    return samples


def compute_visiting_probability(ranges:np.array, lam:float=0.012):
    return np.exp(-lam**2 * (ranges - 20)**2 )


def calculate_distance(car_location:list, station_location:list):
    return abs(car_location[0] - station_location[0]) + abs(car_location[1] - station_location[1])


def estimate_number_of_chargers(visit_probabilities:np.array, quant=95):
    percentile = np.percentile(q=quant, a=visit_probabilities, axis=1)
    return np.ceil(percentile / 2)


def make_column_names(prefix:str, ncols:int):
    return ['{}{}'.format(prefix, i+1) for i in range(ncols)]


def solve_with_kmeans(ev_locations:pd.DataFrame, n_stations:int=600):
    kmeans = KMeans(n_clusters=n_stations, random_state=0).fit(ev_locations[['ev_x', 'ev_y']])
    station_location_ixs = kmeans.predict(ev_locations[['ev_x', 'ev_y']])
    station_locations = kmeans.cluster_centers_
    stations = pd.DataFrame(station_locations, columns=['station_x', 'station_y'])
    stations['station_ix'] = np.arange(0, stations.shape[0])
    ev_locations['station_ix'] = station_location_ixs
    ev_and_station_locations = ev_locations.merge(stations, how='left', on='station_ix')
    return ev_and_station_locations


def solve_with_random_assignment(ev_locations:pd.DataFrame, n_stations:int=600):
    n_evs = ev_locations.shape[0]
    station_location_ixs = np.random.randint(0, 600, size=n_evs)
    station_locations = np.asarray([np.random.uniform(0,290,size=(n_evs,)), np.random.uniform(0,150,size=(n_evs,))]).T
    stations = pd.DataFrame(station_locations, columns=['station_x', 'station_y'])
    stations['station_ix'] = np.arange(0, stations.shape[0])
    ev_locations['station_ix'] = station_location_ixs
    ev_and_station_locations = ev_locations.merge(stations, how='left', on='station_ix')
    return ev_and_station_locations


def plot_solution(ev_locations:pd.DataFrame, costs:pd.DataFrame):
    fig, axs = plt.subplots(figsize=(10,10), ncols=2, nrows=2)
    axs[0,0].scatter(x=ev_locations['ev_x'], y=ev_locations['ev_y'] , c='lightgrey', s=40, label='EV locations')
    axs[0,0].scatter(x=costs['station_x'], y=costs['station_y'] , c=costs['n_chargers'], s=5, label='Stations')
    axs[0,0].set_title('Station and EV locations')

    axs[0,1].hist(costs['n_chargers'], bins=20)
    axs[0,1].set_title('Chargers per station')

    axs[1,1].hist(costs['distance'], bins=20)
    axs[1,1].set_title('Mean distance to station (miles)')

    axs[1,0].hist(costs['charging_cost'], bins=20)
    axs[1,0].set_title('Mean charging cost per station ($)')

    plt.legend()

    plt.show()


def initialize_pop(ev_locations:pd.DataFrame, n_stations:int=600, pop_size:int=10):
    pos = [solve_with_random_assignment(ev_locations, n_stations) for i in range(pop_size)]
    vel = np.zeros((pop_size, 1))  # velocity
    best = np.zeros((pop_size, 1))  # best score for individual
    pop = [[pos[i], vel[i], best[i]] for i in range(pop_size)]
    return pop


def solve_with_pso(ev_locations:pd.DataFrame, n_stations:int=600):

    generations = 5
    a = 0.5
    b = 2.5
    hyper_n = 50
    cog_and_social = {i: [np.random.uniform(a, b), np.random.uniform(a, b)] for i in range(hyper_n)}
    inert = {i: max(0, min(np.random.uniform(1.01, 2) * (0.5 * (cog_and_social[i][1] + cog_and_social[i][0]) - 1), 0.99))
             for i in range(hyper_n)}
    params = [[inert[i], cog_and_social[i][0], cog_and_social[i][1]] for i in range(hyper_n)]

    pop = initialize_pop(ev_locations, n_stations=600)

    exp=0

    w1 = params[exp][0]
    w2 = params[exp][1]
    w3 = params[exp][2]

    overall_best = [[0], 10000000]  # initialize global best

    particle_bests = {i: [pop[i], 10000000] for i in range(len(pop))}  # initialize dictionary of bests for each particle
    # run evolution for a given number of generations
    for generation in range(generations):
        print("Generation {}".format(generation))
        # evaluate fit of the current population

        costs = [compute_cost_per_station(x[0]) for x in pop]
        pop_fit = [X[1] for X in costs]

        print('Evaluation done. Best for this gen is {}'.format(np.max(pop_fit)))

        # find global best
        idx_best = np.argmin(pop_fit)  # get index for best individual in population
        # extract from lists
        if pop_fit[idx_best] < overall_best[1]:  # if idx_best is better than previous best, then update it, np.max(pop_fit)
            overall_best = [pop[idx_best][0], pop_fit[idx_best]]  # list of overall best individual and fitness score

        #  update the best solution for each particle
        for particle in range(len(pop)):
            # if the current fitness of a particle is better than historic bests, then update it
            if pop_fit[particle] < particle_bests[particle][1]:
                particle_bests[particle] = [pop[particle][0], pop_fit[particle]]
                pop[particle][2] = pop[particle][0]

        vel_update = [np.multiply(w1, pop[i][1]) +  # inertia
                        np.multiply(w2*np.random.uniform(0, 1), (np.subtract(pop[i][2], pop[i][0]))) +  # cognitive
                        np.multiply(w3*np.random.uniform(0, 1), (np.subtract(overall_best[0], pop[i][0])))  # social
                        for i in range(len(pop))]

        # update position, velocity with the latest versions
        for particle in range(len(pop)):
            pop[particle][0] = np.add(pop[particle][0], vel_update[particle])
            pop[particle][1] = vel_update[particle]


def load_ev_locations(path:str):
    ev_locations = pd.read_csv(path, names=['ev_x', 'ev_y'])
    ev_locations = ev_locations.loc[ev_locations.index.repeat(10)] # Repeat each location 10 times
    ev_locations['loc_ix'] = ev_locations.index
    ev_locations = ev_locations.reset_index(drop=True)
    return ev_locations


def compute_cost_per_station(solution:pd.DataFrame, n_sims:int=100):
    """This function calculates the average total costs given a solution and a number of simulations.
    
    Keywords:
        
        solution (pd.DataFrame) : A table where each row represents an ev and containts its location ['ev_x', 'ev_y'], the 
            station to which it was assigned ['station_ix'] and the location of said station ['station_x', 'station_y'].
            
    Output:
    
        aggregated_cost_per_station (pd.DataFrame) : A table where each row describes a station and its cost information.

        """

    driving_cost_per_mile = 0.041
    charging_cost_per_mile = 0.0388
    construction_cost_per_station = 5000
    maintenance_fee_per_charger = 500

    solution['distance'] = calculate_distance([solution['ev_x'], solution['ev_y']]
                                , [solution['station_x'], solution['station_y']])
    print(solution['distance'].min(), solution['distance'].max())
    solution['driving_cost'] = solution['distance'] * driving_cost_per_mile

    visit_prob_col_names = make_column_names(prefix="vp", ncols=n_sims)
    weighted_range_col_names = make_column_names(prefix="r", ncols=n_sims)

    ranges = get_ranges(n_sims=n_sims)
    miles_to_charge = 250 - (ranges - solution['distance'].values.reshape(-1,1))
    visit_probs = compute_visiting_probability(ranges)
    visit_probs[miles_to_charge > 100] == 1
    print(visit_probs.sum())
    visit_probs_table = pd.DataFrame(visit_probs, columns=visit_prob_col_names)
    weighted_range = np.multiply(miles_to_charge, visit_probs)
    weighted_range_table = pd.DataFrame(weighted_range, columns=weighted_range_col_names)

    solution = pd.concat([solution, visit_probs_table, weighted_range_table], axis=1)

    range_agg_expression = dict(zip(visit_prob_col_names, ['sum'] * n_sims))
    w_range_agg_expression = dict(zip(weighted_range_col_names, ['sum'] * n_sims))
    agg_expression = {'driving_cost' : 'sum'
                        , 'distance' : 'mean'
                        , 'station_x' : 'mean'
                        , 'station_y' : 'mean'} | range_agg_expression | w_range_agg_expression

    cost_per_station = solution.groupby('station_ix').agg(agg_expression)\
                                    .reset_index()

    aggregated_cost_per_station = cost_per_station[['station_ix', 'driving_cost', 'distance', 'station_x', 'station_y']]
    aggregated_cost_per_station['n_chargers'] = estimate_number_of_chargers(cost_per_station[visit_prob_col_names].values)
    aggregated_cost_per_station['charging_cost'] = cost_per_station[weighted_range_col_names].values.mean(axis=1) * charging_cost_per_mile

    total_charging_cost = np.round(aggregated_cost_per_station['charging_cost'].sum(), 2)
    total_driving_cost = np.round(aggregated_cost_per_station['driving_cost'].sum(), 2)
    total_station_cost = np.round(aggregated_cost_per_station.shape[0] * construction_cost_per_station, 2)
    total_maintenance_cost = np.round(aggregated_cost_per_station['n_chargers'].sum() * maintenance_fee_per_charger, 2)
    total_cost = np.round(total_charging_cost + total_driving_cost + total_station_cost + total_maintenance_cost, 2)

    cost_str = "\nCharging cost: ${:,}\nDriving cost: ${:,}\nConstruction cost: ${:,}\nMaintenance cost: ${:,}\n-----------------------------------------------------\nTotal cost: ${:,}.\n"\
                .format(total_charging_cost, total_driving_cost, total_station_cost, total_maintenance_cost, total_cost)

    print(cost_str)

    return aggregated_cost_per_station, total_cost


