import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


driving_cost_per_mile = 0.041
charging_cost_per_mile = 0.0388
construction_cost_per_station = 5000
maintenance_fee_per_charger = 500


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


def repeat_rows(df:pd.DataFrame, times_to_repeat:int):
    df = df.loc[df.index.repeat(times_to_repeat)] # Repeat each location 10 times
    df['loc_ix'] = df.index
    df = df.reset_index(drop=True)
    return df


def load_ev_locations(path:str):
    ev_locations = pd.read_csv(path, names=['ev_x', 'ev_y'])
    ev_locations = repeat_rows(ev_locations, times_to_repeat=10)
    return ev_locations


def compute_proportional_chargers(cost_per_vehicle:pd.DataFrame):
    cost_per_vehicle['vp_proportion'] = cost_per_vehicle['vp1'] / cost_per_vehicle.groupby('station_ix')['vp1'].transform('sum')
    return cost_per_vehicle['vp_proportion'].values


def compute_adjusted_charger_cost(num_chargers, max_num_chargers=8, charger_cost=500):
    over_charger_capacity_penalty=charger_cost * max_num_chargers + 1 # This way one charger over capacity will cost more than having a full station within capacity
    return np.maximum(0, num_chargers - max_num_chargers) * over_charger_capacity_penalty + np.minimum(max_num_chargers, num_chargers) * charger_cost


def find_most_costly_location(solution:pd.DataFrame, n_sims=10):
    costs_per_station, ev_and_station_assignment, total_cost = compute_cost_per_station(solution, n_sims)
    costs_per_station['adjusted_charger_cost'] = compute_adjusted_charger_cost(costs_per_station['n_chargers'])
    cost_per_ev_location = ev_and_station_assignment[['loc_ix', 'driving_cost', 'station_ix', 'vp1', 'ev_x', 'ev_y']].groupby('loc_ix').agg({'ev_x' : 'first', 'ev_y' : 'first', 'station_ix' : 'first', 'driving_cost' : 'sum', 'vp1' : 'sum'}).reset_index()
    cost_per_vehicle = cost_per_ev_location.merge(costs_per_station[['station_ix', 'station_x', 'station_y', 'n_chargers', 'adjusted_charger_cost', 'station_cost']], on='station_ix', how='left')
    print(cost_per_vehicle.sort_values('station_ix').head(10))
    cost_per_vehicle['proportion_of_visits'] = compute_proportional_chargers(cost_per_vehicle)
    cost_per_vehicle['proportional_chargers'] = cost_per_vehicle['proportion_of_visits'] * cost_per_vehicle['n_chargers']
    cost_per_vehicle['proportional_charger_cost'] = compute_adjusted_charger_cost(cost_per_vehicle['proportional_chargers'])
    cost_per_vehicle['adjusted_cost'] = cost_per_vehicle['proportional_charger_cost'] + cost_per_vehicle['driving_cost']
    # print(cost_per_vehicle[['station_ix', 'station_x', 'station_y', 'n_chargers', 'adjusted_charger_cost', 'proportional_chargers', 'proportional_charger_cost']].sort_values('n_chargers', ascending=False).head(10))
    most_costly_location = cost_per_vehicle.iloc[cost_per_vehicle['adjusted_cost'].idxmax()][['loc_ix', 'ev_x', 'ev_y', 'station_ix', 'proportional_chargers', 'adjusted_cost']]
    costs_per_station = costs_per_station[['station_ix', 'n_chargers', 'station_x', 'station_y']]
    compute_reassignment_cost(costs_per_station, most_costly_location)


def compute_reassignment_cost(costs_per_station, most_costly_location):
    most_costly_location = pd.DataFrame([most_costly_location.values], columns=['mcl_loc_ix', 'mcl_ev_x', 'mcl_ev_y', 'mcl_station_ix', 'mcl_proportional_chargers', 'mcl_adjusted_cost'])
    most_costly_location = repeat_rows(most_costly_location, times_to_repeat=costs_per_station.shape[0])
    costs_per_station = pd.concat([costs_per_station, most_costly_location], axis=1)
    costs_per_station['new_cost'] = calculate_distance([costs_per_station['mcl_ev_x'], costs_per_station['mcl_ev_y']]
                        , [costs_per_station['station_x'], costs_per_station['station_y']]) * driving_cost_per_mile
    costs_per_station['reassigned_chargers'] = costs_per_station['n_chargers'] + costs_per_station['mcl_proportional_chargers']
    costs_per_station.loc[costs_per_station['mcl_station_ix'] == costs_per_station['station_ix'], 'reassigned_chargers'] = costs_per_station['n_chargers']
    costs_per_station['new_charger_proportion'] = costs_per_station['mcl_proportional_chargers'] / costs_per_station['reassigned_chargers']
    costs_per_station['new_cost'] = costs_per_station['new_cost'] + compute_adjusted_charger_cost(costs_per_station['reassigned_chargers']) * costs_per_station['new_charger_proportion']
    costs_per_station['savings'] = costs_per_station['mcl_adjusted_cost'] - costs_per_station['new_cost']

    print(costs_per_station[['loc_ix', 'station_ix', 'n_chargers', 'reassigned_chargers', 'mcl_adjusted_cost', 'new_cost', 'savings', 'mcl_station_ix']].sort_values('savings', ascending=False))


def simulate_evs(solution:pd.DataFrame, n_sims:int=100):

    solution['distance'] = calculate_distance([solution['ev_x'], solution['ev_y']]
                            , [solution['station_x'], solution['station_y']])
    visit_prob_col_names = make_column_names(prefix="vp", ncols=n_sims)
    weighted_range_col_names = make_column_names(prefix="r", ncols=n_sims)

    ranges = get_ranges(n_sims=n_sims)
    miles_to_charge = 250 - (ranges - solution['distance'].values.reshape(-1,1))
    visit_probs = compute_visiting_probability(ranges)
    visit_probs[miles_to_charge > 100] == 1
    visit_probs_table = pd.DataFrame(visit_probs, columns=visit_prob_col_names)
    weighted_range = np.multiply(miles_to_charge, visit_probs)
    weighted_range_table = pd.DataFrame(weighted_range, columns=weighted_range_col_names)
    return pd.concat([solution, visit_probs_table, weighted_range_table], axis=1), visit_prob_col_names, weighted_range_col_names


def compute_cost_per_station(solution:pd.DataFrame, n_sims:int=100):
    """This function calculates the average total costs given a solution and a number of simulations.
    
    Keywords:
        
        solution (pd.DataFrame) : A table where each row represents an ev and containts its location ['ev_x', 'ev_y'], the 
            station to which it was assigned ['station_ix'] and the location of said station ['station_x', 'station_y'].
            
    Output:
    
        aggregated_cost_per_station (pd.DataFrame) : A table where each row describes a station and its cost information.

        """

    solution, visit_prob_col_names, weighted_range_col_names = simulate_evs(solution, n_sims)
    solution['driving_cost'] = solution['distance'] * driving_cost_per_mile

    range_agg_expression = dict(zip(visit_prob_col_names, ['sum'] * n_sims))
    w_range_agg_expression = dict(zip(weighted_range_col_names, ['sum'] * n_sims))
    agg_expression = {'driving_cost' : 'sum'
                        , 'distance' : 'mean'
                        , 'station_x' : 'first'
                        , 'station_y' : 'first'} | range_agg_expression | w_range_agg_expression

    cost_per_station = solution.groupby('station_ix').agg(agg_expression)\
                                    .reset_index()

    aggregated_cost_per_station = cost_per_station[['station_ix', 'driving_cost', 'distance', 'station_x', 'station_y']]
    aggregated_cost_per_station['n_chargers'] = estimate_number_of_chargers(cost_per_station[visit_prob_col_names].values)
    aggregated_cost_per_station['charging_cost'] = cost_per_station[weighted_range_col_names].values.mean(axis=1) * charging_cost_per_mile
    aggregated_cost_per_station['station_cost'] = aggregated_cost_per_station['driving_cost'] \
                                                    + aggregated_cost_per_station['charging_cost'] \
                                                    + aggregated_cost_per_station['n_chargers'] * maintenance_fee_per_charger \
                                                    + construction_cost_per_station

    total_charging_cost = np.round(aggregated_cost_per_station['charging_cost'].sum(), 2)
    total_driving_cost = np.round(aggregated_cost_per_station['driving_cost'].sum(), 2)
    total_station_cost = np.round(aggregated_cost_per_station.shape[0] * construction_cost_per_station, 2)
    total_maintenance_cost = np.round(aggregated_cost_per_station['n_chargers'].sum() * maintenance_fee_per_charger, 2)
    total_cost = np.round(aggregated_cost_per_station['station_cost'].sum(), 2)

    cost_str = "\nCharging cost: ${:,}\nDriving cost: ${:,}\nConstruction cost: ${:,}\nMaintenance cost: ${:,}\n-----------------------------------------------------\nTotal cost: ${:,}.\n"\
                .format(total_charging_cost, total_driving_cost, total_station_cost, total_maintenance_cost, total_cost)

    print(cost_str)

    return aggregated_cost_per_station, solution, total_cost


