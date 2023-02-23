import numpy as np
import pandas as pd
from scipy.stats import truncnorm


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
    solution['driving_cost'] = solution['distance'] * driving_cost_per_mile

    visit_prob_col_names = make_column_names(prefix="vp", ncols=n_sims)
    weighted_range_col_names = make_column_names(prefix="r", ncols=n_sims)

    miles_to_charge = 250 - (get_ranges(n_sims=n_sims) - solution['distance'].values.reshape(-1,1))
    visit_probs = compute_visiting_probability(miles_to_charge)
    visit_probs_table = pd.DataFrame(visit_probs, columns=visit_prob_col_names)
    weighted_range = np.multiply(miles_to_charge, visit_probs)
    weighted_range_table = pd.DataFrame(weighted_range, columns=weighted_range_col_names)

    solution = pd.concat([solution, visit_probs_table, weighted_range_table], axis=1)

    range_agg_expression = dict(zip(visit_prob_col_names, ['sum'] * n_sims))
    w_range_agg_expression = dict(zip(weighted_range_col_names, ['sum'] * n_sims))
    agg_expression = {'driving_cost' : 'sum'
                        , 'distance' : 'sum'
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

    return aggregated_cost_per_station


