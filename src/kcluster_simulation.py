import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import get_ranges, compute_visiting_probability, calculate_distance
from sklearn.cluster import KMeans


driving_cost_per_mile = 0.041
charging_cost_per_mile = 0.0388
construction_cost_per_station = 5000
maintenance_fee_per_charger = 500
number_of_simulations = 10
alpha = 0.05


def main():

    cars = pd.read_csv("data/MOPTA2023_car_locations.csv", names=['x', 'y'])
    cars = cars.loc[cars.index.repeat(10)] # Repeat each location 10 times
    cars['loc_ix'] = cars.index
    cars = cars.reset_index(drop=True)

    kmeans = KMeans(n_clusters=600, random_state=0).fit(cars[['x', 'y']])
    station_location_ixs = kmeans.predict(cars[['x', 'y']])
    station_locations = kmeans.cluster_centers_
    stations = pd.DataFrame(station_locations, columns=['x_station', 'y_station'])
    stations['station_ix'] = np.arange(0, stations.shape[0])
    cars['station_ix'] = station_location_ixs

    ranges = get_ranges(n_sims=number_of_simulations)
    visit_probs = compute_visiting_probability(ranges)
    
    visit_prob_col_names = ['vp{}'.format(i+1) for i in range(number_of_simulations)]
    visit_probs_table = pd.DataFrame(visit_probs, columns=visit_prob_col_names)

    cars['mean_range'] = np.multiply(ranges, visit_probs).mean(axis=1)
    cars = pd.concat([cars, visit_probs_table], axis=1)

    cars_and_stations = cars.merge(stations, how='left', on='station_ix')
    cars_and_stations['distance'] = calculate_distance([cars_and_stations['x'], cars_and_stations['y']]
                                                    , [cars_and_stations['x_station'], cars_and_stations['y_station']])
    cars_and_stations['driving_cost'] = cars_and_stations['distance'] * driving_cost_per_mile
    cars_and_stations['charging_cost'] = (250 - cars_and_stations['mean_range']) * charging_cost_per_mile # * cars_and_stations['visit_prob']

    print(cars_and_stations.head(10))

    range_agg_expression = dict(zip(visit_prob_col_names, ['sum'] * number_of_simulations))
    agg_expression = {'driving_cost' : 'sum'
                        , 'mean_range' : 'sum'
                        , 'charging_cost' : 'sum'
                        , 'distance' : 'mean'
                        , 'x_station' : 'mean'
                        , 'y_station' : 'mean'} | range_agg_expression

    station_stats = cars_and_stations.groupby('station_ix').agg(agg_expression)\
                                    .reset_index()

    percentile = np.percentile(q=95, a=station_stats[visit_prob_col_names].values, axis=1)
    station_stats['n_chargers'] = np.ceil(percentile / 2)

    total_charging_cost = np.round(station_stats['charging_cost'].sum(), 2)
    total_driving_cost = np.round(station_stats['driving_cost'].sum(), 2)
    total_station_cost = np.round(station_stats.shape[0] * construction_cost_per_station, 2)
    total_maintenance_cost = np.round(station_stats['n_chargers'].sum() * maintenance_fee_per_charger, 2)
    total_cost = np.round(total_charging_cost + total_driving_cost + total_station_cost + total_maintenance_cost, 2)

    cost_str = "\nCharging cost: ${:,}\nDriving cost: ${:,}\nConstruction cost: ${:,}\nMaintenance cost: ${:,}\n-----------------------------------------------------\nTotal cost: ${:,}.\n"\
                .format(total_charging_cost, total_driving_cost, total_station_cost, total_maintenance_cost, total_cost)

    print(station_stats.head())

    print(cost_str)

    fig, axs = plt.subplots(figsize=(10,10), ncols=2, nrows=2)
    axs[0,0].scatter(x=cars['x'], y=cars['y'] , c='lightgrey', s=40, label='EV locations')
    axs[0,0].scatter(x=station_stats['x_station'], y=station_stats['y_station'] , c=station_stats['n_chargers'], s=5, label='Stations')
    axs[0,0].set_title('Station and EV locations')

    axs[0,1].hist(station_stats['n_chargers'], bins=20)
    axs[0,1].set_title('Chargers per station')

    axs[1,1].hist(station_stats['distance'], bins=20)
    axs[1,1].set_title('Distance to station (miles)')

    axs[1,0].hist(station_stats['mean_range'], bins=20)
    axs[1,0].set_title('Mean range per station (miles)')

    plt.legend()

    plt.show()


if __name__ == '__main__':
    main()


