import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm
from sklearn.cluster import KMeans


driving_cost_per_mile = 0.041
charging_cost_per_mile = 0.0388
construction_cost_per_station = 5000
maintenance_fee_per_charger = 500


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

    cars['range'] = get_ranges()
    cars['visit_prob'] = get_visiting_probability(cars['range'].values)

    cars_and_stations = cars.merge(stations, how='left', on='station_ix')
    cars_and_stations['distance'] = calculate_distance([cars_and_stations['x'], cars_and_stations['y']], [cars_and_stations['x_station'], cars_and_stations['y_station']])
    cars_and_stations['driving_cost'] = cars_and_stations['distance'] * driving_cost_per_mile
    cars_and_stations['charging_cost'] = (250 - cars_and_stations['range']) * charging_cost_per_mile * cars_and_stations['visit_prob']

    station_stats = cars_and_stations.groupby('station_ix').agg({'driving_cost' : 'sum'
                                            , 'charging_cost' : 'sum'
                                            , 'visit_prob' : 'sum'
                                            , 'distance' : 'mean'
                                            , 'x_station' : 'mean'
                                            , 'y_station' : 'mean'})\
                                    .reset_index()
    station_stats['n_chargers'] = np.ceil(station_stats['visit_prob'] / 2)

    total_charging_cost = np.round(station_stats['charging_cost'].sum(), 2)
    total_driving_cost = np.round(station_stats['driving_cost'].sum(), 2)
    total_station_cost = np.round(station_stats.shape[0] * construction_cost_per_station, 2)
    total_maintenance_cost = np.round(station_stats['n_chargers'].sum() * maintenance_fee_per_charger, 2)
    total_cost = np.round(total_charging_cost + total_driving_cost + total_station_cost + total_maintenance_cost, 2)

    cost_str = "\nCharging cost: ${:,}\nDriving cost: ${:,}\nConstruction cost: ${:,}\nMaintenance cost: ${:,}\n-----------------------------------------------------\nTotal cost: ${:,}.\n"\
                .format(total_charging_cost, total_driving_cost, total_station_cost, total_maintenance_cost, total_cost)

    print(cost_str)

    fig, axs = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
    axs.scatter(x=cars['x'], y=cars['y'] , c='lightgrey', s=40, label='EV locations')
    axs.scatter(x=station_stats['x_station'], y=station_stats['y_station'] , c=station_stats['n_chargers'], s=5, label='Stations')

    plt.legend()

    plt.show()


def get_ranges(n_evs:int=10790):
    pd = truncnorm((20 - 100) / 50, (250 - 100) / 50, loc=100, scale=50)
    samples = pd.rvs(size=n_evs)
    return samples


def get_visiting_probability(ranges, lam=0.012):
    return np.exp(-lam**2 * (ranges - 20)**2 )


def calculate_distance(car_location:list, station_location:list):
    return abs(car_location[0] - station_location[0]) + abs(car_location[1] - station_location[1])


if __name__ == '__main__':
    main()

