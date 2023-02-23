import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import compute_cost_per_station
from sklearn.cluster import KMeans


number_of_simulations = 10
alpha = 0.05
driving_cost_per_mile = 0.041
charging_cost_per_mile = 0.0388
construction_cost_per_station = 5000
maintenance_fee_per_charger = 500


def main():

    cars = pd.read_csv("data/MOPTA2023_car_locations.csv", names=['ev_x', 'ev_y'])
    cars = cars.loc[cars.index.repeat(10)] # Repeat each location 10 times
    cars['loc_ix'] = cars.index
    cars = cars.reset_index(drop=True)

    kmeans = KMeans(n_clusters=600, random_state=0).fit(cars[['ev_x', 'ev_y']])
    station_location_ixs = kmeans.predict(cars[['ev_x', 'ev_y']])
    station_locations = kmeans.cluster_centers_
    stations = pd.DataFrame(station_locations, columns=['station_x', 'station_y'])
    stations['station_ix'] = np.arange(0, stations.shape[0])
    cars['station_ix'] = station_location_ixs

    cars_and_stations = cars.merge(stations, how='left', on='station_ix')

    costs = compute_cost_per_station(solution=cars_and_stations, n_sims=100)

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


