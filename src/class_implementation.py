from utils import compute_adjusted_charger_cost, repeat_rows, simulate_ranges, compute_visiting_probability, calculate_distance, load_ev_locations, generate_random_stations, solve_with_kmeans, compute_cost_per_station, assign_closest
import numpy as np
import copy
import pandas as pd
import time

solution_id = time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime(time.time()))


driving_cost_per_mile = 0.041
charging_cost_per_mile = 0.0388
construction_cost_per_station = 5000
maintenance_fee_per_charger = 500


class ElectricVehicle:
    def __init__(self, x:float, y:float):
        self.x = x
        self.y = y
        self.range = simulate_ranges(n_evs=1, n_sims=1)[0][0]
        self.visit_prob = compute_visiting_probability(self.range)


class Station:

    def __init__(self, x:float, y:float, vehicles:list, construction_cost:float=construction_cost_per_station):
        self.x = x
        self.y = y
        self.vehicles = vehicles
        self.construction_cost = construction_cost


    def calculate_distances(self):
        ev_xs = [v.x for v in self.vehicles]
        ev_ys = [v.y for v in self.vehicles]
        xys = zip(ev_xs, ev_ys)
        self.distances = [calculate_distance([x,y], [self.x, self.y]) for (x,y) in xys]
    

    def calculate_driving_cost(self):
        self.calculate_distances()
        self.driving_cost = np.sum(self.distances) * driving_cost_per_mile


    def calculate_charging_cost(self):
        used_charge = [(250 - v.range) - d for v, d in zip(self.vehicles, self.distances)]
        self.charging_cost = np.sum(used_charge) * charging_cost_per_mile
    

    def calculate_maintenance_cost(self):
        self.n_chargers = get_number_of_chargers(self.vehicles)
        self.maintenance_cost = compute_adjusted_charger_cost(self.n_chargers)


    def estimate_station_cost(self):
        self.calculate_driving_cost()
        self.calculate_charging_cost()
        self.calculate_maintenance_cost()
        self.station_cost = self.charging_cost + self.construction_cost + self.driving_cost + self.maintenance_cost


def generate_random_ev():
    x = np.random.uniform(0,290)
    y = np.random.uniform(0,170)
    ev = ElectricVehicle(x, y)
    return ev


def get_number_of_chargers(vehicles):
    visit_probabilities = np.sum([v.visit_prob for v in vehicles])
    n_chargers = np.ceil(visit_probabilities / 2)
    return n_chargers


def reassign_vehicle(vehicle_ix:int, previous_station_ix:int, new_station_ix:int, stations:list):
    previous_station = stations[previous_station_ix]
    new_station = stations[new_station_ix]
    new_station.vehicles = new_station.vehicles + [previous_station.vehicles[vehicle_ix]]
    previous_station.vehicles = previous_station.vehicles[0:vehicle_ix] + previous_station.vehicles[vehicle_ix+1:]
    new_station.estimate_station_cost()
    previous_station.estimate_station_cost()


ev_locations = load_ev_locations(path="data/MOPTA2023_car_locations.csv")
ev_locations['loc_ix'] = np.array(range(ev_locations.shape[0]))
# ev_locations = repeat_rows(ev_locations, times_to_repeat=10)
# ev_and_station_locations = solve_with_kmeans(ev_locations, n_stations=600)
# # print(ev_and_station_locations.head())

station_locations = pd.DataFrame(generate_random_stations(), columns=['station_ix', 'station_x', 'station_y'])

ev_and_station_locations = assign_closest(ev_locations, station_locations)

ev_and_station_locations = repeat_rows(ev_and_station_locations, times_to_repeat=10)

aggregated_cost_per_station, solution, total_cost = compute_cost_per_station(ev_and_station_locations, 100)

station_locations = ev_and_station_locations[['station_ix', 'station_x', 'station_y']].drop_duplicates()
station_dict = station_locations.set_index('station_ix').T.to_dict('list')


def vehicles_from_df(df):
    stations = []
    station_ixs = list(set(df['station_ix']))
    for s in station_ixs:
        sub_df = df[df['station_ix'] == s]
        xs = sub_df['ev_x']
        ys = sub_df['ev_y']
        vehicles = [ElectricVehicle(x,y) for (x,y) in zip(xs, ys)]
        x, y = station_dict[s]
        new_station = Station(x, y, vehicles)
        stations.append(new_station)
    return stations



stations = vehicles_from_df(ev_and_station_locations)



# n_evs_per_station = [10, 40, 50]
# vehicles1 = [generate_random_ev() for i in range(n_evs_per_station[0])]
# vehicles2 = [generate_random_ev() for i in range(n_evs_per_station[1])]
# vehicles3 = [generate_random_ev() for i in range(n_evs_per_station[2])]

# station1 = Station(5,5,vehicles1)
# station2 = Station(1,0,vehicles2)
# station3 = Station(5,1,vehicles3)

saved_cost = -1
iterations = 0
max_iter = 200
max_chargers = 8

def select_vehicles_to_reassign(vehicles, original_n_chargers):
    vehicles_to_reassign = []
    current_n_chargers = get_number_of_chargers(vehicles)
    while original_n_chargers == current_n_chargers:
        vehicles_to_reassign.append(vehicles[0])
        vehicles = vehicles[1:]
        current_n_chargers = get_number_of_chargers(vehicles)
    return vehicles_to_reassign
    

while (saved_cost < 0 or max_chargers > 8) and iterations < max_iter:

    iterations+= 1

    # stations = [station1, station2, station3]


    for ix, s in enumerate(stations):
        s.estimate_station_cost()
        # # print('station', ix, s.station_cost)

    max_chargers = np.max([s.n_chargers for s in stations])

    initial_total_cost = np.sum(s.station_cost for s in stations)

    max_station_ix = np.argmax([s.station_cost for s in stations])
    # # print('I', len(stations[max_station_ix].vehicles))
    vehicles_to_reassign = select_vehicles_to_reassign(stations[max_station_ix].vehicles, stations[max_station_ix].n_chargers)
    # # print('O', len(stations[max_station_ix].vehicles))
    n_to_reassign = len(vehicles_to_reassign)
    most_costly_station_vehicles = stations[max_station_ix].vehicles.copy()
    # # print('Most costly station is station {}'.format(max_station_ix+1))

    new_stations = copy.deepcopy(stations)

    for ix, ns in enumerate(new_stations):
        if ix != max_station_ix:
            ns.vehicles = ns.vehicles + vehicles_to_reassign
        else:
            ns.vehicles = ns.vehicles[n_to_reassign:]

    for s, ns in zip(stations, new_stations):
        ns.estimate_station_cost()

    cost_increase = [ns.station_cost - s.station_cost for (s,ns) in zip(stations, new_stations)]
    savings = cost_increase[max_station_ix]
    cost_increase[max_station_ix] = initial_total_cost
    lowest_increase_ix = np.argmin(cost_increase)

    old_station = stations[max_station_ix]
    new_station = stations[lowest_increase_ix]

    saved_cost = savings + cost_increase[lowest_increase_ix]
    if saved_cost < 0:
        # # print('A vehicle will be relocated from station {} to station {} with a saved expense of {}.'.format(max_station_ix+1, lowest_increase_ix+1, saved_cost))
        # reassign_vehicle(vehicle_ix=max_vehicle_ix, previous_station_ix=max_station_ix, new_station_ix=lowest_increase_ix, stations=stations)
        new_station.vehicles = new_station.vehicles + vehicles_to_reassign
        old_station.vehicles = old_station.vehicles[n_to_reassign:]
    else:
        print('No more changes will result in savings.')


    total_cost = np.sum(s.station_cost for s in stations)

    # # print('-----------------------\nPrevious cost: {}\nNew cost: {}.\nTotal savings: {}.\n------------------'.format(initial_total_cost, total_cost, initial_total_cost-total_cost))


# for ix, s in enumerate(stations):
#     s.estimate_station_cost()
#     # print('Station ', ix)
#     # print('n_vehicles: ', len(s.vehicles))
#     # print('n_chargers:', s.n_chargers)
#     # print('Cost: ', s.station_cost)
#     # print('.-.-.-.-.-.-.-.-.-.')

def stations_to_df(stations):
    rows = []
    for s_ix, s in enumerate(stations):
        for v in s.vehicles:
            vehicle_row = [v.x, v.y, s_ix, s.x, s.y]
            rows.append(vehicle_row)
    df = pd.DataFrame(rows, columns='ev_x, ev_y, station_ix, station_x, station_y'.split(', '))
    return df

df = stations_to_df(stations)

# print(df.head())

# print('Max chargers: ', max([s.n_chargers for s in stations]))

aggregated_cost_per_station, solution, total_cost = compute_cost_per_station(df, 100)

solution.to_csv(f'data/solutions/{solution_id}.csv', index=False, sep='\t')

with open('data/solutions/index.csv', 'a') as outfile:
    outfile.write(f"{solution_id}\t{str(total_cost)}\n")
outfile.close()