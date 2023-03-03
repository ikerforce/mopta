from utils import compute_adjusted_charger_cost, simulate_ranges, compute_visiting_probability, estimate_number_of_chargers, calculate_distance
import numpy as np
import copy


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

    def get_number_of_chargers(self):
        visit_probabilities = np.sum([v.visit_prob for v in self.vehicles])
        self.n_chargers = np.ceil(visit_probabilities / 2)

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
        self.get_number_of_chargers()
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

def reassign_vehicle(vehicle_ix:int, previous_station_ix:int, new_station_ix:int, stations:list):
    previous_station = stations[previous_station_ix]
    new_station = stations[new_station_ix]
    new_station.vehicles = new_station.vehicles + [previous_station.vehicles[vehicle_ix]]
    previous_station.vehicles = previous_station.vehicles[0:vehicle_ix] + previous_station.vehicles[vehicle_ix+1:]
    new_station.estimate_station_cost()
    previous_station.estimate_station_cost()


n_evs_per_station = [3, 17, 8]
vehicles1 = [generate_random_ev() for i in range(n_evs_per_station[0])]
vehicles2 = [generate_random_ev() for i in range(n_evs_per_station[1])]
vehicles3 = [generate_random_ev() for i in range(n_evs_per_station[2])]

station1 = Station(5,5,vehicles1)
station2 = Station(1,0,vehicles2)
station3 = Station(5,1,vehicles3)
station1.estimate_station_cost()
station2.estimate_station_cost()
station3.estimate_station_cost()

stations = [station1, station2, station3]

initial_total_cost = np.sum(s.station_cost for s in stations)

max_station_ix = np.argmax([s.station_cost for s in stations])
max_vehicle_ix = np.argmax([ev.visit_prob for ev in stations[max_station_ix].vehicles])
most_costly_station_vehicles = stations[max_station_ix].vehicles.copy()
print('Most costly staion is station {}'.format(max_station_ix+1))

new_stations = copy.deepcopy(stations)

for ix, ns in enumerate(new_stations):
    if ix != max_station_ix:
        ns.vehicles = ns.vehicles + [most_costly_station_vehicles[max_vehicle_ix]]
    else:
        ns.vehicles = ns.vehicles[0:max_vehicle_ix] + ns.vehicles[max_vehicle_ix+1:]

for s, ns in zip(stations, new_stations):
    ns.estimate_station_cost()
    print('nchargers: ', s.n_chargers)
    print('nchargers: ', s.maintenance_cost)
    print('Old: ', s.station_cost, 'New: ', ns.station_cost)

cost_increase = [ns.station_cost - s.station_cost for (s,ns) in zip(stations, new_stations)]
savings = cost_increase[max_station_ix]
print(cost_increase)
cost_increase[max_station_ix] = 50000
lowest_increase_ix = np.argmin(cost_increase)

saved_cost = savings + cost_increase[lowest_increase_ix]
if saved_cost < 0:
    print('A vehicle will be relocated from station {} to station {} with a saved expense of {}.'.format(max_station_ix+1, lowest_increase_ix+1, saved_cost))
    reassign_vehicle(vehicle_ix=max_vehicle_ix, previous_station_ix=max_station_ix, new_station_ix=lowest_increase_ix, stations=stations)

old_station = stations[max_station_ix]
new_station = stations[lowest_increase_ix]
new_station.vehicles = new_station.vehicles + [old_station.vehicles[max_vehicle_ix]]
old_station.vehicles = old_station.vehicles[0:max_vehicle_ix] + old_station.vehicles[max_vehicle_ix+1:]


total_cost = np.sum(s.station_cost for s in stations)

print(initial_total_cost, total_cost, initial_total_cost-total_cost)