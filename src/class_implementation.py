from utils import compute_adjusted_charger_cost, simulate_ranges, compute_visiting_probability, estimate_number_of_chargers, calculate_distance
import numpy as np


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
        visit_probabilities = [[v.visit_prob] for v in self.vehicles]
        self.n_chargers = estimate_number_of_chargers(visit_probabilities)

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
        self.station_cost = self.charging_cost + self.construction_cost + self.driving_cost + self.charging_cost



ev1 = ElectricVehicle(0,10)
ev2 = ElectricVehicle(9,7)
ev3 = ElectricVehicle(4,1)
vehicles1 = [ev1, ev2, ev3]

ev1 = ElectricVehicle(3,12)
ev2 = ElectricVehicle(3,3)
ev3 = ElectricVehicle(8,7)
vehicles2 = [ev1, ev2, ev3]

ev1 = ElectricVehicle(11,11)
ev2 = ElectricVehicle(8,8)
ev3 = ElectricVehicle(7,2)
vehicles3 = [ev1, ev2, ev3]

station1 = Station(5,5,vehicles1)
station2 = Station(1,0,vehicles2)
station3 = Station(5,1,vehicles3)
station1.estimate_station_cost()
station2.estimate_station_cost()
station3.estimate_station_cost()
print(station1.station_cost)
print(station2.station_cost)
print(station3.station_cost)

stations = [station1, station2, station3]
max_station_ix = np.argmax([s.station_cost for s in stations])
max_vehicle_ix = np.argmax([ev.visit_prob for ev in stations[max_station_ix].vehicles])
most_costly_station_vehicles = stations[max_station_ix].vehicles.copy()
print('Most costly staion is station {}'.format(max_station_ix+1))

for ix, s in enumerate(stations):
    if ix != max_station_ix:
        s.vehicles = s.vehicles + [most_costly_station_vehicles[max_vehicle_ix]]
    else:
        s.vehicles = most_costly_station_vehicles[0:max_vehicle_ix] + most_costly_station_vehicles[max_vehicle_ix+1:]

for s in stations:
    s.estimate_station_cost()
    print(s.station_cost)
