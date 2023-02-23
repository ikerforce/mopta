from utils import compute_cost_per_station, solve_with_kmeans, plot_solution, load_ev_locations, solve_with_random_assignment


number_of_simulations = 10
alpha = 0.05
driving_cost_per_mile = 0.041
charging_cost_per_mile = 0.0388
construction_cost_per_station = 5000
maintenance_fee_per_charger = 500


def main():

    ev_locations = load_ev_locations(path="data/MOPTA2023_car_locations.csv")

    ev_and_station_locations = solve_with_random_assignment(ev_locations)
    # ev_and_station_locations = solve_with_kmeans(ev_locations, n_stations=600)

    costs = compute_cost_per_station(solution=ev_and_station_locations, n_sims=100)

    plot_solution(ev_locations, costs)

if __name__ == '__main__':
    main()


