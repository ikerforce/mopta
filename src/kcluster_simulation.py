from utils import compute_cost_per_station, solve_with_kmeans, plot_solution, load_ev_locations, solve_with_random_assignment, find_most_costly_location, get_execution_arguments



number_of_simulations = 10

def main():

    args = get_execution_arguments()

    ev_locations = load_ev_locations(path="data/MOPTA2023_car_locations.csv")

    if args.method == 'random':
        ev_and_station_locations = solve_with_random_assignment(ev_locations)
    elif args.method == 'kmeans':
        ev_and_station_locations = solve_with_kmeans(ev_locations, n_stations=600)

    print(ev_and_station_locations.head())

    # costs_per_station, ev_and_station_assignment, total_cost = compute_cost_per_station(solution=ev_and_station_locations, n_sims=args.nsims)

    # find_most_costly_location(ev_and_station_locations, n_sims=args.nsims)

    # plot_solution(ev_locations, costs_per_station)





if __name__ == '__main__':
    main()


