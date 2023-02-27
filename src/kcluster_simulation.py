from utils import compute_cost_per_station, solve_with_kmeans, plot_solution, load_ev_locations, solve_with_random_assignment, solve_with_pso
import argparse



number_of_simulations = 10

def main():

    args = get_execution_arguments()

    ev_locations = load_ev_locations(path="data/MOPTA2023_car_locations.csv")

    if args.method == 'random':
        ev_and_station_locations = solve_with_random_assignment(ev_locations)
    elif args.method == 'kmeans':
        ev_and_station_locations = solve_with_kmeans(ev_locations, n_stations=600)
    # ev_and_station_locations = solve_with_pso(ev_locations)

    costs_per_station, total_cost = compute_cost_per_station(solution=ev_and_station_locations, n_sims=args.nsims)

    plot_solution(ev_locations, costs_per_station)


def get_execution_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsims", help="Number of simulations to be ran. If not specified, the value is 10.")
    parser.add_argument("--method", help="Solution method to be used. Can be kmeans or random.")
    args = parser.parse_args()
    if args.nsims is None:
        args.nsims = 10
    if args.method not in ['kmeans', 'random']:
        raise Exception("{} is not a valid solving method. Please use kmeans or random.".format(args.method))
    return args


if __name__ == '__main__':
    main()


