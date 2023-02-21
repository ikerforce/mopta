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

