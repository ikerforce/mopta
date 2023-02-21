# MOPTA Competition

__current lowerbownd__: `inf` 

## Assumptions
- Distance is measured by Manhattan distance

### Estimation of number of chargers

1. For each ev simulate the visiting probability `n_sims` times.
2. For each station sum the visiting probability of each of the cars assigned to it (keep every simulation independent of each other).
3. After suming the visiting probability of the cars, get the 95-th percentile of all the `n_sims` and thus determine the number of chargers.

### Etimating costs

`charging_cost` = (250 - `ev_range` + `distance_from_ev_location_to_station`) * `charging_cost_per_mile`
`driving_cost` = `distance_from_ev_location_to_station` + `driving_cost_per_mile`
`chargers_cost` = `n_chargers` * `maintenance fee`
`stations_cost` = `n_stations` * `construction cost`

