import numpy as np
import pandas as pd
import json


def extract_from_dataframe(df, parameter, column):

	return df[df['parameter'] == parameter][column].iloc[0]

with open('parameters.json') as json_file:
    PARAMETERS = json.load(json_file)

print(PARAMETERS)

parameters_list = []

for key in PARAMETERS.keys():

	parameters_list.append([key, PARAMETERS[key]['current_value'], PARAMETERS[key]['default_value']])


parameters_df = pd.DataFrame(parameters_list, columns=['parameter', 'current_value', 'default_value'])

print(parameters_df.head())

parameters_new_json = dict()

for row in range(parameters_df.shape[0]):

	parameter = parameters_df['parameter'].iloc[row]

	parameters_new_json[parameter] = {'current_value' : extract_from_dataframe(parameters_df, parameter, 'current_value'), 'default_value' : extract_from_dataframe(parameters_df, parameter, 'default_value')}

print(parameters_new_json)

with open('parameters_new.json', 'w') as fp:
    json.dump(PARAMETERS, fp)