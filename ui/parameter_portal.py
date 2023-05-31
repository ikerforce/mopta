import sys
sys.path.append("../")
from mental_health_sim import simulation
from classes_mental_health_v5_documented import simulation as new_simulation
from classes_mental_health_v5_documented import TOA
from dashboard_functions import *

# Import required libraries
import os
import pickle
import copy
import datetime as dt
import math
import base64

import requests
import pandas as pd
import numpy as np
# from flask import Flask
import json
from dash import dcc, html, no_update
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

Blues = ['rgb(0,0,128)', '  rgb(0,0,255)', 'rgb(65,105,225)', 'rgb(70,130,180)', 'rgb(100,149,237)', 'rgb(30,144,255)', 'rgb(135,206,250)', 'rgb(135,206,235)', 'rgb(0,191,255)']

import random

random.seed(221032)

# Multi-dropdown options
from controls import PATIENT_STATUSES, DEPARTMENT_NAMES, DEPARTMENT_CAPACITY, DEPARTMENT_COLORS, TOOLTIP_INFO
 
with open('parameters.json') as json_file:
    PARAMETERS = json.load(json_file)

parameters_list = []

for key in PARAMETERS.keys():

    parameters_list.append([key, PARAMETERS[key]['current_value'], PARAMETERS[key]['default_value']])

def extract_from_list(list, parameter, column):

    with open('parameters.json') as json_file:
        PARAMETERS = json.load(json_file)

    list = []

    for key in PARAMETERS.keys():

        list.append([key, PARAMETERS[key]['current_value'], PARAMETERS[key]['default_value']])

    if column == 'current_value':

        return [X for X in list if X[0] == parameter][0][1]

    elif column == 'default_value':

        return [X for X in list if X[0] == parameter][0][2]

    else:

        return 'Error, not real column.' 


app = dash.Dash(__name__, routes_pathname_prefix='/parameters/')
server = app.server

# Create controls
patient_status_options = [{'label': str(PATIENT_STATUSES[patient_status]),
                        'value': str(patient_status)}
                       for patient_status in PATIENT_STATUSES]

department_name_options = [{'label': str(DEPARTMENT_NAMES[department]),
                        'value': str(department)}
                       for department in DEPARTMENT_NAMES]


# Tooltip settings
tooltip_style = {"font-size": "80%", 'width': '300px', 'border':'solid black 1px', 'background-color' : 'white'}

# Parameter settings
parameter_style = {"margin-top": "2px", "margin-left" : "40px"}


# Logo formatting
logo_filename = os.getcwd() + '/images/VU_logo.png' # replace with your own image
encoded_logo= base64.b64encode(open(logo_filename, 'rb').read())


layout = dict(
    autosize=True,
    automargin=True,
    margin=dict(
        l=30,
        r=30,
        b=20,
        t=40
    ),
    hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=10), orientation='h'),
    title='Title of the graph',
    mapbox=dict(
        style="light",
        center=dict(
            lon=-78.05,
            lat=42.54
        ),
        zoom=7,
    )
)

# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id='parameters_list'),
        html.Div(
            [
                html.Div(
                    [
                        html.H2(
                            'Mental Health Center',

                        ),
                        html.H4(
                            'Parameter portal',
                        )
                    ],

                    className='eight columns'
                ),
                html.Img(
                    src='data:image/jpeg;base64,{}'.format(encoded_logo.decode()),
                    className='two columns',
                ),
                html.A(
                    html.Button(
                        "About the DSS",
                        id="about"

                    ),
                    href="https://github.com/ikerforce/pbo_mental_health",
                    className="two columns"
                )
            ],
            id="header",
            className='row',
        ),

        html.Div(
                [

                    html.Div(

                        [

                        html.Div([
                            html.Div(id='container-button-basic',
                                     children='')
                        ]),

                        html.A(
                            html.Button(
                            "Back to home",
                            id="change_parameters",
                            ),
                        href="http://127.0.0.1:5000/health-center-dss/",
                        ),


                        html.A(
                            html.Button(
                            "Update parameters",
                            id="button-update",
                            ),


                        ),

                        html.Span("", id="update-message", style={"verticalAlign": "middle"}, n_clicks=0),

                    ]

                    ),

                ]

            ),

        html.Div(
            [

                html.Div(
                    [
                        html.Div(
                                html.H5("Current parameter values"),
                                style={'font-weight' : 'bold'}
                            ),
                        
                        html.Div(
                                [
                                
                                html.P(
                                    [   
                                        html.Div(
                                        [
                                                html.Span(
                                                    TOOLTIP_INFO['acute_arrival_slider']['caption'],
                                                    id='tooltip_' + 'acute_arrival_slider',
                                                    style={"cursor": "pointer"},
                                                ),
                                                html.Div(id='current_value_acute_arrival_slider', children=" ({current_value}): ".format(current_value = extract_from_list(parameters_list, 'acute_arrival_slider','current_value'))),
                                                dcc.Input(id='acute_arrival_current', type='number', min=0, max=5, step=0.05, style={'margin-left' : '10px'}, value=extract_from_list(parameters_list, 'acute_arrival_slider','current_value')),
                                            ],
                                            className='row'
                                        ),
                                    ],
                                    className="control_label",
                                    style=parameter_style,
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_INFO['acute_arrival_slider']['description'],
                                    target="tooltip_" + 'acute_arrival_slider',
                                    style=tooltip_style,
                                ),

                            ]
                        ),

                        html.Div(
                                [

                                html.P(
                                    [   
                                        html.Div(
                                        [
                                                html.Span(
                                                    TOOLTIP_INFO['semi_acute_arrival_slider']['caption'],
                                                    id='tooltip_' + 'semi_acute_arrival_slider',
                                                    style={"cursor": "pointer"},
                                                ),
                                                html.Div(id='current_value_semi_acute_arrival_slider', children=" ({current_value}): ".format(current_value = extract_from_list(parameters_list, 'semi_acute_arrival_slider','current_value'))),
                                                dcc.Input(id='semi_acute_arrival_current', type='number', min=0, max=10, step=0.05, style={'margin-left' : '10px'}, value=extract_from_list(parameters_list, 'semi_acute_arrival_slider','current_value')),
                                            ],
                                            className='row'
                                        ),
                                    ],
                                    className="control_label",
                                    style=parameter_style,
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_INFO['semi_acute_arrival_slider']['description'],
                                    target="tooltip_" + 'semi_acute_arrival_slider',
                                    style=tooltip_style,
                                ),

                            ]
                        ),

                        html.Div(
                                [

                                html.P(
                                    [   
                                        html.Div(
                                        [
                                                html.Span(
                                                    TOOLTIP_INFO['regular_arrival_slider']['caption'],
                                                    id='tooltip_' + 'regular_arrival_slider',
                                                    style={"cursor": "pointer"},
                                                ),
                                                html.Div(id='current_value_regular_arrival_slider', children=" ({current_value}): ".format(current_value = extract_from_list(parameters_list, 'regular_arrival_slider','current_value'))),
                                                dcc.Input(id='regular_arrival_current', type='number', min=0, max=10, step=0.05, style={'margin-left' : '10px'}, value=extract_from_list(parameters_list, 'regular_arrival_slider','current_value')),
                                            ],
                                            className='row'
                                        ),
                                    ],
                                    className="control_label",
                                    style=parameter_style,
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_INFO['regular_arrival_slider']['description'],
                                    target="tooltip_" + 'regular_arrival_slider',
                                    style=tooltip_style,
                                ),

                            ]
                        ),

                        html.Div(
                                [

                                html.P(
                                    [
                                        html.Span(
                                            TOOLTIP_INFO['IC_beds']['caption'] + " ({current_value}): ".format(current_value = extract_from_list(parameters_list, 'IC_beds', 'current_value')),
                                            id='tooltip_' + 'IC_beds',
                                            style={"cursor": "pointer"},
                                        ),
                                        dcc.Input(id='IC_beds_current', type='number', min=0, step=1, value=extract_from_list(parameters_list, 'IC_beds', 'current_value'), style={'margin-left' : '10px'}),
                                    ],
                                    className="control_label",
                                    style=parameter_style,
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_INFO['IC_beds']['description'],
                                    target="tooltip_" + 'IC_beds',
                                    style=tooltip_style,
                                ),

                            ]
                        ),

                        html.Div(
                                [

                                html.P(
                                    [
                                        html.Span(
                                            TOOLTIP_INFO['Gesloten_beds']['caption'] + " ({current_value}): ".format(current_value = extract_from_list(parameters_list, 'Gesloten_beds', 'current_value')),
                                            id='tooltip_' + 'Gesloten_beds',
                                            style={"cursor": "pointer"},
                                        ),
                                        dcc.Input(id='Gesloten_beds_current', type='number', min= 0, step=1, value=extract_from_list(parameters_list, 'Gesloten_beds', 'current_value'), style={'margin-left' : '10px'}),
                                    ],
                                    className="control_label",
                                    style=parameter_style,
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_INFO['Gesloten_beds']['description'],
                                    target="tooltip_" + 'Gesloten_beds',
                                    style=tooltip_style,
                                ),

                            ]
                        ),


                        html.Div(
                                [

                                html.P(
                                    [   
                                        html.Div(
                                        [
                                                html.Span(
                                                    TOOLTIP_INFO['TOA_beds']['caption'],
                                                    id='tooltip_' + 'TOA_beds',
                                                    style={"cursor": "pointer"},
                                                ),
                                                html.Div(id='current_value_TOA_beds', children=" ({current_value}): ".format(current_value = extract_from_list(parameters_list, 'TOA_beds','current_value'))),
                                                dcc.Input(id='TOA_beds_current', type='number', min=0, max=10, step=1, style={'margin-left' : '10px'}, value=extract_from_list(parameters_list, 'TOA_beds','current_value')),
                                            ],
                                            className='row'
                                        ),
                                    ],
                                    className="control_label",
                                    style=parameter_style,
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_INFO['TOA_beds']['description'],
                                    target="tooltip_" + 'TOA_beds',
                                    style=tooltip_style,
                                ),

                            ]
                        ),

                        html.Div(
                                [

                                html.P(
                                    [
                                        html.Span(
                                            TOOLTIP_INFO['Open_beds']['caption'] + " ({current_value}): ".format(current_value = extract_from_list(parameters_list, 'Open_beds', 'current_value')),
                                            id='tooltip_' + 'Open_beds',
                                            style={"cursor": "pointer"},
                                        ),
                                        dcc.Input(id='Open_beds_current', type='number', min= 0, step=1, value=extract_from_list(parameters_list, 'Open_beds', 'current_value'), style={'margin-left' : '10px'}),
                                    ],
                                    className="control_label",
                                    style=parameter_style,
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_INFO['Open_beds']['description'],
                                    target="tooltip_" + 'Open_beds',
                                    style=tooltip_style,
                                ),

                            ]
                        ),

                        html.Div(
                                [

                                html.P(
                                    [
                                        html.Span(
                                            TOOLTIP_INFO['HIBZ_beds']['caption'] + " ({current_value}): ".format(current_value = extract_from_list(parameters_list, 'HIBZ_beds', 'current_value')),
                                            id='tooltip_' + 'HIBZ_beds',
                                            style={"cursor": "pointer"},
                                        ),
                                        dcc.Input(id='HIBZ_beds_current', type='number', min= 0, step=1, value=extract_from_list(parameters_list, 'HIBZ_beds', 'current_value'), style={'margin-left' : '10px'}),
                                    ],
                                    className="control_label",
                                    style=parameter_style,
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_INFO['HIBZ_beds']['description'],
                                    target="tooltip_" + 'HIBZ_beds',
                                    style=tooltip_style,
                                ),

                            ]
                        ),

                        html.Div(
                                [

                                html.P(
                                    [
                                        html.Span(
                                            TOOLTIP_INFO['Trace_beds']['caption'] + " ({current_value}): ".format(current_value = extract_from_list(parameters_list, 'Trace_beds', 'current_value')),
                                            id='tooltip_' + 'Trace_beds',
                                            style={"cursor": "pointer"},
                                        ),
                                        dcc.Input(id='Trace_beds_current', type='number', min= 0, step=1, value=extract_from_list(parameters_list, 'Trace_beds', 'current_value'), style={'margin-left' : '10px'}),
                                    ],
                                    className="control_label",
                                    style=parameter_style,
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_INFO['Trace_beds']['description'],
                                    target="tooltip_" + 'Trace_beds',
                                    style=tooltip_style,
                                ),

                            ]
                        ),

                        html.Div(
                                [

                                html.P(
                                    [
                                        html.Span(
                                            TOOLTIP_INFO['SPOCK_beds']['caption'] + " ({current_value}): ".format(current_value = extract_from_list(parameters_list, 'SPOCK_beds', 'current_value')),
                                            id='tooltip_' + 'SPOCK_beds',
                                            style={"cursor": "pointer"},
                                        ),
                                        dcc.Input(id='SPOCK_beds_current', type='number', min= 0, step=1, value=extract_from_list(parameters_list, 'SPOCK_beds','current_value'), style={'margin-left' : '10px'}),
                                    ],
                                    className="control_label",
                                    style=parameter_style,
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_INFO['SPOCK_beds']['description'],
                                    target="tooltip_" + 'SPOCK_beds',
                                    style=tooltip_style,
                                ),

                            ]
                        ),

                        html.Div(
                                [

                                html.P(
                                    [
                                        html.Span(
                                            TOOLTIP_INFO['EBK JTP_beds']['caption'] + " ({current_value}): ".format(current_value = extract_from_list(parameters_list, 'EBK JTP_beds','current_value')),
                                            id='tooltip_' + 'EBK JTP_beds',
                                            style={"cursor": "pointer"},
                                        ),
                                        dcc.Input(id='EBK JTP_beds_current', type='number', min= 0, step=1, value=extract_from_list(parameters_list, 'EBK JTP_beds','current_value'), style={'margin-left' : '10px'}),
                                    ],
                                    className="control_label",
                                    style=parameter_style,
                                ),
                                dbc.Tooltip(
                                    TOOLTIP_INFO['EBK JTP_beds']['description'],
                                    target="tooltip_" + 'EBK JTP_beds',
                                    style=tooltip_style,
                                ),

                            ]
                        ),

                        html.P(
                            [
                                html.Span(
                                    TOOLTIP_INFO['simulation_slider']['caption'],
                                    id='tooltip_' + 'simulation_slider',
                                    style={"cursor": "pointer"},
                                ),
                            ],
                            className="control_label",
                            style=parameter_style,
                        ),

                        dcc.Slider(
                            id='simulation_slider',
                            min=0,
                            max=2,
                            value=1,
                            step=1,
                            marks={0:'Low', 1:'Medium', 2:'High'},
                            className="dcc_control"
                        ),

                    ],
                    className="pretty_container four columns"
                ),
            ],
            className="row"
        ),


    ],
    id="mainContainer",
    style={
        "display": "flex",
        "flex-direction": "column"
    }
)


@app.callback(
    [Output('container-button-basic', 'children'), Output('parameters_list', 'data')],
    Input('button-update', 'n_clicks'),
    State('acute_arrival_current', 'value')
    , State('semi_acute_arrival_current', 'value')
    , State('regular_arrival_current', 'value')
    # , State('IC_beds_current', 'value')
    # , State('Gesloten_beds_current', 'value')
    , State('TOA_beds_current', 'value')
    # , State('Open_beds_current', 'value')
    # , State('HIBZ_beds_current', 'value')
    # , State('Trace_beds_current', 'value')
    # , State('SPOCK_beds_current', 'value')
    # , State('EBK_JTP_beds_current', 'value')
    )
def update_output(n_clicks
    , acute_arrival_current
    , semi_acute_arrival_current
    , regular_arrival_current
    # , IC_beds_current
    # , Gesloten_beds_current
    , TOA_beds_current
    # , Open_beds_current
    # # , HIBZ_beds_current
    # # , Trace_beds_current
    # # , SPOCK_beds_current
    # # , EBK_JTP_beds_current
    ):

    PARAMETERS["acute_arrival_slider"]["current_value"] = acute_arrival_current
    PARAMETERS["semi_acute_arrival_slider"]["current_value"] = semi_acute_arrival_current
    PARAMETERS["regular_arrival_slider"]["current_value"] = regular_arrival_current
    # PARAMETERS["IC_beds"]["current_value"] =  IC_beds_current
    # PARAMETERS["Gesloten_beds"]["current_value"] =  Gesloten_beds_current
    PARAMETERS["TOA_beds"]["current_value"] =  TOA_beds_current
    # PARAMETERS["Open_beds"]["current_value"] =  Open_beds_current
    # PARAMETERS["HIBZ_beds"]["current_value"] =  HIBZ_beds_current
    # PARAMETERS["Trace_beds"]["current_value"] =  Trace_beds_current
    # PARAMETERS["SPOCK_beds"]["current_value"] =  SPOCK_beds_current
    # PARAMETERS["EBK JTP_beds"]["current_value"] =  EBK_JTP_beds_current

    with open('parameters_test.json', 'w') as fp:
        json.dump(PARAMETERS, fp)

    parameters_list = []

    for key in PARAMETERS.keys():

        parameters_list.append([key, PARAMETERS[key]['current_value'], PARAMETERS[key]['default_value']])

    return ['The parameters were updated', parameters_list]


@app.callback([Output('current_value_acute_arrival_slider', 'children'), Output('acute_arrival_current', 'value')]
        , Input('parameters_list', 'data'),)
def update_parameters(parameters_list):

    updated_parameter = [x for x in parameters_list if x[0] == 'acute_arrival_slider'][0][1]

    return [": (" + str(updated_parameter) + ") ", updated_parameter]


@app.callback([Output('current_value_semi_acute_arrival_slider', 'children'), Output('semi_acute_arrival_current', 'value')]
        , Input('parameters_list', 'data'),)
def update_parameters(parameters_list):

    updated_parameter = [x for x in parameters_list if x[0] == 'semi_acute_arrival_slider'][0][1]

    return [": (" + str(updated_parameter) + ") ", updated_parameter]


@app.callback([Output('current_value_regular_arrival_slider', 'children'), Output('regular_arrival_current', 'value')]
        , Input('parameters_list', 'data'),)
def update_parameters(parameters_list):

    updated_parameter = [x for x in parameters_list if x[0] == 'regular_arrival_slider'][0][1]

    return [": (" + str(updated_parameter) + ") ", updated_parameter]


@app.callback([Output('current_value_TOA_beds', 'children'), Output('TOA_beds_current', 'value')]
        , Input('parameters_list', 'data'),)
def update_parameters(parameters_list):

    updated_parameter = [x for x in parameters_list if x[0] == 'TOA_beds'][0][1]

    return [": (" + str(updated_parameter) + ") ", updated_parameter]


# Main
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True, port=5001)
