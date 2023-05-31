import sys
sys.path.append("../")
from dashboard_functions import *

# Import required libraries
import os
import pickle
import copy
import datetime as dt
import math
import base64
import plotly.graph_objects as go # or plotly.express as px
import json

import requests
import pandas as pd
import numpy as np
from flask import Flask
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State

Blues = ['rgb(0,0,128)', '  rgb(0,0,255)', 'rgb(65,105,225)', 'rgb(70,130,180)', 'rgb(100,149,237)', 'rgb(30,144,255)', 'rgb(135,206,250)', 'rgb(135,206,235)', 'rgb(0,191,255)']

import random

random.seed(221032)

# Multi-dropdown options
from controls import TOOLTIP_INFO

with open('parameters.json') as json_file:
    PARAMETERS = json.load(json_file)

app = dash.Dash(__name__, routes_pathname_prefix='/vulture/')
server = app.server


# Tooltip settings
tooltip_style = {"font-size": "80%", 'width': '300px', 'border':'solid black 1px', 'background-color' : 'white'}

# Parameter settings
parameter_style = {"margin-top": "2px", "margin-left" : "40px"}

# Logo formatting
logo_filename = os.getcwd() + '/images/VU_logo.png' # replace with your own image
encoded_logo= base64.b64encode(open(logo_filename, 'rb').read())

driving_cost_per_mile = 0.041
charging_cost_per_mile = 0.0388
construction_cost_per_station = 5000
maintenance_fee_per_charger = 500
no_assignment_penalty = 5000
exceed_range_penalty = 5000


penn_data = {"name": "Pennsylvania",
    "min_lat": 39.7199,
    "max_lat": 42.5167,
    "min_lng": -80.5243,
    "max_lng": -74.707}



def prepare_vehicles(path="../data/vehicles.csv"):
    vehicles = pd.read_csv(path, sep="\t")
    vehicles["ev_x"] = penn_data["min_lng"] + (vehicles["ev_x"] / 290) * np.abs(penn_data["min_lng"] - penn_data["max_lng"])
    vehicles["ev_y"] = penn_data["min_lat"] + (vehicles["ev_y"] / 150) * np.abs(penn_data["min_lat"] - penn_data["max_lat"])
    return vehicles

def prep_data(path="../data/solution.csv"):
    location_data = pd.read_csv(path, sep="\t")
    location_data["station_x"] = penn_data["min_lng"] + (location_data["station_x"] / 290) * np.abs(penn_data["min_lng"] - penn_data["max_lng"])
    location_data["station_y"] = penn_data["min_lat"] + (location_data["station_y"] / 150) * np.abs(penn_data["min_lat"] - penn_data["max_lat"])
    return location_data

location_data = prep_data()
vehicles = prepare_vehicles()

# Create global chart template
mapbox_access_token = 'pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w'

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
        accesstoken=mapbox_access_token,
        style="light",
        center=dict(
            lon=-78.05,
            lat=42.54
        ),
        zoom=7,
    )
)

def create_input(param_name, v):
    return  html.Div(
            [

            html.P(
                [
                    html.Span(
                        TOOLTIP_INFO[param_name]['caption'] + " ({current_value}): ".format(current_value = v),
                        id='tooltip_' + param_name,
                        style={"cursor": "pointer"},
                    ),
                    dcc.Input(id=f'{param_name}_current', type='number', min=0, step=1, value=v, style={'margin-left' : '10px'}),
                ],
                className="control_label",
                style=parameter_style,
            ),
            dbc.Tooltip(
                TOOLTIP_INFO[param_name]['description'],
                target="tooltip_" + param_name,
                style=tooltip_style,
            ),

        ]
    )


# Create app layout
app.layout = html.Div(
    [
        dcc.Store(id='new_location_data'),
        html.Div(
            [
                html.Div(
                    [
                        html.H2(
                            'VUlture',

                        ),
                        html.H4(
                            'EV Station location assigner',
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
                        "About",
                        id="about"

                    ),
                    href="https://github.com/ikerforce/mopta",
                    className="two columns"
                )
            ],
            id="header",
            className='row',
        ),

        html.Div(
                [

                    html.Div(

                        [html.A(
                            html.Button(
                            "Reset to default",
                            id="change_parameters",
                            ),
                        href="http://127.0.0.1:5000/vulture/",
                        ),


                        html.A(
                            html.Button(
                            "Change default values",
                            id="change_default",
                            ),
                        href="http://127.0.0.1:5001/parameters/",
                        ),]
                    ),
                ]
            ),

        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                                html.H5("Control panel"),
                                style={'font-weight' : 'bold'}
                            ),
                                                
                        create_input("max_chargers", 8),
                        create_input("max_stations", 600),
                        create_input("max_range", 250),
                        create_input("mean_range", 100),
                        create_input("build_cost", 5000),
                        create_input("maintenance_cost", 500),
                        create_input("charging_cost", 0.0388),
                        create_input("driving_cost", 0.041),

                        html.P(
                            [
                                html.Span(
                                    TOOLTIP_INFO['service_level_slider']['caption'],
                                    id='tooltip_' + 'service_level_slider',
                                    style={"cursor": "pointer"},
                                ),
                            ],
                            className="control_label",
                            style={"margin-top": "2px"}
                        ),
                        dbc.Tooltip(
                            TOOLTIP_INFO['service_level_slider']['description'],
                            target="tooltip_" + 'service_level_slider',
                            style=tooltip_style,
                        ),
                        dcc.Slider(
                            id='service_level_slider',
                            min=0,
                            max=0.9,
                            value=0,
                            step=0.1,
                            marks=dict([[x, str(int(100 * x)) + '%'] for x in np.arange(0.1, 1, 0.2)]),
                            className="dcc_control",
                        ),

                        html.Div(children=[
                            html.A(
                                html.Button(
                                "Run simulation",
                                id="run_simulation",
                                ),
                            ),


                            html.A(
                                html.Button(
                                "Refresh",
                                id="refresh_simulation",
                                ),
                            ),
                        ]),

                        html.Div(
                            [
                                dcc.Graph(id='simulation_cost')
                            ],
                            className='pretty_container twelve columns',
                        ),

                    ],
                    className="pretty_container four columns"
                ),
                html.Div(
                    [
                        html.Div(
                            [

                                html.Div(
                                    [
                                        html.Div(
                                            [
                                               html.P(
                                                    [
                                                        html.Span(
                                                            TOOLTIP_INFO['cost_text']['caption'],
                                                            id='tooltip_' + 'cost_text',
                                                            style={"cursor": "pointer"},
                                                        ),
                                                    ],
                                                    className="control_label",
                                                    style={"margin-top": "2px"}
                                                ),
                                                dbc.Tooltip(
                                                    TOOLTIP_INFO['cost_text']['description'],
                                                    target="tooltip_" + 'cost_text',
                                                    style=tooltip_style,
                                                ),
                                                html.H6(
                                                    id="cost_text",
                                                    className="info_text"
                                                )
                                            ],
                                            id="patients",
                                            className="pretty_container"
                                        ),

                                        html.Div(
                                            [

                                                html.P(
                                                    [
                                                        html.Span(
                                                            TOOLTIP_INFO['n_stations_text']['caption'],
                                                            id='tooltip_' + 'n_stations_text',
                                                            style={"cursor": "pointer"},
                                                        ),
                                                    ],
                                                    className="control_label",
                                                    style={"margin-top": "2px"}
                                                ),
                                                dbc.Tooltip(
                                                    TOOLTIP_INFO['n_stations_text']['description'],
                                                    target="tooltip_" + 'n_stations_text',
                                                    style=tooltip_style,
                                                ),
                                                html.H6(
                                                    id="n_stations_text",
                                                    className="info_text"
                                                )
                                            ],
                                            id="n_stations_textID",
                                            className="pretty_container"
                                        ),
                                        
                                        html.Div(
                                            [
                                                html.P(
                                                    [
                                                        html.Span(
                                                            TOOLTIP_INFO['n_chargers_text']['caption'],
                                                            id='tooltip_' + 'n_chargers_text',
                                                            style={"cursor": "pointer"},
                                                        ),
                                                    ],
                                                    className="control_label",
                                                    style={"margin-top": "2px"}
                                                ),
                                                dbc.Tooltip(
                                                    TOOLTIP_INFO['n_chargers_text']['description'],
                                                    target="tooltip_" + 'n_chargers_text',
                                                    style=tooltip_style,
                                                ),
                                                html.H6(
                                                    id="n_chargers_text",
                                                    className="info_text"
                                                )
                                            ],
                                            id="toa_los",
                                            className="pretty_container"
                                        ),

                                        html.Div(
                                            [
                                                html.P(
                                                    [
                                                        html.Span(
                                                            TOOLTIP_INFO['avg_chargers_text']['caption'],
                                                            id='tooltip_' + 'avg_chargers_text',
                                                            style={"cursor": "pointer"},
                                                        ),
                                                    ],
                                                    className="control_label",
                                                    style={"margin-top": "2px"}
                                                ),
                                                dbc.Tooltip(
                                                    TOOLTIP_INFO['avg_chargers_text']['description'],
                                                    target="tooltip_" + 'avg_chargers_text',
                                                    style=tooltip_style,
                                                ),
                                                html.H6(
                                                    id="avg_chargers_text",
                                                    className="info_text"
                                                )
                                            ],
                                            id="something",
                                            className="pretty_container"
                                        ),
                                    ],
                                    id="tripleContainer",
                                )

                            ],
                            id="infoContainer",
                            className="row"
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Graph(
                                            id='main_map',
                                        )
                                    ],
                                    id="countGraphContainer2",
                                    className="pretty_container twelve columns"
                                )
                            ],
                            className="row"
                        )
                    ],
                    id="rightCol",
                    className="pretty_container eight columns"
                )
            ],
            className="row"
        ),
        
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='cost_breakdown')
                    ],
                    className='pretty_container seven columns',
                ),
                html.Div(
                    [
                        dcc.Graph(id='waiting_histogram_graph')
                    ],
                    className='pretty_container five columns',
                ),
            ],
            className='row'
        ),
        
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id='chargers_graph')
                    ],
                    className='pretty_container four columns',
                ),
                html.Div(
                    [
                        dcc.Graph(id='driving_graph')
                    ],
                    className='pretty_container four columns',
                ),
                html.Div(
                    [
                        dcc.Graph(id='charging_graph')
                    ],
                    className='pretty_container four columns',
                ),
            ],
            className='row'
        ),
    ],
    id="mainContainer",
    style={
        "display": "flex",
        "flex-direction": "column"
    }
)




@app.callback(Output('new_location_data', 'data'),
            [Input('service_level_slider', 'value'), Input('run_simulation', 'n_clicks'), ])
def update_new_location_data(service_level_slider, n_clicks):

    location_data = prep_data().to_dict(orient="records")

    return location_data


@app.callback(Output('n_stations_text', 'children'),
              [Input('new_location_data', 'data')])
def update_n_stations_text(new_location_data):

    solution = pd.DataFrame().from_dict(new_location_data)

    return str(solution.shape[0])


@app.callback(Output('cost_text', 'children'),
              [Input('new_location_data', 'data')])
def update_n_stations_text(new_location_data):

    solution = pd.DataFrame().from_dict(new_location_data)

    return "${:,}".format(np.round(solution.station_cost.sum(), 2))


@app.callback(Output('n_chargers_text', 'children'),
              [Input('new_location_data', 'data')])
def update_n_stations_text(new_location_data):

    solution = pd.DataFrame().from_dict(new_location_data)

    return str(solution.n_chargers.sum())


@app.callback(Output('avg_chargers_text', 'children'),
              [Input('new_location_data', 'data')])
def update_n_stations_text(new_location_data):

    solution = pd.DataFrame().from_dict(new_location_data)

    return str(np.round(solution.n_chargers.sum() / solution.shape[0], 2))



@app.callback(Output('cost_breakdown', 'figure'),
              [Input('new_location_data', 'data')])
def make_main_cost_breakdown(new_location_data):

    layout_individual = copy.deepcopy(layout)

    solution = pd.DataFrame().from_dict(new_location_data)

    mc = solution.n_chargers.sum() * maintenance_fee_per_charger
    bc = solution.shape[0] * construction_cost_per_station
    dc = solution.driving_cost.sum()
    cc = solution.charging_cost.sum()

    cost_types = ['Maintenance', 'Construction', 'Driving cost', 'Charging cost']

    cost_totals = [mc, bc, dc, cc]

    traces = [dict(
        type='pie',
        labels=cost_types,
        values=cost_totals,
        name='Cost breakdown',
        text=['', '', ''],
        hoverinfo="value+text+percent",
        textinfo="label+percent+name",
        hole=0.5,
        marker=dict(
            colors=[Blues[8], Blues[6], Blues[4], Blues[2]]
        ),
        # domain={"x": [0.1, 0.9], 'y':[0.1, 0.9]},
    )]

    layout_individual['title'] = 'Breakdown of costs'

    figure = dict(data=traces, layout=layout_individual)
    return figure


# Selectors, main graph -> aggregate graph
@app.callback(Output('charging_graph', 'figure'),
              [Input('new_location_data', 'data')])
def make_aggregate_figure(new_location_data):

    layout_aggregate = copy.deepcopy(layout)

    solution = pd.DataFrame().from_dict(new_location_data)

    data = [
        dict(
            type='histogram',
            # mode='lines+markers',
            name='Charging cost distribution',
            # histnorm='probability density',
            x=solution['charging_cost'],
            line=dict(
                shape="spline",
                smoothing="2",
                color='#F9ADA0'
            )
        ),
    ]

    layout_aggregate['title'] = 'Disitribution of charging cost'
    layout_aggregate['xaxis'] = {'title' : 'Charging cost', 'range' : [0, solution.charging_cost.max()]}
    layout_aggregate['yaxis'] = {'title' : 'Stations'}
    layout_aggregate['margin'] = dict(l=45, b=35)

    figure = dict(data=data, layout=layout_aggregate)
    return figure


# Selectors, main graph -> aggregate graph
@app.callback(Output('driving_graph', 'figure'),
              [Input('new_location_data', 'data')])
def make_aggregate_figure(new_location_data):

    layout_aggregate = copy.deepcopy(layout)

    solution = pd.DataFrame().from_dict(new_location_data)

    data = [
        dict(
            type='histogram',
            # mode='lines+markers',
            name='Driven distance distribution',
            # histnorm='probability density',
            x=solution['distance'],
            line=dict(
                shape="spline",
                smoothing="2",
                color='#F9ADA0'
            )
        ),
    ]

    layout_aggregate['title'] = 'Disitribution of driven distance'
    layout_aggregate['xaxis'] = {'title' : 'Distance', 'range' : [0, solution.distance.max()]}
    layout_aggregate['yaxis'] = {'title' : 'Stations'}
    layout_aggregate['margin'] = dict(l=45, b=35)

    figure = dict(data=data, layout=layout_aggregate)
    return figure

# Selectors, main graph -> aggregate graph
@app.callback(Output('chargers_graph', 'figure'),
              [Input('new_location_data', 'data')])
def make_aggregate_figure(new_location_data):

    layout_aggregate = copy.deepcopy(layout)

    solution = pd.DataFrame().from_dict(new_location_data)

    data = [
        dict(
            type='histogram',
            # mode='lines+markers',
            name='Number of chargers distribution',
            # histnorm='probability density',
            x=solution['n_chargers'],
            line=dict(
                shape="spline",
                smoothing="2",
                color='#F9ADA0'
            )
        ),
    ]

    layout_aggregate['title'] = 'Disitribution of chargers'
    layout_aggregate['xaxis'] = {'title' : 'Chargers', 'range' : [0, solution.n_chargers.max()]}
    layout_aggregate['yaxis'] = {'title' : 'Stations'}
    layout_aggregate['margin'] = dict(l=45, b=35)

    figure = dict(data=data, layout=layout_aggregate)
    return figure



# Selectors, main graph -> aggregate graph
@app.callback(Output('simulation_cost', 'figure'),
              [Input('new_location_data', 'data')])
def make_aggregate_figure(new_location_data):

    layout_aggregate = copy.deepcopy(layout)

    solution = pd.DataFrame().from_dict(new_location_data)

    xs = np.arange(500)
    rand_xs = xs + np.random.normal(0, 1, size=xs.shape)
    ys = 400-np.log(10+rand_xs)

    data = [
        dict(
            type='line',
            # mode='lines+markers',
            name='Simulation cost',
            # histnorm='probability density',
            x=xs,
            y=ys,
            line=dict(
                shape="spline",
                smoothing="2",
                color='#F9ADA0'
            )
        ),
    ]

    layout_aggregate['title'] = 'Simulation cost'
    layout_aggregate['xaxis'] = {'title' : 'Generation', 'range' : [0, np.max(xs)]}
    layout_aggregate['yaxis'] = {'title' : 'Cost'}
    layout_aggregate['margin'] = dict(l=45, b=35)

    figure = dict(data=data, layout=layout_aggregate)
    return figure

# Selectors -> count graph
@app.callback(Output('main_map', 'figure'),
              [Input('new_location_data', 'data')])
def make_map_figure(new_location_data):

    new_location_data = pd.DataFrame().from_dict(new_location_data)

    fig = go.Figure(go.Scattermapbox(
            lat=vehicles['ev_y'].values,
            lon=vehicles['ev_x'].values,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=5,
                color="black",
                opacity=1
            ),
            showlegend=False
        ))


    fig.update_traces(hoverinfo="none", hovertemplate=None)


    fig.add_trace(go.Scattermapbox(
            lat=new_location_data['station_y'].values,
            lon=new_location_data['station_x'].values,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                color="#80B767",
                opacity=1
            ),
            text=new_location_data['station_cost'].values,
            showlegend=False
        ))

    fig.update_layout(
        autosize=True,
        hovermode='closest',
        margin=dict(t=0, b=0, l=0, r=0),
        height=900,
        mapbox=dict(
            style='basic',
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=40.538,
                lon=-77.448
            ),
            pitch=0,
            zoom=7
        ),
    )

    return fig


# Main
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
