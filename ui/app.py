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
logo_filename = os.getcwd() + '/images/vulture_logo.jpeg' # replace with your own image
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


def style_results(r):
    result_in_html = \
        html.A(
            [
                html.Section(
                    [
                        html.H2("Station {r['station_ix']}:", className="thumbnail-title"),
                        html.P(r['cost']),
                        html.Div(
                            [
                                dcc.Graph(id="cost_breakdown_station")
                            ],
                            className="article-info-wrapper"
                            )
                    ],
                    className="article-text-wrapper w-clearfix"),
            ],
            className="article w-inline-block w-clearfix"
        )
    return [result_in_html]


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
    # plot_bgcolor="rgb(10,10,10)",
    # paper_bgcolor="rgb(10,10,10)",
    legend=dict(font=dict(size=10), orientation='h'),
    title='Title of the graph',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="dark",
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
                    dcc.Input(id=f'{param_name}_current', type='number', min=0, step=0.0001, value=v, style={'margin-left' : '10px'}),
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
        dcc.Store(id='parameters'),
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
                    className='one column',
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
                    ]
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
                                n_clicks=0
                                ),
                            ),


                            html.A(
                                html.Button(
                                "Refresh",
                                id="refresh_simulation",
                                n_clicks=0
                                ),
                            ),
                        ]),
                        html.Div(id='simulation-status', children=''),
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
                                        ),
                                        dcc.Tooltip(id="graph-tooltip"),
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
                    className='pretty_container four columns',
                ),
                html.Div(
                    [
                        dcc.Graph(id='chargers_graph')
                    ],
                    className='pretty_container three columns',
                ),
                html.Div(
                    [
                        dcc.Graph(id='driving_graph')
                    ],
                    className='pretty_container three columns',
                ),
                html.Div(
                    [
                        dcc.Graph(id='charging_graph')
                    ],
                    className='pretty_container three columns',
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


# create_input("max_chargers", 8),
# create_input("max_stations", 600),
# create_input("max_range", 250),
# create_input("mean_range", 100),
# create_input("build_cost", 5000),
# create_input("maintenance_cost", 500),
# create_input("charging_cost", 0.0388),
# create_input("driving_cost", 0.041),

@app.callback(
    Output('simulation-status', 'children'),
    Input('run_simulation', 'n_clicks'),
    [State("max_chargers_current", 'value'),
    State("max_stations_current", 'value'),
    State("max_range_current", 'value'),
    State("mean_range_current", 'value'),
    State("build_cost_current", 'value'),
    State("maintenance_cost_current", 'value'),
    State("charging_cost_current", 'value'),
    State("driving_cost_current", 'value'),
    State("service_level_slider", "value")]
)
def update_output(n_clicks, max_chargers_current, max_stations_current, max_range_current, mean_range_current, build_cost_current, maintenance_cost_current, charging_cost_current, driving_cost_current, service_level):
    if n_clicks > 0:
        os.system(f"/home/ikerforce/anaconda3/envs/mopta/bin/python ../src/PSO_V2.py --max_chargers {max_chargers_current} --max_stations {max_stations_current} --max_range {max_range_current} --mean_range {mean_range_current} --construction_cost {build_cost_current} --maintenance_cost {maintenance_cost_current} --charging_cost {charging_cost_current} --driving_cost {driving_cost_current} --service_level {service_level}")
        return 'The new simulation is running'
    else:
        return ''


@app.callback(Output('new_location_data', 'data'),
            [Input('run_simulation', 'n_clicks')])
def update_new_location_data(n_clicks):
    print('SOMETHING')

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


@app.callback(Output('parameters', 'data'),
            [Input("max_chargers", "value")
                , Input("max_stations", "value")
                , Input("max_range", "value")
                , Input("mean_range", "value")
                , Input("build_cost", "value")
                , Input("maintenance_cost", "value")
                , Input("charging_cost", "value")
                , Input("driving_cost", "value")])
def update_n_stations_text(max_chargers, max_stations, max_range, mean_range, build_cost, maintenance_cost, charging_cost, driving_cost):

    params = dict(zip("max_chargers, max_stations, max_range, mean_range, build_cost, maintenance_cost, charging_cost, driving_cost".split(", "), [max_chargers, max_stations, max_range, mean_range, build_cost, maintenance_cost, charging_cost, driving_cost]))

    return params


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

    cost_types = ['Charging cost', 'Maintenance', 'Construction', 'Driving cost']

    cost_totals = [cc, mc, bc, dc]

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
    layout_individual['legend'] = dict(yanchor="top", y=0.99, xanchor="left", x=0.85)

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
            marker=dict(color=Blues[6]),
            line=dict(
                shape="spline",
                smoothing="2",
                color=Blues[2]
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
            marker=dict(color=Blues[6]),
            line=dict(
                shape="spline",
                smoothing="2"
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
            marker=dict(color=Blues[6]),
            line=dict(
                shape="spline",
                smoothing="2",
                color=Blues[6]
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
              Input('refresh_simulation', 'n_clicks'))
def make_aggregate_figure(n_clicks):

    layout_aggregate = copy.deepcopy(layout)

    print('hfhdjfhkadsf', n_clicks)

    if n_clicks > 0:
        fitness = np.loadtxt("../results/gen_fitness.csv", delimiter="\t")
        ys_agg = fitness[:,1]
        ys_best = fitness[:,2]
    else:
        ys_agg = np.loadtxt("../results/best_6.txt")[1::]
        ys_best = np.loadtxt("../results/pop_fitness_6_agg.txt")[1::]

    xs = np.arange(ys_agg.shape[0])

    data = [
        dict(
            type='line',
            # mode='lines+markers',
            name='Mean population cost',
            # histnorm='probability density',
            x=xs,
            y=ys_agg,
            line=dict(
                shape="spline",
                smoothing="2",
                color=Blues[2]
            )
        ),
        dict(
            type='line',
            # mode='lines+markers',
            name='Best population cost',
            # histnorm='probability density',
            x=xs,
            y=ys_best,
            line=dict(
                shape="spline",
                smoothing="2",
                color=Blues[4]
            )
        ),
    ]

    layout_aggregate['title'] = 'Simulation cost'
    layout_aggregate['xaxis'] = {'title' : 'Generation', 'range' : [0, 50]}
    layout_aggregate['yaxis'] = {'range' : [np.min(ys_best)-1000, np.max(ys_agg)+1000]}
    layout_aggregate['margin'] = dict(l=45, b=40)
    layout_aggregate['legend'] = dict(yanchor="top", y=0.99, xanchor="left", x=0.65)

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
                size=10,
                color="black",
                opacity=1
            ),
            showlegend=False,
        ))


    fig.update_traces(hoverinfo="none", hovertemplate=None)


    fig.add_trace(go.Scattermapbox(
            lat=new_location_data['station_y'].values,
            lon=new_location_data['station_x'].values,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=12,
                color=Blues[4],
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
            style='light',
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


@app.callback(Output('cost_breakdown_station', 'figure'),
              [Input('new_location_data', 'data')])
def make_main_cost_breakdown(new_location_data):

    layout_individual = copy.deepcopy(layout)

    solution = pd.DataFrame().from_dict(new_location_data)

    mc = solution.n_chargers.sum() * maintenance_fee_per_charger
    bc = solution.shape[0] * construction_cost_per_station
    dc = solution.driving_cost.sum()
    cc = solution.charging_cost.sum()

    cost_types = ['Charging cost', 'Maintenance', 'Construction', 'Driving cost']

    cost_totals = [cc, mc, bc, dc]

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
    layout_individual['legend'] = dict(yanchor="top", y=0.99, xanchor="left", x=0.85)

    figure = dict(data=traces, layout=layout_individual)
    return figure


@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    [Input("main_map", "hoverData"), Input("new_location_data", "data")],
)
def display_hover(hoverData, new_location_data):

    if hoverData is None:
        return False, no_update, no_update

    layout_individual = copy.deepcopy(layout)
    

    pt = hoverData["points"][0]
    bbox = pt["bbox"]
    num = pt["pointNumber"]

    info = new_location_data[num]

    mc = info["n_chargers"] * maintenance_fee_per_charger
    bc = construction_cost_per_station
    dc = info["driving_cost"]
    cc = info["charging_cost"]

    cost_types = ['Charging cost', 'Maintenance', 'Construction', 'Driving cost']

    cost_totals = [cc, mc, bc, dc]
    info['cost'] = "Station cost ${c}".format(c=round(sum(cost_totals),2))

    traces = dict(
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
    )

    layout_individual['title'] = 'Breakdown of costs'
    layout_individual['legend'] = dict(yanchor="top", y=0.99, xanchor="left", x=0.85)

    figure = dict(data=traces, layout=layout_individual)

    children = style_results(info, figure)

    return True, bbox, children


# Main
if __name__ == '__main__':
    app.server.run(debug=True, threaded=True)
