#pip install dash==0.35.1  # The core dash backend
#pip install dash-html-components==0.13.4  # HTML components
#pip install dash-core-components==0.42.1  # Supercharged components
#pip install dash-table==3.1.11  # Interactive DataTable component (new!)

from tqdm import tqdm
import numpy as np
import pandas as pd
import mplcursors
from six import iteritems
from collections import defaultdict, Counter
from dash.dependencies import Input, Output, State
from datetime import datetime, timedelta

import dash
import dash_table
import dash_core_components as dcc
import plotly.graph_objs as go
import dash_html_components as html
import math
import os
import os.path
import flask
import shutil
from tempfile import mkstemp

import utils, metrics, plottery
debug=False

# Read Input
data_file = 'user_user_time'
clabels = ('Retweet', 'Citation', 'Response')
clmap = dict(enumerate(clabels, 1))
df_base = pd.read_csv(data_file, names=['c', 'u', 'v', 'ts'], comment='#', sep=' ').sort_values(by=['ts'])
min_time, max_time = int(df_base.ts.min()), int(df_base.ts.max())

# Start Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Extract and discretize subset
min_time_dt = datetime.fromtimestamp(min_time)
max_time_dt = datetime.fromtimestamp(max_time)
step = int((min_time_dt + timedelta(days=1)).timestamp() - min_time)
marks = {i: datetime.fromtimestamp(i).strftime("%d %b") for i in range(min_time, max_time, step)}
marks[min_time] = min_time_dt.strftime("%d%b %Y %H:%M")
marks[max_time] = max_time_dt.strftime("%d%b %Y %H:%M")

tmp_dir = 'tmp'

percentage_data = html.Div(id='dp', children=[
        html.Div(id='dp-tag', children=['Select Data From Time Range']),
        html.Div(id='ds-div',
            children=dcc.RangeSlider(
             id='ds-slider',
             min=min_time,
             max=max_time,
             marks=marks,
             value=[min_time, max_time],
            ),
            style=dict(width="97.5%", marginLeft="0.7%")
       )],
    )


discrete_time = html.Div(id='dt', children=[
        html.Div(id='dt-title-div', children=[
            html.Label('Discretization'), html.Label('Time Bin Width:')],
            style = dict(
                width='8%',
                display='table-cell',
            )
            ),
        html.Div(id='dt-days-div', children=[
            html.Label('Days'),
            dcc.Input(id='dt-days', type='number', value=0, min=0, max=59)],
            style = dict(
                width='10%',
                display='table-cell',
            )
        ),
        html.Div(id='dt-hours-div', children=[
            html.Label('Hours'),
            dcc.Input(id='dt-hours', type='number', value=0, min=0, max=23)],
            style = dict(
                width='10%',
                display='table-cell',
            )
        ),
        html.Div(id='dt-minutes-div', children=[
            html.Label('Minutes'),
            dcc.Input(id='dt-minutes', type='number', value=0, min=0, max=59)],
            style = dict(
                width='10%',
                display='table-cell',
            )
        ),
        html.Div(id='dt-seconds-div', children=[
            html.Label('Seconds'),
            dcc.Input(id='dt-seconds', type='number', value=0, min=0, max=59)],
            style = dict(
                width='10%',
                display='table-cell',
            )
            ),
        ],
        style = dict(
            width='60%',
            marginTop='3.0%',
            marginLeft='19.4%',
            display='table',
        ))


plot_type = html.Div(id='plot-type', children=[
    html.Label('Plot Type'),
    dcc.Dropdown(id='plot-type-dropdown',
        options=[{'label': '1-Dimensional', 'value': '1D'},
                 {'label': '2-Dimensional', 'value': '2D'},
                 {'label': 'Plots with 2 Measures', 'value': '2M'}])],
                 style={'marginTop':'1%'})


plots = {'1D':  [{'label': '|W(u, .)| = time number of node u ∈ V', 'value': 'time_number_of'}, # NodeStream measures
                 {'label': '|W(u, .)|/|T| = time coverage of node u ∈ V (in %)', 'value': 'time_coverage_of'},
                 {'label': '|W(., t)| = node number at time instant t ∈ T (in nodes)', 'value': 'node_number_at'},
                 {'label': '|W(., t)|/|V| = node density at time instant t ∈ T (in %)', 'value': 'node_density_at'},
                 {'label': '|E(u, ., .)| = neighbour number of node', 'value': 'neighbor_number_of'},
                 {'label': '|E(u, ., .)| = |E(u, ., .)|/|E\'(u, ., .)| neighbour coverage of node', 'value':'neighor_coverage_of'},
                 {'label': '|E(., ., t)| = link number at time instant', 'value':'link_number_at'},
                 {'label': '|E(., ., t)|/|E\'(., ., t)| = link density at time instant t ∈ T (in %)', 'value': 'link_density_at'}],
         '2D':  [{'label': '|E(u, v, .)| = link number of node couple', 'value': 'link_number_of_node_couple'},
                 {'label': '|E(u, v, .)|/|E\'(u, v, .)| = link coverage of node', 'value': 'link_coverage_of_node_couple'},
                 {'label': '|E(u, ., t)| = neighbour number of node u ∈ V at time instant', 'value': 'neighbor_number_of_at'},
                 {'label': '|E(u, ., t)|/|E\'(u, ., t)| = neighbour density of node u ∈ V at time instant', 'value': 'neighbor_density_of_at'}]}

plots['2M'] = plots['1D']

directional = {'neighbor_number_of',
               'neighor_coverage_of',
               'link_number_of_node_couple',
               'link_coverage_of_node_couple',
               'link_coverage_of_node_couple',
               'neighbor_number_of_at',
               'neighbor_density_of_at'}

coverage_measures = {'time_coverage_of', 'node_density_at', 'neighor_coverage_of', 'link_density_at', 'link_coverage_of_node_couple', 'neighbor_density_of_at'}

v_times_v_plots = {'link_number_of_node_couple', 'link_coverage_of_node_couple'}

v_times_t_plots = {'neighbor_number_of_at', 'neighbor_density_of_at'}

time_plots = {'node_number_at', 'node_density_at', 'link_number_at', 'link_density_at'}

layouts = [{'label': 'Spring-Layout', 'value': 'spring'},
           {'label': 'Spectral-Layout', 'value': 'spectral'},
           {'label': 'Kamada-Kawai-Layout', 'value': 'kamada-kawai'},
           {'label': 'Random-Layout', 'value': 'random'},
           {'label': 'Shell-Layout', 'value': 'shell'}]

measures = html.Div(id='measures', children=[
    html.Label('Measures'),
    dcc.Dropdown(id='measures-dropdown',  options=[])])

plot_options = html.Div(id='Plot-Options', children=[
    html.Label('Options'),
    dcc.Dropdown(id='plot-options-dropdown',  options=[], multi=True)],
    )

app.layout = html.Div(id='base-html', children=[
    percentage_data,
    html.Div(id='blank', style=dict(height="10%")),
    discrete_time,
    html.Button(id='stats-button', n_clicks=0, children='Print Stats', style={'marginLeft':'46%', 'marginTop':'1%'}),
    html.Div(id='stats', children=[], style={'marginTop':'1%', 'width': '100%'}),
    plot_type,
    measures,
    plot_options,
    html.Button(id='figure-button', n_clicks=0, children='Plot Figure', style={'marginLeft':'46%', 'marginTop':'1.3%'}),
    html.Div(id='figure-plot', children=[], style={'marginTop':'1.5%'}),
    html.Div(id='download-div', children=[
        html.A(id='download-button', n_clicks=0, children='Download .csv', style={'marginLeft':'47%', 'marginTop':'1.3%'})
        ], style={'display': 'none'})          
        ])

@app.callback(
    dash.dependencies.Output('dp-tag', 'children'),
    [dash.dependencies.Input('ds-slider', 'value')])
def update_percentage_score(value):
    global df_base_index
    perc = "100"
    if value:
        tmin, tmax = value
        df_base_index = ((df_base.ts <= tmax) & (df_base.ts >= tmin))
        perc = str(round((df_base_index.sum()*100.0/float(df_base.shape[0])), 2))
    return 'Select Data From Range [' + perc + '%]'


@app.callback(
    dash.dependencies.Output('stats', 'children'),
    [dash.dependencies.Input('stats-button', 'n_clicks'),
     dash.dependencies.Input('ds-slider', 'value'),
     dash.dependencies.Input('dt-seconds', 'value'),
     dash.dependencies.Input('dt-minutes', 'value'),
     dash.dependencies.Input('dt-hours', 'value'),
     dash.dependencies.Input('dt-days', 'value')])
def print_stats(n_clicks, data_percentage, s, m, h, d):
    if not n_clicks:
        return
    step = timedelta(days=d, seconds=s, minutes=m, hours=h)
    if step:
        stream_graphs, _ = get_sg(data_percentage, step)
        # stats:
        # - Number of nodes
        # - Number of links
        # - Number of time-instants
        # - Number of Interactions
        def as_percentage(item):
            return {c: "{:.2f}%".format(e*100.0) for c, e in iteritems(item)}

        def add_units(item, unit):
            return {c: " ".join([str(e), unit]) for c, e in iteritems(item)}

        children, data = [], {}
        data['Categories'] = clmap
        data['Number of Nodes'] = metrics.stats_number_of_nodes(stream_graphs)
        data['Number of Unique Interactions'] = metrics.stats_number_of_links(stream_graphs)
        data['Number of Time-Instants'] = metrics.stats_number_of_time_instants(stream_graphs)
        data['Number of Interactions'] = metrics.stats_number_of_interactions(stream_graphs)
        data['Node Stream Size'] = add_units(metrics.stats_node_stream_size(stream_graphs), "(time-instants)")
        data['Node Stream Coverage (%)'] = as_percentage(metrics.stats_node_stream_coverage(stream_graphs))
        data['Link Stream Size'] = add_units(metrics.stats_link_stream_size(stream_graphs), "(time-instants)")
        data['Link Stream Coverage (%)'] = as_percentage(metrics.stats_link_stream_coverage(stream_graphs))

        order = ['Categories',
                 'Number of Time-Instants',
                 'Number of Nodes',
                 'Number of Interactions',
                 'Number of Unique Interactions',
                 'Node Stream Size',
                 'Node Stream Coverage (%)',
                 'Link Stream Size',
                 'Link Stream Coverage (%)']
        df = pd.DataFrame(data, columns=order)
        return dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict("rows"),
            style_cell={'textAlign': 'left'})
    else:
        return "Please set a descretization step."

@app.callback(
    dash.dependencies.Output('measures-dropdown', 'options'),
    [dash.dependencies.Input('plot-type-dropdown', 'value'),
     dash.dependencies.Input('measures-dropdown', 'value')])
def add_measures(value, selections):
    if value:
        if selections is not None:
            if value == '2M' and isinstance(selections, list):
                if len(selections) == 1:
                    if selections[0] in time_plots:
                        # time plots
                        return [elem for elem in plots['2M'] if elem['value'] in time_plots]
                    else:
                        # vertex plots
                        return [elem for elem in plots['2M'] if elem['value'] not in time_plots]
                elif len(selections) >= 2:
                    return [elem for elem in plots[value] if elem['value'] in selections]
        return plots[value]
    return []

@app.callback(
    dash.dependencies.Output('measures-dropdown', 'multi'),
    [dash.dependencies.Input('plot-type-dropdown', 'value')])
def add_measures_multi(value):
    return (value == '2M')

@app.callback(
    dash.dependencies.Output('plot-options-dropdown', 'options'),
    [dash.dependencies.Input('plot-type-dropdown', 'value'),
     dash.dependencies.Input('measures-dropdown', 'value'),
     dash.dependencies.Input('plot-options-dropdown', 'value')])
def add_measures(value, selections, po_selections):
    if not po_selections:
        po_selections = {}
    else:
        po_selections = set(po_selections)
    if not selections or not value:
        return []

    def filter_options(opt, group):
        return [opt for opt in options if opt['value'] not in group]

    def add_from_group(plot_extra):
        for p in plot_extra:
            if p['value'] in po_selections:
                return [p]
        return plot_extra

    options = [{'label': 'Log (x-axis)', 'value': 'log_x'},
               {'label': 'Log (y-axis)', 'value': 'log_y'}]
    if value == '2M':
        flag_x, flag_y = False, False
        if len(selections) == 1:
            flag_x = (selections[0] in directional)
            flag_y = flag_x
        elif len(selections) == 2:
            flag_x = (selections[0] in directional)
            flag_y = (selections[1] in directional)
        if flag_x:
            direction_extra_x = [{'label': 'In-Direction-x', 'value': 'in-x'},
                                 {'label': 'Out-Direction-x', 'value': 'out-x'},
                                 {'label': 'Uni-Direction-x', 'value': 'both-x'}]

            if len(selections) == 1:
                if 'in-y' in po_selections:
                    direction_extra_x.pop(0)                
                elif 'out-y' in po_selections:
                    direction_extra_x.pop(1)
                elif 'both-y' in po_selections:
                    direction_extra_x.pop(2)
            options += add_from_group(direction_extra_x)
        if flag_y:
            direction_extra_y = [{'label': 'In-Direction-y', 'value': 'in-y'},
                                 {'label': 'Out-Direction-y', 'value': 'out-y'},
                                 {'label': 'Uni-Direction-y', 'value': 'both-y'}]

            if len(selections) == 1:
                if 'in-x' in po_selections:
                    direction_extra_y.pop(0)
                elif 'out-x' in po_selections:
                    direction_extra_y.pop(1)
                elif 'both-x' in po_selections:
                    direction_extra_y.pop(2)

            options += add_from_group(direction_extra_y)
    elif value == '1D':
        if selections in time_plots:
            time_options = [{'label': 'Time-Curve', 'value': 'time_curve'}, {'label': 'Barplot', 'value': 'barplot'}]
            options += add_from_group(time_options)
        options += [{'label': 'Display-Min', 'value':'min'},
                    {'label': 'Display-Max', 'value':'max'},
                    {'label': 'Display-Mean', 'value':'mean'},
                    {'label': 'Display-Median', 'value':'median'}]
        if 'barplot' in po_selections:
            options += [{'label': 'Order by Labels', 'value': 'order_by_labels'}]
    elif value == '2D':
        plot_extra = [{'label': 'Heatmap', 'value': 'heatmap'},
                      {'label': 'Barplot', 'value': 'barplot'}]
        if selections in v_times_v_plots:
            plot_extra += [{'label': 'Weighted Graph', 'value': 'weighted_graph'}]
        if selections in v_times_t_plots:
            plot_extra += [{'label': 'Multiple Curves', 'value': 'multiple_curves'}]
        options += add_from_group(plot_extra)
        if any(pt in po_selections for pt in ['multiple_curves', 'barplot']):
            options += [{'label': 'Display-Min', 'value':'min'},
                        {'label': 'Display-Max', 'value':'max'},
                        {'label': 'Display-Mean', 'value':'mean'},
                        {'label': 'Display-Median', 'value':'median'}]
        if 'barplot' in po_selections:
            options += [{'label': 'Order by Labels', 'value': 'order_by_labels'}]

    if value in ['1D', '2D']:
        direction_extra = [{'label': 'In-Direction', 'value': 'in'},
                           {'label': 'Out-Direction', 'value': 'out'},
                           {'label': 'Uni-Direction', 'value': 'both'}]
        options += add_from_group(direction_extra)
    if 'weighted_graph' in po_selections:
        group = {'log_x', 'log_y'}
        options = filter_options(options, group)
        options += add_from_group(layouts)

    return options

@app.callback(
    dash.dependencies.Output('figure-plot', 'children'),
    [dash.dependencies.Input('figure-button', 'n_clicks')],
    [dash.dependencies.State('plot-type-dropdown', 'value'),
     dash.dependencies.State('measures-dropdown', 'value'),
     dash.dependencies.State('plot-options-dropdown', 'value'),
     dash.dependencies.State('ds-slider', 'value'),
     dash.dependencies.State('dt-seconds', 'value'),
     dash.dependencies.State('dt-minutes', 'value'),
     dash.dependencies.State('dt-hours', 'value'),
     dash.dependencies.State('dt-days', 'value')])
def display_graph(n_clicks,
                  plot_type,
                  measures,
                  plot_options,
                  data_percentage,
                  s, m, h, d):
    if not n_clicks:
        return
    if plot_type and measures:
        step = timedelta(days=d, seconds=s, minutes=m, hours=h)
        if step:
            if plot_type == '2M' and len(measures) == 1 and measures[0] not in directional:
                return "If you choose one measure, it must be directional."
            stream_graphs, date_bins = get_sg(data_percentage, step)
            return plot_router(plot_type, measures, plot_options, stream_graphs, date_bins)
        else:
           return "Please set a descretization step."


def plot_router(plot_type, measures, plot_options, stream_graphs, date_bins):
    xaxis, yaxis, layout, plot_extra_args = dict(), dict(), dict(), dict()
    plot_options = (set() if plot_options is None else set(plot_options))
    xtitle, ytitle, ignore_cl = None, None, False
    
    csv_comment = "# 1 = retweet; 2 = citation; 3 réponse"
    csv_header = ["Category"]

    def measure_1D(measures, direction='both'):
        if measures == 'time_number_of':
            data = metrics.time_number_of(stream_graphs)
            xtitle = 'Nodes'
            ytitle = 'Time Number'
        elif measures == 'time_coverage_of':
            data = metrics.time_coverage_of(stream_graphs)
            xtitle = 'Nodes'
            ytitle = 'Time Coverage'
        elif measures == 'node_number_at':
            data = metrics.node_number_at(stream_graphs)
            xtitle = 'Time'
            ytitle = 'Number of Nodes'
        elif measures == 'node_density_at':
            data = metrics.node_density_at(stream_graphs)
            xtitle = 'Times'
            ytitle = 'Node Density'
        elif measures == 'neighbor_number_of':
            data = metrics.neighbor_number_of(stream_graphs, direction)
            xtitle = 'Time'
            ytitle = 'Neighbor Number'
        elif measures == 'neighor_coverage_of':
            data = metrics.neighor_coverage_of(stream_graphs, direction)
            xtitle = 'Time'
            ytitle = 'Neighor Coverage'
        elif measures == 'link_number_at':
            data = metrics.link_number_at(stream_graphs)
            xtitle = 'Time'
            ytitle = 'Link Number'
        elif measures == 'link_density_at':
            data = metrics.link_density_at(stream_graphs)
            xtitle = 'Time'
            ytitle = 'Link Density'
        return data, xtitle, ytitle

    if plot_type == '1D':
        if 'in' in plot_options:
            direction = 'in'
        elif 'out' in plot_options:
            direction = 'out'
        else:
            direction = 'both'
        data, xtitle, ytitle = measure_1D(measures, direction)
        if measures in directional:
            ytitle += ("" if direction == 'both' else "(" + direction + ")")

        # Set plottery options
        if 'time_curve' in plot_options:
            plot_fun = plottery.scatterplot_line
        elif 'barplot' in plot_options:
            plot_fun = plottery.barplot
        elif measures in time_plots:
            plot_fun = plottery.scatterplot_line
        else:
            plot_fun = plottery.barplot

        if measures in time_plots:
            plot_extra_args['x_map'] = date_bins
            if plot_fun is plottery.barplot:
                plot_extra_args['sort_by_x'] = True
                plot_extra_args['map_x_before_sorting'] = False

        data_csv = [(c, u, v) for c, d in iteritems(data) for u, v in iteritems(d)]
        csv_header += [xtitle, ytitle]

    elif plot_type == '2D':
        if 'in' in plot_options:
            direction = 'in'
        elif 'out' in plot_options:
            direction = 'out'
        else:
            direction = 'both'

        if measures == 'link_number_of_node_couple':
            data = metrics.link_number_of_node_couple(stream_graphs, direction)
            xtitle = 'Node'
            ytitle = 'Node'
            ztitle = 'Link Number'
            xytitle = 'Node Couple'
            data_csv = [(c, x, y, z) for c, d_c in iteritems(data) for (x, y), z in iteritems(d_c)]
        elif measures == 'link_coverage_of_node_couple':
            data = metrics.link_coverage_of_node_couple(stream_graphs, direction)
            xtitle = 'Node'
            ytitle = 'Node'
            ztitle = 'Link Coverage'
            xytitle = 'Node Couple'            
            csv_header += [xtitle, ytitle, ztitle]
            data_csv = [(c, x, y, z) for c, d_c in iteritems(data) for (x, y), z in iteritems(d_c)]
        elif measures == 'neighbor_number_of_at':
            data = metrics.neighbor_number_of_at(stream_graphs, direction)
            xtitle = 'Time'
            ytitle = 'Node'
            ztitle = 'Neighbor Number'
            xytitle = 'Node x Time'
            csv_header += [xtitle, ytitle, ztitle]
            data_csv = [(c, x, y, z) for c, d_c in iteritems(data) for x, d in iteritems(d_c) for y, z in iteritems(d)]
        elif measures == 'neighbor_density_of_at':        
            data = metrics.neighbor_density_of_at(stream_graphs, direction)
            xtitle = 'Time'
            ytitle = 'Node'
            ztitle = 'Neighbor Density'
            xytitle = 'Node x Time'
            data_csv = [(c, x, y, z) for c, d_c in iteritems(data) for x, d in iteritems(d_c) for y, z in iteritems(d)]
        if measures in directional:
            ztitle += ("" if direction == 'both' else " (" + direction + ")")



        # Set plottery options
        has_plot_type_flag = len(plot_options & {'barplot', 'heatmap', 'weighted_graph', 'multiple_curves'})
        if 'barplot' in plot_options:
            plot_fun = plottery.barplot
            xtitle = xytitle
            ytitle = ztitle
        elif 'heatmap' in plot_options:
            plot_fun = plottery.heatmap
            layout['title'] = ztitle
        elif 'weighted_graph' in plot_options or (not has_plot_type_flag and measures in v_times_v_plots):
            plot_fun = plottery.graphplot

            gly = plot_options & {'spring', 'spectral', 'kamada-kawai', 'random', 'shell'}
            if len(gly):
                plot_extra_args['layout'] = list(gly)[0]
            plot_extra_args['direction'] = direction
            layout['title'] = 'Aggregated Graph'
        elif 'multiple_curves' in plot_options or (not has_plot_type_flag and measures in v_times_t_plots):
            plot_fun = plottery.multi_scatterplot
            plot_extra_args['x_map'] = date_bins
            ytitle = ztitle

        if measures in v_times_t_plots:
            if plot_fun is plottery.barplot:
                # Unravel
                data = utils.unravel_time(data, True)
                def date_map(tup):
                    return ":".join([tup[0], str(date_bins[tup[1]])])
                plot_extra_args['x_map'] = date_map
            elif plot_fun is plottery.heatmap:
                data = utils.unravel_time(data)
                plot_extra_args['x_map'] = date_bins

    elif plot_type == '2M':
        if 'in_x' in plot_options:
            direction_x = 'in'
        elif 'out_x' in plot_options:
            direction_x = 'out'
        elif 'both_x' in plot_options:
            direction_x = 'both'
        else:
            direction_x = None
        
        if 'in_y' in plot_options:
            direction_y = 'in'
        elif 'out_y' in plot_options:
            direction_y = 'out'
        elif 'both_y' in plot_options:
            direction_y = 'both'
        else:
            direction_y = None

        if len(measures) == 1:
            if direction_x is None:
                if direction_y is None:
                    direction_x, direction_y = 'in', 'out'
                elif direction_y == 'out':
                    direction_x = 'in'
                else:
                    direction_x = 'out'

            if direction_y is None:
                if direction_x == 'in':
                    direction_y = 'out'
                else:
                    direction_y = 'in'
            measures = [measures[0], measures[0]]
        elif len(measures) == 2:
            if direction_x is None and direction_y is None:
                direction_x = 'both'
                direction_y = 'both'
            elif direction_x is None:
                direction_y = direction_x
            elif direction_y is None:
                direction_x = direction_y

        data_x, base_title, xtitle = measure_1D(measures[0], direction_x)
        if measures[0] in directional:
            xtitle += ("" if direction_x == 'both' else " (" + direction_x + ")")
        data_y, _, ytitle = measure_1D(measures[1], direction_y)
        if measures[1] in directional:
            ytitle += ("" if direction_y == 'both' else " (" + direction_y + ")")

        data = {c: ([data_x[c], data_y[c]]) for c in range(1, len(clabels) + 1)}
        plot_fun = plottery.scatterplot_points
        if measures[0] in time_plots:
            plot_extra_args['text_map'] = date_bins

        data_csv = [(c, k, x[k], y[k]) for c, (x, y) in iteritems(data) for k in set(x.keys()) & set(y.keys())]
        csv_header += [base_title, xtitle, ytitle]

    if plot_fun is plottery.graphplot:
        axis = dict(
            showline=False, # hide axis line, grid, ticklabels and  title
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title='')
        xaxis, yaxis = axis, axis
    else:
        xaxis['title'], yaxis['title'] = xtitle, ytitle
        xaxis['autorange'], yaxis['autorange'] = True, True
    layout['xaxis'], layout['yaxis'] = xaxis, yaxis
    layout['hovermode'] = 'closest'
    
    if 'log_x' in plot_options:
        xaxis['type'] = 'log'
    if 'log_y' in plot_options:
        yaxis['type'] = 'log'
    if 'min' in plot_options:
        plot_extra_args['plot_min'] = True
    if 'max' in plot_options:
        plot_extra_args['plot_max'] = True
    if 'mean' in plot_options:
        plot_extra_args['plot_mean'] = True
    if 'median' in plot_options:
        plot_extra_args['plot_median'] = True
    if 'order_by_labels' in plot_options:
        plot_extra_args['sort_by_x'] = True
#    if ensemble and plot_fun is plottery.barplot:
#        layout['barmode'] = 'group'
    fig = plot_figure(data, (plot_fun, plot_extra_args), layout)

    if isinstance(fig, str):
        return fig
    else:
        prefix = "__".join(["_".join([l.lower() for l in h.split(" ")]) for h in csv_header[1:]])+"__"
        fd, address = mkstemp(suffix=".csv", prefix=prefix, dir=tmp_dir)
        f = open(fd, 'w+')
        f.write(csv_comment + "\n")
        pd.DataFrame(data_csv, columns=csv_header).to_csv(f, header=True, index=False)
        return [fig, html.Div(id='json-data', children=address, style={'display': 'none'})]

def plot_figure(data, plot_fun, layout_args):
    figures = dict()
    base_title = layout_args.pop('title', "")
    for m, data_m in iteritems(data):
        try:
            if base_title != "":
                title = base_title + " [" + clmap[m] + "]"
            else:
                title = clmap[m]

            figure = {
                'data': plot_fun[0](data_m, **plot_fun[1]),
                'layout' : go.Layout(
                    title = title,
                    **layout_args
                )}
            figures[m] = [dcc.Graph(id='figure-'+str(m), figure=figure)]
        except MemoryError:
            return "Memory Error [" + clmap[m] + "]: Please use a smaller a portion of the data for this plot."

    if len(figures) > 1:
        children, default = [], None
        for c, fig in iteritems(figures):
            value = 't_' + str(c)
            if default is None:
                default=value
            children.append(dcc.Tab(label=clmap[c], value=value, children=fig))
        return dcc.Tabs(id="Choose Category", value=default, children=children)
    elif len(figures):
        return figures.values()[0]


@app.callback(
    dash.dependencies.Output('download-div', 'style'),
    [dash.dependencies.Input('figure-plot', 'children')])
def display_download(children):
    return ({'display': 'block'} if bool(children) and not isinstance(children, str) else {'display': 'none'})


@app.callback(
    Output('download-button', 'href'),
   [dash.dependencies.Input('figure-plot', 'children')])
def update_href(children):
    if bool(children) and not isinstance(children, str):
        if len(children) > 1:
            address = children[1]['props']['children']
            path = os.path.normpath(address)
            ra = path.split(os.sep)[-2:]
            return '/{}'.format(os.path.join(ra[0], ra[1]))


@app.server.route('/tmp/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(os.path.join(root_dir, 'tmp'), path)

def get_sg(data_percentage, step):
    df = df_base[(df_base.ts <= data_percentage[1]) & (df_base.ts >= data_percentage[0])]
    df, bins = utils.time_discretizer(df, step)
    date_bins = utils.make_bin_map(bins)
    dfc = utils.categorical_split(df)
    return {c: utils.make_minimal_stream_graph(df) for c, df in iteritems(dfc)}, date_bins


if __name__ == '__main__':
    try:
        os.mkdir(tmp_dir)
    except Exception:
        shutil.rmtree(tmp_dir)
        os.mkdir(tmp_dir)
    app.run_server(port=4050, debug=debug)
    # Remove temporary files
    shutil.rmtree(tmp_dir)
