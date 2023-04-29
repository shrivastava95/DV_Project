#required imports
import dash
import requests
from pydash import *
import dash_core_components as dcc
import dash_html_components as html
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from collections import Counter
from datetime import datetime
import pandas as pd
import dash_table as dt
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import numpy as np
from plotly_calplot import calplot
import time
from plotly_calplot.layout_formatter import apply_general_colorscaling
import plotly.express as px

#filter warnings
import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

#helper function for calendar plot
today_date = datetime.today().date()
wait = True
color_init = True

def get_date(int_value):
    date = datetime.utcfromtimestamp(int_value).date()
    return date

def get_date_fig(data, year):
    dates = [
        get_date(submission['creationTimeSeconds'])
        for submission in data['result']
    ]
    dummy_start_date = min(dates)
    dummy_end_date = today_date
    dummy_df = pd.DataFrame({
        "ds": pd.date_range(dummy_start_date, dummy_end_date),
        "value": np.random.randint(low=0, high=1, size=(pd.to_datetime(dummy_end_date) - pd.to_datetime(dummy_start_date)).days + 1,)
    })
    counts = dict()
    for date in dates:
        if date <= today_date:
            counts[date] = counts.get(date, 0) + 1
    for i in range(dummy_df.shape[0]):
        dummy_df.iloc[i, 1] = counts.get(dummy_df.iloc[i, 0].date(), 0)
    dummy_df = dummy_df[dummy_df['ds'].apply(lambda x:x.year) == year]
    
    fig = calplot(
        dummy_df,
        x="ds",
        y="value",
        gap=3.5,
        years_title=True,
        month_lines_width=3, 
        month_lines_color="#999",
        total_height=250,
        showscale=True,
    )
    fig.update_layout(
        yaxis=dict(title_text=f"{year}", titlefont=dict(size=30))
    )
    fig.layout.annotations[0].update(text="Daily Solves")
    max_val = max(dummy_df.iloc[:, 1])
    return fig, max_val

def get_figures_from_user_data(data):
    total_year_options = [i for i in range(2008, today_date.year + 1)]
    user_years_figs = {}
    for y in total_year_options:
        try:
            fig, max_val = get_date_fig(data, y)
            user_years_figs[y] = fig
            user_years_figs[-y] = max_val
        except:
            pass
    return user_years_figs

#define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

card2 = dbc.Card(
    [
        dbc.CardHeader("Submission Languages"),
        dbc.CardBody(dcc.Graph(id='languages-graph')),
    ],
    style={"width": "50%","display": "inline-block", "padding": "10px"},
)

card3 = dbc.Card(
    [
        dbc.CardHeader("Problems Solved by Difficulty"),
        dbc.CardBody(dcc.Graph(id='difficulty-graph')),
    ],
    style={"width": "50%","display": "inline-block", "padding": "10px"},
)

card4 = dbc.Card(
    [
        dbc.CardHeader("Problems Solved by Tag"),
        dbc.CardBody(dcc.Graph(id='tag-graph')),
    ],
    style={"width": "50%","display": "inline-block", "padding": "10px"},
)

card5 = dbc.Card(
    [
        dbc.CardHeader("Time Taken to Solve Problems of Each Difficulty"),
        dbc.CardBody(dcc.Graph(id='time-taken-violin')),
    ],
    style={"width": "100%","display": "inline-block", "padding": "10px"},
)

card6 = dbc.Card(
    [
        dbc.CardHeader("Daily Solves"),
        dbc.CardBody(
            html.Div([
                dcc.Dropdown(
                    id='year-dropdown',
                ),
                dcc.Graph(id='calendar-display'),
                dcc.Markdown(
                    '### Color Scaling',
                    style={
                        'textAlign': 'center',
                        'fontSize': '12px',
                        'fontWeight': 'bold',
                        'marginBottom': '10px',
                        'fontFamily': 'Arial, sans-serif',
                        'color': '#BBB'
                    }
                ),
                dcc.Slider(
                    id='colorscale-slider',
                    min=0,
                    max=60,
                    step=1,
                    value=10,
                    marks={i: str(i) for i in range(0, 61, 5)},
                    tooltip={'always_visible': True, 'placement': 'bottom'}
                )
            ])
        )
    ]
)

card7 = dbc.Card(
    [
        dbc.CardHeader("Verdicts Graph"),
        dbc.CardBody(dcc.Graph(id='verdicts-graph')),
    ],
    style={"width": "50%", "display": "inline-block", "padding": "10px"},
)

card8 = dbc.Card(
    [
        dbc.CardHeader("Total number of contests"),
        dbc.CardBody(dcc.Graph(id='contests-graph')),
    ],
    style={"width": "50%","display": "inline-block", "padding": "10px"},
)

card9 = dbc.Card(
    [
        dbc.CardHeader("Total number of unsolved questions"),
        dbc.CardBody(dcc.Graph(id='unsolved-graph')),
    ],
    style={"width": "50%", "display": "inline-block", "padding": "10px"},
)

#define layout of the app
app.layout = html.Div([
    html.H1(children='Codeforces Submissions'),

    dbc.RadioItems(
        id='handle-selection',
        options=[
            {'label': 'One handle', 'value': 'one'},
            {'label': 'Two handles', 'value': 'two'}
        ],
        value='one'
    ),

    html.Div(id='handle-1-container', children=[
        html.Label('Enter handle 1:'),
        dbc.Input(id='input-1', type='text', placeholder='Enter handle 1'),
        html.Br()
    ], style={'display': 'none'}),

    html.Div(id='handle-2-container', children=[
        html.Label('Enter handle 2:'),
        dbc.Input(id='input-2', type='text',placeholder='Enter handle 2'),
        html.Br()
    ], style={'display': 'none'}),

    html.Button('Submit', id='submit-button', n_clicks=0),

    dbc.Row([card2, card3]),
    dbc.Row([card4, card7]),
    dbc.Row([card8, card9]),
    dbc.Row([card5]),
    dbc.Row([card6])

])

# Define a callback to show/hide the input boxes depending on the selected option
@app.callback(
    [dash.dependencies.Output('handle-1-container', 'style'),
     dash.dependencies.Output('handle-2-container', 'style'),
     dash.dependencies.Output('input-1', 'value'),
     dash.dependencies.Output('input-2', 'value')],
    [dash.dependencies.Input('handle-selection', 'value')]
)
#function to show/hide input boxes
def show_hide_handles(value):
    if value == 'one':
        return {'display': 'block'}, {'display': 'none'}, '', ''
    else:
        return {'display': 'block'}, {'display': 'block'}, '', ''

#app callback and respective functions 1-6
@app.callback(
        Output('year-dropdown', 'options'),
        Input('submit-button', 'n_clicks'),
        dash.dependencies.State('input-1', 'value')
)
def update_user_data(n_clicks, handle):
    if n_clicks > 0 and handle:
        global data
        global wait
        url = 'https://codeforces.com/api/user.status'
        params = {'handle': handle}

        response = requests.get(url, params=params)
        data = response.json()

        user_years_figs = get_figures_from_user_data(data)
        dropdown_options = tuple([{'label': str(year), 'value': year} for year in user_years_figs.keys() if year > 0])
        wait = True
        return dropdown_options
    return {}
    
@app.callback(
        Output('year-dropdown', 'value'),
        Input('submit-button', 'n_clicks'),
        dash.dependencies.State('input-1', 'value'),
)
def update_calendar_scaling_init(n_clicks, handle):
    global wait
    global color_init

    if n_clicks > 0 and handle and color_init:
        time.sleep(1)
        global data
        
        user_years_figs = get_figures_from_user_data(data)
        dropdown_options = [{'label': str(year), 'value': year} for year in user_years_figs.keys() if year > 0]
        color_init = False
        return max(list(user_years_figs.keys()))
    return {}

@app.callback(Output('calendar-display', 'figure'),
            Input('year-dropdown', 'value'),
            Input('colorscale-slider', 'value'),  
            Input('submit-button', 'n_clicks'),
            dash.dependencies.State('input-1', 'value'))
def update_figure(year, slider_value, n_clicks, handle):
    if n_clicks > 0 and handle and year:
        global data

        user_years_figs = get_figures_from_user_data(data)
        fig = apply_general_colorscaling(user_years_figs[year], 0, slider_value)
        return fig
    return {}

@app.callback(
    Output('time-taken-violin', 'figure'),
    Input('submit-button', 'n_clicks'),
    dash.dependencies.State('input-1', 'value')
)
def update_time_taken_violin(n_clicks, handle):
    global wait 
    if n_clicks > 0 and handle:
        time.sleep(1)
        global data
        solved = filter_(data['result'], lambda x: x['verdict'] == 'OK')
        difficulties = group_by(solved, lambda x: x['problem']['index'][0])

        df = pd.DataFrame(columns=['Difficulty', 'Time Taken'])

        for d in difficulties.items():
            for s in d[1]:
                timer = s['timeConsumedMillis'] / 1000 / 60  # convert to minutes
                df_row = pd.DataFrame({'Difficulty': [d[0]], 'Time Taken': [timer]})
                df = pd.concat([df, df_row])

        figure = px.violin(df, x="Difficulty", y="Time Taken", box=True, points="all", hover_data=df.columns,
                           color_discrete_sequence=px.colors.qualitative.Pastel,
                           category_orders={"Difficulty": ["A", "B", "C", "D", "E"]})
        figure.update_layout(title="Time Taken to Solve Problems of Each Difficulty")

        return figure

    return {}

def update_tag_graph(n_clicks, handle):
    global wait
    if n_clicks > 0 and handle:
        time.sleep(1)
        global data
        solved = filter_(data['result'], lambda x: x['verdict'] == 'OK')
        tags = group_by(solved, lambda x: x['problem']['tags'][0] if x['problem']['tags'] else 'None')

        tags_count = {}
        for tag, problems in tags.items():
            tags_count[tag] = len(problems)

        sorted_tags_count = dict(sorted(tags_count.items(), key=lambda item: item[1], reverse=True))

        if len(sorted_tags_count) > 0:
            fig = go.Figure(data=go.Scatterpolar(
                r=list(sorted_tags_count.values()),
                theta=list(sorted_tags_count.keys()),
                fill='toself'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, max(sorted_tags_count.values())]
                    )),
                showlegend=False,
                title='Problems Solved by Tag'
            )

            return fig

    return {}

@app.callback(
    Output('tag-graph', 'figure'),
    Input('submit-button', 'n_clicks'),
    dash.dependencies.State('input-1', 'value')
)
def update_tag_graph_callback(n_clicks, handle):
    return update_tag_graph(n_clicks, handle)

@app.callback(
    Output('difficulty-graph', 'figure'),
    Input('submit-button', 'n_clicks'),
    dash.dependencies.State('input-1', 'value')
)
def update_difficulty_graph(n_clicks, handle):
    global wait
    if n_clicks > 0 and handle:
        time.sleep(1)
        global data
        solved = filter_(data['result'], lambda x: x['verdict'] == 'OK')
        difficulties = group_by(solved, lambda x: x['problem']['index'][0])

        sorted_difficulties = sorted(difficulties.items())

        figure = {
            'data': [{'x': [d[0] for d in sorted_difficulties],
                      'y': [len(d[1]) for d in sorted_difficulties],
                      'type': 'bar',
                      'marker': {'color': '#FDB813'}
                      }],
            'layout': {'title': 'Problems Solved by Difficulty', 'showlegend': False}
        }

        return figure

    return {}

@app.callback(
    Output('languages-graph', 'figure'),
    Input('submit-button', 'n_clicks'),
    dash.dependencies.State('input-1', 'value')
)
def update_languages_graph(n_clicks, handle):
    global wait
    if n_clicks > 0 and handle:
        time.sleep(1)
        global data
        languages = group_by(data['result'], lambda x: x['programmingLanguage'])

        figure = {
            'data': [{'labels': list(languages.keys()),
                      'values': [len(languages[l]) for l in languages],
                      'type': 'pie',
                      'hole': 0.4,
                      'marker': {'colors': ['#FDB813','#B5E61D','#5FA99E','#FDB813','#2CA02C','#FFC0CB']},
                      }],
            'layout': {'title': 'Submission Languages', 'showlegend': True, 'depth': 3}
        }

        return figure

    return {}

@app.callback(
    dash.dependencies.Output('verdicts-graph', 'figure'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-1', 'value'),
     dash.dependencies.State('input-2', 'value'),
     dash.dependencies.State('handle-selection', 'value')]
)
def update_rating_graph(n_clicks, handle1, handle2, handle_type):
    url = 'https://codeforces.com/api/user.status'
    if handle_type == 'one':
        params1 = {'handle': handle1}
        response1 = requests.get(url, params=params1)
        data1 = response1.json()
        verdicts1 = group_by(data1['result'], 'verdict')
        trace1 = go.Bar(
            x=list(verdicts1.keys()),
            y=[len(verdicts1[v]) for v in verdicts1],
            name=handle1
        )
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(trace1, row=1, col=1)
        fig.update_layout(barmode='group')

    else:
        params1 = {'handle': handle1}
        params2 = {'handle': handle2}
        response1 = requests.get(url, params=params1)
        response2 = requests.get(url, params=params2)
        data1 = response1.json()
        data2 = response2.json()
        verdicts1 = group_by(data1['result'], 'verdict')
        verdicts2 = group_by(data2['result'], 'verdict')
        trace1 = go.Bar(
            x=list(verdicts1.keys()),
            y=[len(verdicts1[v]) for v in verdicts1],
            name=handle1
        )
        trace2 = go.Bar(x=list(verdicts2.keys()),
        y=[len(verdicts2[v]) for v in verdicts2],
        name=handle2
        )
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=1)
        fig.update_layout(barmode='group')
    
    return fig

@app.callback(
    dash.dependencies.Output('contests-graph', 'figure'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-1', 'value'),
     dash.dependencies.State('input-2', 'value'),
     dash.dependencies.State('handle-selection', 'value')]
)
def update_rating_graph(n_clicks, handle1, handle2, handle_type):
    url = 'https://codeforces.com/api/user.status'
    if handle_type == 'one':
        params1 = {'handle': handle1}
        response1 = requests.get(url, params=params1)
        data1 = response1.json()
        contests1 = group_by(data1['result'], lambda x: (x['contestId'], x.get('name', 'N/A')))
        total_contests1 = len(contests1.keys())
        trace3 = go.Bar(
            x=[handle1],
            y=[total_contests1],
            name="Total Contests"
        )
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(trace3, row=1, col=1)
        fig.update_layout(barmode='group')

    else:
        params1 = {'handle': handle1}
        params2 = {'handle': handle2}
        response1 = requests.get(url, params=params1)
        response2 = requests.get(url, params=params2)
        data1 = response1.json()
        data2 = response2.json()
        contests1 = group_by(data1['result'], lambda x: (x['contestId'], x.get('name', 'N/A')))
        total_contests1 = len(contests1.keys())
        trace3 = go.Bar(
            x=[handle1],
            y=[total_contests1],
            name="Total Contests"
        )
        contests2 = group_by(data2['result'], lambda x: (x['contestId'], x.get('name', 'N/A')))
        total_contests2 = len(contests2.keys())
        trace5 = go.Bar(
            x=[handle2],
            y=[total_contests2],
            name="Total Contests"
        )
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(trace3, row=1, col=1)
        fig.add_trace(trace5, row=1, col=1)
        fig.update_layout(barmode='group')

    return fig

@app.callback(
    dash.dependencies.Output('unsolved-graph', 'figure'),
    [dash.dependencies.Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-1', 'value'),
     dash.dependencies.State('input-2', 'value'),
     dash.dependencies.State('handle-selection', 'value')]
)
def update_rating_graph(n_clicks, handle1, handle2, handle_type):
    url = 'https://codeforces.com/api/user.status'
    if handle_type == 'one':
        params1 = {'handle': handle1}
        response1 = requests.get(url, params=params1)
        data1 = response1.json()
        unsolved1 = [r['problem']['name'] for r in data1['result'] if r['verdict'] != 'OK']
        unsolved1_count = Counter(unsolved1)

        x1 = list(unsolved1_count.keys())
        y1 = [unsolved1_count[k] for k in x1]

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Bar(x=[handle1], y=[sum(y1)], name=handle1), row=1, col=1)
        fig.update_layout(title_text='Unsolved Questions', xaxis_title='Player Handle', yaxis_title='Number of Unsolved Questions')
    else:
        params1 = {'handle': handle1}
        params2 = {'handle': handle2}
        response1 = requests.get(url, params=params1)
        response2 = requests.get(url, params=params2)
        data1 = response1.json()
        data2 = response2.json()
        unsolved1 = [r['problem']['name'] for r in data1['result'] if r['verdict'] != 'OK']
        unsolved2 = [r['problem']['name'] for r in data2['result'] if r['verdict'] != 'OK']
        unsolved1_count = Counter(unsolved1)
        unsolved2_count = Counter(unsolved2)

        x1 = list(unsolved1_count.keys())
        y1 = [unsolved1_count[k] for k in x1]

        x2 = list(unsolved2_count.keys())
        y2 = [unsolved2_count[k] for k in x2]

        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Bar(x=[handle1], y=[sum(y1)], name=handle1), row=1, col=1)
        fig.add_trace(go.Bar(x=[handle2], y=[sum(y2)], name=handle2), row=1, col=1)
        fig.update_layout(title_text='Unsolved Questions', xaxis_title='Player Handle', yaxis_title='Number of Unsolved Questions')

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)