import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

# Read the data and ensure it's sorted by time
df = pd.read_csv('test_results.csv')
df = df.sort_values('Timestamp_sec').reset_index(drop=True)

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = html.Div([
    dbc.Container([
        html.H1("EEG Waves and Confusion Visualization", className="text-center my-4"),
        
        # Controls
        dbc.Row([
            dbc.Col([
                html.Label("Update Interval (ms)"),
                dcc.Slider(
                    id='speed-slider',
                    min=100,
                    max=2000,
                    value=500,
                    marks={i: str(i) for i in range(100, 2001, 300)}
                )
            ], width=6),
            dbc.Col([
                html.Label("Data Points to Show"),
                dcc.Slider(
                    id='window-slider',
                    min=10,
                    max=100,
                    value=30,
                    marks={i: str(i) for i in [10, 20, 30, 50, 75, 100]}
                )
            ], width=6)
        ], className="mb-4"),
        
        # Buttons
        dbc.Row([
            dbc.Col([
                dbc.Button("Start/Stop", id="start-stop-button", color="primary", className="me-2"),
                dbc.Button("Reset", id="reset-button", color="secondary")
            ], width=12, className="text-center mb-4")
        ]),
        
        # Graph
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='live-graph'),
                dcc.Interval(
                    id='interval-component',
                    interval=500,
                    n_intervals=0,
                    disabled=True
                ),
                dcc.Store(id='current-index', data=0),
                dcc.Store(id='animation-state', data={'is_playing': False})
            ])
        ])
    ])
])

def create_figure(start_idx, num_points):
    # Get the data window
    end_idx = min(start_idx + num_points, len(df))
    window_data = df.iloc[start_idx:end_idx]
    
    # Create figure
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=('EEG Waves', 'Confusion Probability'),
                       vertical_spacing=0.15)
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=window_data['Timestamp_sec'], y=window_data['Theta'],
                  name='Theta', line=dict(color='blue', width=2),
                  mode='lines+markers'), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=window_data['Timestamp_sec'], y=window_data['Gamma1'],
                  name='Gamma', line=dict(color='red', width=2),
                  mode='lines+markers'), row=1, col=1)
    
    fig.add_trace(
        go.Scatter(x=window_data['Timestamp_sec'], y=window_data['Predicted_Prob'],
                  name='Confusion Probability', line=dict(color='green', width=2),
                  mode='lines+markers'), row=2, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="EEG Waves and Confusion Over Time",
        plot_bgcolor='rgba(240, 240, 240, 0.8)',
        paper_bgcolor='white',
        margin=dict(t=100),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Update axes
    fig.update_yaxes(title_text="Amplitude", row=1, col=1, gridcolor='white')
    fig.update_yaxes(title_text="Probability", row=2, col=1, gridcolor='white',
                     range=[0, 1])
    fig.update_xaxes(title_text="Time (seconds)", gridcolor='white')
    
    return fig

@app.callback(
    [Output('live-graph', 'figure'),
     Output('current-index', 'data'),
     Output('interval-component', 'disabled'),
     Output('interval-component', 'interval'),
     Output('animation-state', 'data')],
    [Input('interval-component', 'n_intervals'),
     Input('start-stop-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('speed-slider', 'value'),
     Input('window-slider', 'value')],
    [State('current-index', 'data'),
     State('animation-state', 'data')]
)
def update_graph(n_intervals, start_stop_clicks, reset_clicks, speed, num_points, 
                current_idx, animation_state):
    ctx = dash.callback_context
    if not ctx.triggered:
        return create_figure(0, num_points), 0, True, speed, {'is_playing': False}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle button clicks
    if button_id == 'reset-button':
        return create_figure(0, num_points), 0, True, speed, {'is_playing': False}
    elif button_id == 'start-stop-button':
        new_state = not animation_state['is_playing']
        return dash.no_update, current_idx, not new_state, speed, {'is_playing': new_state}
    
    # Update index based on the interval
    if animation_state['is_playing']:
        current_idx = current_idx + 1
        if current_idx >= len(df) - num_points:
            current_idx = 0
    
    # Create and return the figure
    fig = create_figure(current_idx, num_points)
    return fig, current_idx, dash.no_update, speed, animation_state

if __name__ == '__main__':
    app.run(debug=True, port=8050) 