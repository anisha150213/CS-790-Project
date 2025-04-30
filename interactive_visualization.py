import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Read the data and sort by timestamp
df = pd.read_csv('test_results.csv')

# Get unique students and videos
students = sorted(df['SubjectID'].unique())
videos = sorted(df['VideoID'].unique())

def filter_data(df, student=None, video=None):
    filtered_df = df.copy()
    if student is not None:
        filtered_df = filtered_df[filtered_df['SubjectID'] == student]
    if video is not None:
        filtered_df = filtered_df[filtered_df['VideoID'] == video]
    # Sort by timestamp to ensure proper time series display
    filtered_df = filtered_df.sort_values('Timestamp_sec')
    return filtered_df

def create_interactive_plot():
    # Create figure
    fig = go.Figure()

    # Initial data (first student and video)
    initial_df = filter_data(df, student=students[0], video=videos[0])

    # Create frames for animation
    frames = []
    max_time = initial_df['Timestamp_sec'].max()
    time_steps = np.linspace(0, max_time, 100)  # 100 frames for smooth animation
    
    for t in time_steps:
        frame_data = initial_df[initial_df['Timestamp_sec'] <= t]
        frame = go.Frame(
            data=[
                go.Scatter(
                    x=frame_data['Timestamp_sec'],
                    y=frame_data['Theta'],
                    name="Theta",
                    line=dict(color='blue', width=2),
                    mode='lines'
                ),
                go.Scatter(
                    x=frame_data['Timestamp_sec'],
                    y=frame_data['Gamma1'],
                    name="Gamma",
                    line=dict(color='green', width=2),
                    mode='lines'
                ),
                go.Scatter(
                    x=frame_data['Timestamp_sec'],
                    y=frame_data['Alpha1'],
                    name="Alpha",
                    line=dict(color='purple', width=2),
                    mode='lines'
                )
            ],
            name=str(t)
        )
        frames.append(frame)

    # Add initial traces
    wave_colors = {'Theta': 'blue', 'Gamma': 'green', 'Alpha': 'purple'}
    for wave, color in wave_colors.items():
        column = 'Theta' if wave == 'Theta' else ('Gamma1' if wave == 'Gamma' else 'Alpha1')
        fig.add_trace(
            go.Scatter(
                x=[initial_df['Timestamp_sec'].iloc[0]],
                y=[initial_df[column].iloc[0]],
                name=wave,
                line=dict(color=color, width=2),
                mode='lines'
            )
        )

    # Create dropdown menus
    updatemenus = [
        # Play button menu
        dict(
            type="buttons",
            showactive=False,
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 50, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                )
            ],
            direction="left",
            pad={"r": 10, "t": 10},
            x=0.1,
            y=0,
            xanchor="right",
            yanchor="top"
        ),
        # Student selection dropdown
        dict(
            buttons=[dict(
                args=[{
                    "visible": [True, True, True],
                    "x": [filter_data(df, student=student, video=videos[0])['Timestamp_sec']]*3,
                    "y": [
                        filter_data(df, student=student, video=videos[0])['Theta'],
                        filter_data(df, student=student, video=videos[0])['Gamma1'],
                        filter_data(df, student=student, video=videos[0])['Alpha1']
                    ]
                }],
                label=f"Student {student}",
                method="restyle"
            ) for student in students],
            direction="down",
            showactive=True,
            x=0.1,
            y=1.1,
            xanchor="left",
            yanchor="top",
            name="Student"
        ),
        # Video selection dropdown
        dict(
            buttons=[dict(
                args=[{
                    "visible": [True, True, True],
                    "x": [filter_data(df, student=students[0], video=video)['Timestamp_sec']]*3,
                    "y": [
                        filter_data(df, student=students[0], video=video)['Theta'],
                        filter_data(df, student=students[0], video=video)['Gamma1'],
                        filter_data(df, student=students[0], video=video)['Alpha1']
                    ]
                }],
                label=f"Video {video}",
                method="restyle"
            ) for video in videos],
            direction="down",
            showactive=True,
            x=0.3,
            y=1.1,
            xanchor="left",
            yanchor="top",
            name="Video"
        ),
        # Wave selection dropdown
        dict(
            buttons=[
                dict(
                    args=[{"visible": [True, True, True]}],
                    label="All Waves",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [True, False, False]}],
                    label="Theta Only",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [False, True, False]}],
                    label="Gamma Only",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [False, False, True]}],
                    label="Alpha Only",
                    method="restyle"
                )
            ],
            direction="down",
            showactive=True,
            x=0.5,
            y=1.1,
            xanchor="left",
            yanchor="top",
            name="Waves"
        )
    ]

    # Update layout
    fig.update_layout(
        title="Brain Wave Analysis Over Time",
        xaxis_title="Video Time (seconds)",
        yaxis_title="Wave Amplitude",
        showlegend=True,
        height=700,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=100),
        updatemenus=updatemenus,
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.05
            )
        ),
        yaxis=dict(
            range=[0, max(initial_df['Theta'].max(),
                         initial_df['Gamma1'].max(),
                         initial_df['Alpha1'].max()) * 1.1]
        )
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Add frames to the figure
    fig.frames = frames

    return fig

# Create and save the plot
fig = create_interactive_plot()
fig.write_html("brain_waves_visualization.html")
print("Visualization has been saved to 'brain_waves_visualization.html'")
print("Please open this file in your web browser to view the interactive plot.") 