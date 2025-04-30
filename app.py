from flask import Flask, render_template, jsonify
import pandas as pd
import json

app = Flask(__name__)

# Load the data
df = pd.read_csv('Results/test_results.csv')

@app.route('/')
def index():
    # Get unique students and videos for dropdowns
    students = sorted(df['SubjectID'].unique())
    videos = sorted(df['VideoID'].unique())
    return render_template('index.html', students=students, videos=videos)

@app.route('/get_data/<student_id>/<video_id>')
def get_data(student_id, video_id):
    # Filter data for selected student and video
    filtered_data = df[(df['SubjectID'] == float(student_id)) & (df['VideoID'] == float(video_id))]
    
    # Convert to list of dictionaries for JSON serialization
    data = filtered_data.to_dict('records')
    
    # Convert numpy types to Python native types
    for record in data:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
            elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                record[key] = str(value)
            else:
                record[key] = float(value) if isinstance(value, (int, float)) else value
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True) 