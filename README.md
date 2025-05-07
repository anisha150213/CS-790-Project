# EEG Analysis and Visualization Project

This project focuses on analyzing and visualizing EEG (Electroencephalogram) data, particularly focusing on attention, mediation, and confusion detection during video watching sessions.

## Project Overview

The project includes several key components:
- EEG data analysis and processing
- Interactive visualizations of brain wave patterns
- Confusion detection and analysis
- Web-based dashboard for data exploration
- Statistical analysis and model performance evaluation

## Features

- Real-time EEG wave visualization
- Attention and mediation tracking
- Confusion event detection and analysis
- Interactive web interface for data exploration
- Statistical analysis and correlation studies
- Model performance evaluation and comparison

## Prerequisites

- Python 3.8 or higher
- FFmpeg (for video generation)
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd CS-790-Project
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Install FFmpeg:
- Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)
- Linux: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

## Project Structure

```
├── app.py                    # Flask web application
├── analysis_visuals.py       # Data analysis and visualization functions
├── interactive_visualization.py  # Interactive plotting utilities
├── eeg_plot.py              # EEG data plotting functions
├── confusion_wave_visualization.py  # Confusion analysis visualization
├── templates/               # Web application templates
├── output/                  # Generated visualizations and analysis results
├── Results/                 # Analysis results and processed data
├── Dataset/                 # Raw EEG data
└── saved_models/           # Trained machine learning models
```

## Usage

1. Start the web application:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`

3. For data analysis and visualization:
```bash
python analysis_visuals.py
```

## Data Analysis

The project includes several analysis components:
- Time series analysis of EEG waves
- Distribution analysis of brain wave patterns
- Correlation studies between different metrics
- Confusion event analysis
- Model performance evaluation

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

Copyright (c) 2024 CS-790-Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

