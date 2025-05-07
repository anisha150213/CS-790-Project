import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
import os
import matplotlib
import numpy as np
import json
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve, auc

# Set the path to the ffmpeg executable
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:/Program Files/ffmpeg/bin/ffmpeg.exe'

# Define the output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

def create_animation(data):
    # Create figure and axis with larger size
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set limits with some padding
    ax.set_xlim(data['Timestamp_sec'].min() - 5, data['Timestamp_sec'].max() + 5)
    ax.set_ylim(0, 100)  # Attention and Mediation are typically 0-100
    
    # Create lines for both Attention and Mediation
    attention_line, = ax.plot([], [], lw=2, label="Attention", color='blue')
    mediation_line, = ax.plot([], [], lw=2, label="Mediation", color='green')
    
    # Add time indicator
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    
    # Customize the plot
    ax.set_title('EEG Attention and Mediation Over Time', fontsize=14, pad=20)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')
    
    def init():
        attention_line.set_data([], [])
        mediation_line.set_data([], [])
        time_text.set_text('')
        return attention_line, mediation_line, time_text

    def update(frame):
        x = data['Timestamp_sec'][:frame]
        attention_y = data['Attention'][:frame]
        mediation_y = data['Mediation'][:frame]
        
        attention_line.set_data(x, attention_y)
        mediation_line.set_data(x, mediation_y)
        
        # Update time indicator
        current_time = data['Timestamp_sec'][frame-1] if frame > 0 else 0
        time_text.set_text(f'Time: {current_time:.1f}s')
        
        return attention_line, mediation_line, time_text

    # Create animation with slower interval for better visibility
    ani = FuncAnimation(fig, update, frames=len(data), init_func=init, 
                       blit=True, interval=100)  # Increased interval to 100ms

    # Save the animation with high quality settings
    writer = FFMpegWriter(fps=10, codec='libx264', 
                         extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    ani.save(f"{output_dir}/attention_mediation_animation.mp4", writer=writer, dpi=300)
    plt.close()

def process_data(data, window_size=50):
    """
    Process the data to create windowed views and normalize EEG values
    """
    # Convert timestamps to relative time (starting from 0)
    start_time = data['Timestamp_sec'].min()
    data['relative_time'] = data['Timestamp_sec'] - start_time
    
    # Create windows of data
    total_points = len(data)
    windows = []
    
    for i in range(0, total_points - window_size + 1):
        window = data.iloc[i:i+window_size]
        window_data = {
            'time': window['relative_time'].tolist(),
            'alpha1': np.log10(window['Alpha1']).tolist(),  # Log scale for better visualization
            'gamma1': np.log10(window['Gamma1']).tolist(),
            'theta': np.log10(window['Theta']).tolist(),
            'confusion_pressed': window['confusion_pressed'].tolist(),
            'predicted_prob': window['predicted_prob'].tolist()
        }
        windows.append(window_data)
    
    return windows

def create_data_files(csv_path, output_dir="static/data"):
    """
    Create JSON files containing processed data for the web interface
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and process data
    data = pd.read_csv(csv_path)
    
    # Get unique combinations of SubjectID and VideoID
    combinations = data[['SubjectID', 'VideoID']].drop_duplicates()
    
    # Process data for each combination
    for _, row in combinations.iterrows():
        subject_id = row['SubjectID']
        video_id = row['VideoID']
        
        # Filter data for this combination
        mask = (data['SubjectID'] == subject_id) & (data['VideoID'] == video_id)
        subset_data = data[mask].copy()
        
        # Process the data into windows
        windows = process_data(subset_data)
        
        # Save to JSON file
        filename = f"data_{subject_id}_{video_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(windows, f)

def create_output_dirs():
    """Create output directories for different types of visualizations"""
    dirs = ['output/time_series', 'output/distributions', 'output/correlations', 
            'output/aggregates', 'output/confusion_analysis']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def plot_time_series(data, output_dir):
    """Create time series plots for EEG waves and confusion indicators"""
    # Plot EEG waves over time
    plt.figure(figsize=(15, 8))
    plt.plot(data['Timestamp_sec'], np.log10(data['Alpha1']), label='Alpha1', alpha=0.7)
    plt.plot(data['Timestamp_sec'], np.log10(data['Gamma1']), label='Gamma1', alpha=0.7)
    plt.plot(data['Timestamp_sec'], np.log10(data['Theta']), label='Theta', alpha=0.7)
    plt.title('EEG Waves Over Time (Log Scale)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Log10(Power)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{output_dir}/eeg_waves_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot confusion indicators
    plt.figure(figsize=(15, 8))
    plt.plot(data['Timestamp_sec'], data['Predicted_Prob_tuned'], label='Predicted Probability (Tuned)', color='blue')
    plt.plot(data['Timestamp_sec'], data['Predicted_Prob_default'], label='Predicted Probability (Default)', color='green', alpha=0.5)
    confusion_times = data[data['userdefined'] == 1]['Timestamp_sec']
    plt.vlines(confusion_times, 0, 1, colors='red', label='User Confusion Button Press', alpha=0.5)
    plt.title('Confusion Indicators Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Probability')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f'{output_dir}/confusion_indicators_time_series.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_distributions(data, output_dir):
    """Create distribution plots for EEG waves and confusion probabilities"""
    # EEG waves distributions
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    sns.histplot(np.log10(data['Alpha1']), kde=True)
    plt.title('Alpha1 Distribution')
    plt.xlabel('Log10(Power)')
    
    plt.subplot(132)
    sns.histplot(np.log10(data['Gamma1']), kde=True)
    plt.title('Gamma1 Distribution')
    plt.xlabel('Log10(Power)')
    
    plt.subplot(133)
    sns.histplot(np.log10(data['Theta']), kde=True)
    plt.title('Theta Distribution')
    plt.xlabel('Log10(Power)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/eeg_waves_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Predicted probability distributions
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    sns.histplot(data['Predicted_Prob_default'], kde=True)
    plt.title('Default Model Predictions')
    plt.xlabel('Probability')
    
    plt.subplot(122)
    sns.histplot(data['Predicted_Prob_tuned'], kde=True)
    plt.title('Tuned Model Predictions')
    plt.xlabel('Probability')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predicted_prob_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_correlations(data, output_dir):
    """Create correlation plots between different metrics"""
    # Select relevant columns
    cols = ['Alpha1', 'Gamma1', 'Theta', 'Predicted_Prob_default', 'Predicted_Prob_tuned', 'userdefined']
    corr_data = data[cols].copy()
    corr_data[['Alpha1', 'Gamma1', 'Theta']] = np.log10(corr_data[['Alpha1', 'Gamma1', 'Theta']])
    
    # Create correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of EEG Waves and Confusion Indicators')
    plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_confusion_events(data, output_dir):
    """Analyze patterns around confusion button presses"""
    window_size = 50  # samples before and after confusion event
    confusion_events = data[data['userdefined'] == 1].index
    
    # Collect EEG patterns around confusion events
    eeg_patterns = {wave: [] for wave in ['Alpha1', 'Gamma1', 'Theta']}
    for event_idx in confusion_events:
        start_idx = max(0, event_idx - window_size)
        end_idx = min(len(data), event_idx + window_size)
        for wave in eeg_patterns.keys():
            pattern = np.log10(data[wave].iloc[start_idx:end_idx].values)
            if len(pattern) == window_size * 2:
                eeg_patterns[wave].append(pattern)
    
    # Plot average EEG patterns around confusion events
    plt.figure(figsize=(12, 6))
    for wave, patterns in eeg_patterns.items():
        if patterns:
            mean_pattern = np.mean(patterns, axis=0)
            plt.plot(range(-window_size, window_size), mean_pattern, label=wave)
    
    plt.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Confusion Event')
    plt.title('Average EEG Patterns Around Confusion Events')
    plt.xlabel('Samples Relative to Confusion Event')
    plt.ylabel('Log10(Power)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/confusion_event_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_by_video(data, output_dir):
    """Create aggregate statistics by video"""
    # Calculate statistics per video
    video_stats = data.groupby('VideoID').agg({
        'Alpha1': ['mean', 'std'],
        'Gamma1': ['mean', 'std'],
        'Theta': ['mean', 'std'],
        'userdefined': 'sum',
        'Predicted_Prob_tuned': 'mean',
        'Predicted_Prob_default': 'mean'
    }).round(3)
    
    # Plot confusion events per video
    plt.figure(figsize=(12, 6))
    video_stats['userdefined']['sum'].plot(kind='bar')
    plt.title('Number of Confusion Events by Video')
    plt.xlabel('Video ID')
    plt.ylabel('Number of Confusion Events')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_events_by_video.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot average prediction probabilities by video
    plt.figure(figsize=(12, 6))
    plt.plot(video_stats.index, video_stats['Predicted_Prob_default']['mean'], 'o-', label='Default Model')
    plt.plot(video_stats.index, video_stats['Predicted_Prob_tuned']['mean'], 'o-', label='Tuned Model')
    plt.title('Average Predicted Confusion Probability by Video')
    plt.xlabel('Video ID')
    plt.ylabel('Average Predicted Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/avg_predictions_by_video.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save statistics to CSV
    video_stats.to_csv(f'{output_dir}/video_statistics.csv')

def analyze_model_performance(data, output_dir):
    """Analyze model performance metrics"""
    # Compare model predictions with actual confusion events
    plt.figure(figsize=(12, 5))
    
    # Default model ROC curve
    plt.subplot(121)
    true_labels = data['True_Label']
    pred_probs_default = data['Predicted_Prob_default']
    fpr, tpr, _ = roc_curve(true_labels, pred_probs_default)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Default Model ROC Curve')
    plt.legend()
    
    # Tuned model ROC curve
    plt.subplot(122)
    pred_probs_tuned = data['Predicted_Prob_tuned']
    fpr, tpr, _ = roc_curve(true_labels, pred_probs_tuned)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Tuned Model ROC Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_performance_roc.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate all visualizations"""
    # Create output directories
    dirs = create_output_dirs()
    
    # Read data
    csv_path = "Results/test_results.csv"
    data = pd.read_csv(csv_path)
    
    # Generate all visualizations
    plot_time_series(data, 'output/time_series')
    plot_distributions(data, 'output/distributions')
    plot_correlations(data, 'output/correlations')
    analyze_confusion_events(data, 'output/confusion_analysis')
    analyze_by_video(data, 'output/aggregates')
    analyze_model_performance(data, 'output/correlations')
    
    print("All visualizations have been generated in the 'output' directory.")

if __name__ == "__main__":
    main()