import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import pandas as pd
import os
import matplotlib

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

# Example usage
if __name__ == "__main__":
    csv_path = "Results/test_results.csv"
    data = pd.read_csv(csv_path)
    create_animation(data)