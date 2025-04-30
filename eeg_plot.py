import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import tkinter as tk
from tkinter import ttk
import matplotlib

matplotlib.use('TkAgg')


class EEGVisualizer:
    def __init__(self):
        self.df = None
        self.current_subject = None
        self.current_video = None
        self.anim = None
        self.window_size = 10  # Default window size in seconds
        self.line_theta = None
        self.line_gamma = None
        self.line_alpha = None
        self.line_confusion = None
        self.current_frame = 0
        self.is_playing = False
        self.filtered_df = None
        self.animation_speed = 50  # Default speed (ms)
        self.scroll_active = True  # Enable scrolling by default

        # Display options
        self.show_theta = True
        self.show_gamma = True
        self.show_alpha = True

    def load_data(self):
        """Load and validate the EEG data from CSV file"""
        try:
            print("Reading data...")
            self.df = pd.read_csv('test_results.csv')

            # Ensure required columns exist
            required_columns = ['SubjectID', 'VideoID', 'Timestamp_sec', 'Theta', 'Gamma1', 'Alpha1', 'Predicted_Prob',
                                'userdefined']
            for col in required_columns:
                if col not in self.df.columns:
                    print(f"Warning: Missing column '{col}' in data file")
                    # Create the column with default values if it doesn't exist
                    if col in ['Theta', 'Gamma1', 'Alpha1', 'Predicted_Prob']:
                        self.df[col] = 0.0
                    elif col == 'userdefined':
                        self.df[col] = 0
                    else:
                        self.df[col] = 1  # Default subject and video IDs

            # Sort and reset index
            self.df = self.df.sort_values('Timestamp_sec').reset_index(drop=True)

            # Ensure numeric data types
            numeric_columns = ['SubjectID', 'VideoID', 'Timestamp_sec', 'Theta', 'Gamma1', 'Predicted_Prob',
                               'userdefined']
            for col in numeric_columns:
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                except:
                    print(f"Warning: Converting column '{col}' to numeric")
                    self.df[col] = 0.0  # Default to 0 if conversion fails

            # Fill NaN values
            self.df = self.df.fillna(0)

            # Get unique subjects and videos
            self.subjects = sorted(self.df['SubjectID'].unique().tolist())
            self.videos = sorted(self.df['VideoID'].unique().tolist())

            if not self.subjects:
                print("Warning: No subjects found, creating default")
                self.subjects = [1]
                self.df['SubjectID'] = 1

            if not self.videos:
                print("Warning: No videos found, creating default")
                self.videos = [1]
                self.df['VideoID'] = 1

            print(f"Successfully loaded data with {len(self.df)} rows")
            print(f"Found {len(self.subjects)} subjects and {len(self.videos)} videos")

        except FileNotFoundError:
            print("Warning: test_results.csv not found, creating sample data")
            # Create sample data
            timestamps = np.linspace(0, 10, 100)
            self.df = pd.DataFrame({
                'SubjectID': [1] * 100,
                'VideoID': [1] * 100,
                'Timestamp_sec': timestamps,
                'Theta': np.sin(timestamps) * 0.5 + 0.5,
                'Gamma1': np.cos(timestamps) * 0.3 + 0.7,
                'Alpha1': np.sin(timestamps * 2) * 0.4 + 0.6,
                'Predicted_Prob': np.random.rand(100) * 0.3 + 0.3,
                'userdefined': [1 if i % 20 == 0 else 0 for i in range(100)]
            })
            self.subjects = [1]
            self.videos = [1]
        except Exception as e:
            print(f"Error loading data: {str(e)}, creating sample data")
            # Create sample data
            timestamps = np.linspace(0, 10, 100)
            self.df = pd.DataFrame({
                'SubjectID': [1] * 100,
                'VideoID': [1] * 100,
                'Timestamp_sec': timestamps,
                'Theta': np.sin(timestamps) * 0.5 + 0.5,
                'Gamma1': np.cos(timestamps) * 0.3 + 0.7,
                'Alpha1': np.sin(timestamps * 2) * 0.4 + 0.6,
                'Predicted_Prob': np.random.rand(100) * 0.3 + 0.3,
                'userdefined': [1 if i % 20 == 0 else 0 for i in range(100)]
            })
            self.subjects = [1]
            self.videos = [1]

    def setup_gui(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("EEG Visualization")

        # Create dropdown frame
        dropdown_frame = ttk.Frame(self.root)
        dropdown_frame.pack(pady=5)

        # Subject dropdown
        ttk.Label(dropdown_frame, text="Select Subject:").pack(side=tk.LEFT, padx=5)
        self.subject_var = tk.StringVar()
        if self.subjects:
            self.subject_var.set(str(self.subjects[0]))
        subject_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.subject_var)
        subject_dropdown['values'] = [str(s) for s in self.subjects]
        subject_dropdown.pack(side=tk.LEFT, padx=5)

        # Video dropdown
        ttk.Label(dropdown_frame, text="Select Video:").pack(side=tk.LEFT, padx=5)
        self.video_var = tk.StringVar()
        if self.videos:
            self.video_var.set(str(self.videos[0]))
        video_dropdown = ttk.Combobox(dropdown_frame, textvariable=self.video_var)
        video_dropdown['values'] = [str(v) for v in self.videos]
        video_dropdown.pack(side=tk.LEFT, padx=5)

        # Wave selection frame
        wave_frame = ttk.LabelFrame(self.root, text="Wave Display Options")
        wave_frame.pack(pady=5, fill=tk.X, padx=10)

        # Theta checkbox
        self.theta_var = tk.BooleanVar(value=True)
        theta_check = ttk.Checkbutton(wave_frame, text="Theta", variable=self.theta_var,
                                      command=self.update_wave_display)
        theta_check.pack(side=tk.LEFT, padx=20, pady=5)

        # Gamma checkbox
        self.gamma_var = tk.BooleanVar(value=True)
        gamma_check = ttk.Checkbutton(wave_frame, text="Gamma", variable=self.gamma_var,
                                      command=self.update_wave_display)
        gamma_check.pack(side=tk.LEFT, padx=20, pady=5)

        # Alpha checkbox
        self.alpha_var = tk.BooleanVar(value=True)
        alpha_check = ttk.Checkbutton(wave_frame, text="Alpha", variable=self.alpha_var,
                                      command=self.update_wave_display)
        alpha_check.pack(side=tk.LEFT, padx=20, pady=5)

        # Create matplotlib figure
        self.fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 2])

        # EEG waves subplot
        self.ax1 = plt.subplot(gs[0])
        self.ax1.set_title('EEG Waves')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True)

        # Confusion probability subplot
        self.ax2 = plt.subplot(gs[1])
        self.ax2.set_title('Confusion Probability')
        self.ax2.set_ylabel('Probability')
        self.ax2.set_xlabel('Time (seconds)')
        self.ax2.grid(True)

        # Initialize empty lines
        self.line_theta, = self.ax1.plot([], [], 'b-', label='Theta')
        self.line_gamma, = self.ax1.plot([], [], 'r-', label='Gamma')
        self.line_alpha, = self.ax1.plot([], [], 'g-', label='Alpha')
        self.line_confusion, = self.ax2.plot([], [], 'purple', label='Confusion Probability')
        self.ax1.legend()

        # Add canvas to tkinter window
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Control frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5)

        # Play/Pause button
        self.play_button = ttk.Button(control_frame, text="Play", command=self.toggle_animation)
        self.play_button.pack(side=tk.LEFT, padx=5)

        # Reset button
        reset_button = ttk.Button(control_frame, text="Reset", command=self.reset_animation)
        reset_button.pack(side=tk.LEFT, padx=5)

        # Speed control frame
        speed_frame = ttk.Frame(self.root)
        speed_frame.pack(pady=5)

        # Speed label
        ttk.Label(speed_frame, text="Animation Speed:").pack(side=tk.LEFT, padx=5)

        # Speed slider
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_slider = ttk.Scale(
            speed_frame,
            from_=0.1,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.speed_var,
            command=self.update_speed,
            length=200
        )
        speed_slider.pack(side=tk.LEFT, padx=5)

        # Speed display label
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_label.pack(side=tk.LEFT, padx=5)

        # Add scrolling toggle
        self.scroll_var = tk.BooleanVar(value=True)
        scroll_check = ttk.Checkbutton(speed_frame, text="Auto-scroll",
                                       variable=self.scroll_var,
                                       command=self.toggle_scrolling)
        scroll_check.pack(side=tk.LEFT, padx=20)

        # Status bar for messages
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Make sure dropdowns and wave selection work correctly
        subject_dropdown.bind('<<ComboboxSelected>>', lambda e: self.root.after(100, self.update_data))
        video_dropdown.bind('<<ComboboxSelected>>', lambda e: self.root.after(100, self.update_data))

        # Preset button frame
        preset_frame = ttk.Frame(self.root)
        preset_frame.pack(pady=5)

        # Add preset buttons for common wave combinations
        ttk.Label(preset_frame, text="Quick Selections:").pack(side=tk.LEFT, padx=5)

        ttk.Button(preset_frame, text="All Waves",
                   command=lambda: self.set_waves(True, True, True)).pack(side=tk.LEFT, padx=5)

        ttk.Button(preset_frame, text="Theta Only",
                   command=lambda: self.set_waves(True, False, False)).pack(side=tk.LEFT, padx=5)

        ttk.Button(preset_frame, text="Gamma Only",
                   command=lambda: self.set_waves(False, True, False)).pack(side=tk.LEFT, padx=5)

        ttk.Button(preset_frame, text="Alpha Only",
                   command=lambda: self.set_waves(False, False, True)).pack(side=tk.LEFT, padx=5)

        # Initial data loading
        self.root.after(100, self.update_data)

    def update_speed(self, value):
        try:
            speed = float(value)
            # Limit minimum interval to prevent performance issues
            min_interval = 20  # 20ms minimum interval
            max_interval = 500  # 500ms maximum interval

            # Calculate new interval
            base_interval = 50  # Base speed is 50ms
            new_interval = int(base_interval / speed)

            # Apply limits
            new_interval = max(min_interval, min(new_interval, max_interval))

            self.speed_label.config(text=f"{speed:.1f}x")
            self.animation_speed = new_interval

            # Update window size based on speed
            # Faster speeds = larger window to show more data
            self.window_size = 10 / speed  # Adjust window size inverse to speed

            # Only update animation if it exists and is playing
            if self.anim is not None and hasattr(self.anim, 'event_source') and self.is_playing:
                self.anim.event_source.interval = self.animation_speed

            self.status_var.set(f"Speed set to {speed:.1f}x | Window: {self.window_size:.1f}s")
        except Exception as e:
            print(f"Error updating speed: {str(e)}")
            self.status_var.set(f"Error updating speed")

    def init_animation(self):
        # Clear the lines
        self.line_theta.set_data([], [])
        self.line_gamma.set_data([], [])
        self.line_alpha.set_data([], [])
        self.line_confusion.set_data([], [])

        # Clear any existing vertical lines for confusion indicators
        for line in self.ax2.lines[1:]:  # Keep the main confusion probability line
            line.remove()

        return self.line_theta, self.line_gamma, self.line_alpha, self.line_confusion

    def update_wave_display(self):
        """Update which brain waves are displayed based on checkboxes"""
        try:
            self.show_theta = self.theta_var.get()
            self.show_gamma = self.gamma_var.get()
            self.show_alpha = self.alpha_var.get()

            # Update line visibility
            self.line_theta.set_visible(self.show_theta)
            self.line_gamma.set_visible(self.show_gamma)
            self.line_alpha.set_visible(self.show_alpha)

            # Refresh the canvas
            self.canvas.draw()

            # Update status
            waves = []
            if self.show_theta: waves.append("Theta")
            if self.show_gamma: waves.append("Gamma")
            if self.show_alpha: waves.append("Alpha")

            if waves:
                self.status_var.set(f"Showing: {', '.join(waves)}")
            else:
                self.status_var.set("No brain waves selected")
        except Exception as e:
            print(f"Error updating wave display: {str(e)}")
            self.status_var.set("Error updating display")

    def toggle_scrolling(self):
        """Toggle auto-scrolling on/off"""
        try:
            self.scroll_active = self.scroll_var.get()

            if self.scroll_active:
                self.status_var.set("Auto-scrolling enabled")
            else:
                self.status_var.set("Auto-scrolling disabled - fixed window")

            # If animation is playing, update the current view based on scrolling state
            if self.is_playing and self.filtered_df is not None and not self.filtered_df.empty:
                min_time = self.filtered_df['Timestamp_sec'].min()
                max_time = self.filtered_df['Timestamp_sec'].max()

                if not self.scroll_active:
                    # When scrolling is disabled, show the full data range
                    self.ax1.set_xlim(min_time, max_time)
                    self.ax2.set_xlim(min_time, max_time)
                else:
                    # When scrolling is enabled, show current window
                    current_time = self.current_frame
                    window_start = max(current_time - self.window_size, min_time)
                    self.ax1.set_xlim(window_start, current_time)
                    self.ax2.set_xlim(window_start, current_time)

                self.canvas.draw()

        except Exception as e:
            print(f"Error toggling scrolling: {str(e)}")

    def animate(self, frame):
        try:
            if self.filtered_df is None or self.filtered_df.empty:
                return self.line_theta, self.line_gamma, self.line_alpha, self.line_confusion

            # Store current frame for other methods to access
            self.current_frame = frame

            # For scrolling display, we need all data up to the current frame
            current_data = self.filtered_df[self.filtered_df['Timestamp_sec'] <= frame]

            if not current_data.empty:
                # Get all timestamps and data for the full visible history
                times = current_data['Timestamp_sec'].values
                theta_values = current_data['Theta'].values
                gamma_values = current_data['Gamma1'].values
                alpha_values = current_data['Alpha1'].values
                confusion_values = current_data['Predicted_Prob'].values

                # Update the line data
                self.line_theta.set_data(times, theta_values)
                self.line_gamma.set_data(times, gamma_values)
                self.line_alpha.set_data(times, alpha_values)
                self.line_confusion.set_data(times, confusion_values)

                # Apply visibility settings
                self.line_theta.set_visible(self.show_theta)
                self.line_gamma.set_visible(self.show_gamma)
                self.line_alpha.set_visible(self.show_alpha)

                # Calculate window limits
                min_time = self.filtered_df['Timestamp_sec'].min()
                max_time = self.filtered_df['Timestamp_sec'].max()

                if self.scroll_active:
                    # Scrolling window - show fixed time window that moves with current time
                    window_start = max(frame - self.window_size, min_time)
                    window_end = frame

                    # If we're at the beginning, ensure we still show a full window
                    if window_end - window_start < self.window_size:
                        window_end = min(window_start + self.window_size, max_time)
                else:
                    # Fixed window - show all data
                    window_start = min_time
                    window_end = max_time

                # Update axes limits
                self.ax1.set_xlim(window_start, window_end)
                self.ax2.set_xlim(window_start, window_end)

                # Clear previous vertical lines (confusion indicators and time markers)
                for ax in [self.ax1, self.ax2]:
                    lines_to_remove = []
                    for line in ax.lines:
                        if line not in [self.line_theta, self.line_gamma, self.line_alpha, self.line_confusion]:
                            lines_to_remove.append(line)
                    for line in lines_to_remove:
                        line.remove()

                # Add confusion indicators within the visible window
                confusion_points = current_data[current_data['userdefined'] == 1]
                if not confusion_points.empty:
                    for t in confusion_points['Timestamp_sec']:
                        if window_start <= t <= window_end:
                            self.ax2.axvline(x=t, color='red', alpha=0.3, linewidth=2)

                # Add current time marker
                self.ax1.axvline(x=frame, color='black', alpha=0.5, linewidth=1, linestyle='--')
                self.ax2.axvline(x=frame, color='black', alpha=0.5, linewidth=1, linestyle='--')

                # Update y-axis limits if at the beginning
                if frame == min_time or self.ax1.get_ylim()[1] == 1:  # Initial setup
                    visible_data = self.filtered_df[(self.filtered_df['Timestamp_sec'] >= window_start) &
                                                    (self.filtered_df['Timestamp_sec'] <= window_end)]

                    if not visible_data.empty:
                        max_theta = visible_data['Theta'].max() if self.show_theta else 0
                        max_gamma = visible_data['Gamma1'].max() if self.show_gamma else 0
                        max_alpha = visible_data['Alpha1'].max() if self.show_alpha else 0
                        max_value = max(max_theta, max_gamma, max_alpha, 0.1)  # Use at least 0.1

                        self.ax1.set_ylim(0, max_value * 1.1)
                        self.ax2.set_ylim(0, 1.1)

                # Update status bar
                waves_active = []
                if self.show_theta: waves_active.append("Theta")
                if self.show_gamma: waves_active.append("Gamma")
                if self.show_alpha: waves_active.append("Alpha")
                waves_str = ", ".join(waves_active) if waves_active else "None"

                scroll_status = "Scrolling" if self.scroll_active else "Full view"
                self.status_var.set(f"Time: {frame:.2f}s | {scroll_status} | Showing: {waves_str}")

        except Exception as e:
            print(f"Animation error: {str(e)}")

        return self.line_theta, self.line_gamma, self.line_alpha, self.line_confusion

    def update_data(self):
        try:
            # Stop any existing animation
            if self.anim is not None:
                try:
                    if hasattr(self.anim, 'event_source'):
                        self.anim.event_source.stop()
                except:
                    pass
                self.anim = None
            self.is_playing = False
            self.play_button.configure(text="Play")

            # Get current selections
            try:
                subject = float(self.subject_var.get())
                video = float(self.video_var.get())
            except:
                # If there's any issue, use the first available subject and video
                if self.subjects and self.videos:
                    subject = self.subjects[0]
                    video = self.videos[0]
                    self.subject_var.set(str(subject))
                    self.video_var.set(str(video))
                else:
                    print("No valid subjects or videos")
                    self.status_var.set("No valid data available")
                    return

            # Get the filtered data
            self.filtered_df = self.df[(self.df['SubjectID'] == subject) &
                                       (self.df['VideoID'] == video)].copy()

            if self.filtered_df.empty:
                print(f"No data found for subject {subject} and video {video}")
                self.status_var.set(f"No data for Subject {subject}, Video {video}")

                # Create dummy data for this combination
                timestamps = np.linspace(0, 10, 100)
                self.filtered_df = pd.DataFrame({
                    'SubjectID': [subject] * 100,
                    'VideoID': [video] * 100,
                    'Timestamp_sec': timestamps,
                    'Theta': np.sin(timestamps) * 0.5 + 0.5,
                    'Gamma1': np.cos(timestamps) * 0.3 + 0.7,
                    'Alpha1': np.sin(timestamps * 2) * 0.4 + 0.6,
                    'Predicted_Prob': np.random.rand(100) * 0.3 + 0.3,
                    'userdefined': [1 if i % 20 == 0 else 0 for i in range(100)]
                })

            # Reset the animation
            self.current_frame = self.filtered_df['Timestamp_sec'].min()
            self.reset_animation()
            self.status_var.set(f"Data loaded: Subject {subject}, Video {video}")
        except Exception as e:
            print(f"Error updating data: {str(e)}")
            self.status_var.set("Error updating data")

    def toggle_animation(self):
        try:
            # If no data is available or no animation exists, reset first
            if self.filtered_df is None or self.filtered_df.empty or self.anim is None:
                self.reset_animation()
                if self.anim is None:  # If reset didn't create an animation, exit
                    self.status_var.set("Cannot play: No data available")
                    return

            # Toggle play/pause
            if self.is_playing:
                try:
                    if hasattr(self.anim, 'event_source'):
                        self.anim.event_source.stop()
                    self.play_button.configure(text="Play")
                    self.status_var.set("Paused")
                except:
                    pass
            else:
                try:
                    if hasattr(self.anim, 'event_source'):
                        self.anim.event_source.start()
                    self.play_button.configure(text="Pause")
                    self.status_var.set("Playing")
                except:
                    pass

            self.is_playing = not self.is_playing
        except Exception as e:
            print(f"Error toggling animation: {str(e)}")
            self.status_var.set("Error controlling playback")

    def set_waves(self, theta, gamma, alpha):
        """Set wave visibility based on preset selections"""
        try:
            # Update checkboxes
            self.theta_var.set(theta)
            self.gamma_var.set(gamma)
            self.alpha_var.set(alpha)

            # Update internal state and display
            self.update_wave_display()
        except Exception as e:
            print(f"Error setting wave preset: {str(e)}")

    def reset_animation(self):
        try:
            # Stop any existing animation
            if self.anim is not None:
                try:
                    if hasattr(self.anim, 'event_source'):
                        self.anim.event_source.stop()
                except:
                    pass
                self.anim = None

            self.is_playing = False
            self.play_button.configure(text="Play")

            if self.filtered_df is None or self.filtered_df.empty:
                self.status_var.set("Cannot reset: No data available")
                return

            # Clear the plots
            plt.figure(self.fig.number)  # Make sure we're operating on the right figure
            self.init_animation()

            # Set initial axis limits
            try:
                min_time = self.filtered_df['Timestamp_sec'].min()
                max_time = self.filtered_df['Timestamp_sec'].max()

                # Set initial x-axis limits based on scrolling preference
                if self.scroll_active:
                    # Start with a window at the beginning
                    self.ax1.set_xlim(min_time, min_time + self.window_size)
                    self.ax2.set_xlim(min_time, min_time + self.window_size)
                else:
                    # Show all data
                    self.ax1.set_xlim(min_time, max_time)
                    self.ax2.set_xlim(min_time, max_time)

                # Set y-axis limits
                max_theta = self.filtered_df['Theta'].max() if self.show_theta else 0
                max_gamma = self.filtered_df['Gamma1'].max() if self.show_gamma else 0
                max_alpha = self.filtered_df['Alpha1'].max() if self.show_alpha else 0
                self.ax1.set_ylim(0, max(max_theta, max_gamma, max_alpha, 0.1) * 1.1)
                self.ax2.set_ylim(0, 1.1)

                # Reset frame counter
                self.current_frame = min_time
            except Exception as e:
                print(f"Error setting axis limits: {str(e)}")

            # Create new animation with proper error handling
            try:
                frames = sorted(self.filtered_df['Timestamp_sec'].unique())
                if not frames:  # If no unique timestamps, create dummy frames
                    frames = np.linspace(0, 10, 100)

                # Create animation with faster update rate for smoother scrolling
                self.anim = FuncAnimation(
                    self.fig,
                    self.animate,
                    frames=frames,
                    init_func=self.init_animation,
                    interval=self.animation_speed,
                    blit=True,
                    repeat=False,
                    cache_frame_data=False  # Don't cache frame data for better memory performance
                )

                self.status_var.set("Animation reset")
            except Exception as e:
                print(f"Error creating animation: {str(e)}")
                self.status_var.set("Error creating animation")

            # Make sure the canvas updates
            try:
                self.canvas.draw()
            except:
                pass
        except Exception as e:
            print(f"Error resetting animation: {str(e)}")
            self.status_var.set("Reset failed")

            # Set initial axis limits
            try:
                max_theta = self.filtered_df['Theta'].max() if self.show_theta else 0
                max_gamma = self.filtered_df['Gamma1'].max() if self.show_gamma else 0
                max_alpha = self.filtered_df['Alpha1'].max() if self.show_alpha else 0
                self.ax1.set_ylim(0, max(max_theta, max_gamma, max_alpha) * 1.1)
                self.ax2.set_ylim(0, 1.1)

                min_time = self.filtered_df['Timestamp_sec'].min()
                max_time = self.filtered_df['Timestamp_sec'].max()
                self.ax1.set_xlim(min_time, min_time + self.window_size)
                self.ax2.set_xlim(min_time, min_time + self.window_size)

                # Reset frame counter
                self.current_frame = min_time
            except Exception as e:
                print(f"Error setting axis limits: {str(e)}")

            # Create new animation with proper error handling
            try:
                frames = sorted(self.filtered_df['Timestamp_sec'].unique())
                if not frames:  # If no unique timestamps, create dummy frames
                    frames = np.linspace(0, 10, 100)

                self.anim = FuncAnimation(
                    self.fig,
                    self.animate,
                    frames=frames,
                    init_func=self.init_animation,
                    interval=self.animation_speed,
                    blit=True,
                    repeat=False
                )

                self.status_var.set("Animation reset")
            except Exception as e:
                print(f"Error creating animation: {str(e)}")
                self.status_var.set("Error creating animation")

            # Make sure the canvas updates
            try:
                self.canvas.draw()
            except:
                pass
        except Exception as e:
            print(f"Error resetting animation: {str(e)}")
            self.status_var.set("Reset failed")

    def run(self):
        try:
            self.load_data()
            self.setup_gui()
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.root.mainloop()
        except Exception as e:
            print(f"Error running application: {str(e)}")
            if hasattr(self, 'root') and self.root is not None:
                self.root.destroy()

    def on_closing(self):
        """Handle window closing event"""
        try:
            if self.anim is not None and hasattr(self.anim, 'event_source'):
                self.anim.event_source.stop()
        except:
            pass

        try:
            plt.close('all')  # Close all matplotlib figures
        except:
            pass

        if hasattr(self, 'root') and self.root is not None:
            self.root.destroy()


if __name__ == "__main__":
    try:
        visualizer = EEGVisualizer()
        visualizer.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")