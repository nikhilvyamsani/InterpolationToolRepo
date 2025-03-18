import gradio as gr
import os
import glob
from scipy.interpolate import CubicSpline, interp1d
import numpy as np
import pandas as pd
import folium
from folium.plugins import MousePosition
import geopy.distance
import cv2
import re
import subprocess
from datetime import datetime, timedelta
import webbrowser
import concurrent.futures

class VideoGPSProcessor:
    @staticmethod
    def interpolate_gps_in_csv(csv_path, start_frame, end_frame, video_name, gps_coords):
        """
        Interpolates GPS points for a specific frame range and updates the CSV.
        """
        try:
            df = pd.read_csv(csv_path)
            required_columns = ['Video_Name', 'Startframe', 'Endframe', 'Latitude', 'Longitude']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain the following columns: {required_columns}")

            video_df = df[df['Video_Name'] == video_name].copy()
            
            if start_frame > end_frame:
                raise ValueError("Start frame must be less than or equal to end frame.")

            max_frame = video_df['Endframe'].max()
            if end_frame > max_frame:
                print(f"Warning: End frame {end_frame} exceeds max frame {max_frame}. Using {max_frame} instead.")
                end_frame = max_frame

            if not isinstance(gps_coords, list) or len(gps_coords) < 4 or len(gps_coords) % 2 != 0:
                raise ValueError("GPS coordinates must be provided as lat1,lon1,lat2,lon2,... (at least 2 points).")

            coords = [(gps_coords[i], gps_coords[i + 1]) for i in range(0, len(gps_coords), 2)]
            frames_to_interpolate = video_df[(video_df['Startframe'] >= start_frame) & (video_df['Endframe'] <= end_frame)]

            if frames_to_interpolate.empty:
                raise ValueError(f"No frames found in range {start_frame}-{end_frame} for video '{video_name}'.")

            frame_indices = frames_to_interpolate.index
            frame_numbers = frames_to_interpolate['Startframe'].to_numpy()

            input_points = np.linspace(0, 1, len(coords))
            latitudes = np.array([coord[0] for coord in coords])
            longitudes = np.array([coord[1] for coord in coords])
            output_points = np.linspace(0, 1, len(frame_numbers))

            try:
                if len(coords) >= 4:
                    lat_spline = CubicSpline(input_points, latitudes)
                    lon_spline = CubicSpline(input_points, longitudes)
                    interp_latitudes = lat_spline(output_points)
                    interp_longitudes = lon_spline(output_points)
                else:
                    lat_interp = interp1d(input_points, latitudes, bounds_error=False, fill_value='extrapolate')
                    lon_interp = interp1d(input_points, longitudes, bounds_error=False, fill_value='extrapolate')
                    interp_latitudes = lat_interp(output_points)
                    interp_longitudes = lon_interp(output_points)
            except Exception as e:
                print(f"Interpolation failed: {e}")
                return df

            df.loc[frame_indices, 'Latitude'] = interp_latitudes
            df.loc[frame_indices, 'Longitude'] = interp_longitudes
            df.loc[frame_indices, 'Position'] = [f"N{lat}E{lon}" for lat, lon in zip(interp_latitudes, interp_longitudes)]

            print(f"Interpolation completed for frames {start_frame}-{end_frame} in video '{video_name}'.")
            return df
        

        except Exception as e:
            print(f"Error in interpolate_gps_in_csv: {str(e)}")
            return df

    @staticmethod
    def decdeg2dms(dd):
        """Convert decimal degrees to degrees, minutes, seconds format."""
        is_positive = dd >= 0
        dd = abs(dd)
        minutes, seconds = divmod(dd * 3600, 60)
        degrees, minutes = divmod(minutes, 60)
        degrees = degrees if is_positive else -degrees
        return f"{str(int(degrees)).zfill(2)}.{str(int(minutes)).zfill(2)}.{seconds:05.2f}"
    
    @staticmethod
    def split(ltng):
        """
        Convert lat/long string to float coordinates
        arguments: combined string format of latlong
        returns: latitude and longitude (float)
        """
        # Handle default or empty case
        if not ltng or ltng == 'N0.00000E0.00000' or pd.isna(ltng):
            return 0.0, 0.0
        
        # Extract latitude
        lat = 0.0
        lon = 0.0
        
        # First character indicates N/S hemisphere
        if ltng.startswith('N'):
            # Northern hemisphere (positive latitude)
            parts = ltng[1:].split('E' if 'E' in ltng else 'W')
            lat = float(parts[0])
        elif ltng.startswith('S'):
            # Southern hemisphere (negative latitude)
            parts = ltng[1:].split('E' if 'E' in ltng else 'W')
            lat = -float(parts[0])
        
        # Check for E/W in the string to determine longitude
        if 'E' in ltng:
            # Eastern hemisphere (positive longitude)
            lon = float(ltng.split('E')[1])
        elif 'W' in ltng:
            # Western hemisphere (negative longitude)
            lon = -float(ltng.split('W')[1])
        
        return lat, lon

    @staticmethod
    def merge(lat, lon):
        """
        Convert latitude and longitude to string format
        arguments: latitude and longitude (float)
        returns: combined string format of latlong
        """
        # Handle invalid coordinates
        if pd.isna(lat) or pd.isna(lon) or (lat == 0.0 and lon == 0.0):
            return None
            
        # Round to 6 decimal places for precision
        lat = round(lat, 6)
        lon = round(lon, 6)
        
        # Create the formatted string
        result = ''
        
        # Add latitude with N/S hemisphere indicator
        if lat >= 0:
            result += 'N' + str(abs(lat))
        else:
            result += 'S' + str(abs(lat))
        
        # Add longitude with E/W hemisphere indicator
        if lon >= 0:
            result += 'E' + str(abs(lon))
        else:
            result += 'W' + str(abs(lon))
        
        return result

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate the Haversine distance between two lat/lon points."""
        # Check for invalid coordinates
        if pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2):
            return 0.0
            
        coords_1 = (lat1, lon1)
        coords_2 = (lat2, lon2)
        return geopy.distance.geodesic(coords_1, coords_2).meters

    @staticmethod
    def is_valid_gps(gps_str):
        """Check if GPS string is valid"""
        if pd.isna(gps_str) or gps_str is None:
            return False
        # Check for invalid or empty GPS format
        if ('N0.00000' in gps_str or 'E0.00000' in gps_str or 
            'S0.00000' in gps_str or 'W0.00000' in gps_str or
            gps_str == 'N0.00000E0.00000'):
            return False
        return True

    def get_gps(self, video_path):
        """
        Extract GPS data from video metadata
        
        Args:
            video_path (str): Path to the MP4 video file
        
        Returns:
            tuple: DataFrame with GPS data, Success flag
        """
        try:
            # Use ExifTool to extract metadata
            tool_path = 'exiftool'  
            command = f'{tool_path} -ee -G3 "{video_path}"'
            result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, text=True)
            meta = result.stdout.split('\n')

            visited = [meta[0][:8], '']
            val = {'time': [], 'gps': [], 'speed': [], 'diff': []}

            try:
                vname = os.path.basename(video_path).split(".")[0]
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
            except Exception as ex:
                print(f"Video error: {ex}")
                return 'Video Error', False

            noerror_id = 0
            for x in range(len(meta)):
                if meta[x][:8] not in visited:
                    visited.append(meta[x][:8])
                    try:
                        c_t = datetime.strptime(meta[x][-20:-1], "%Y:%m:%d %H:%M:%S")
                    except Exception as ex:
                        print(f"Time parsing error: {ex}")
                        c_t = datetime.now()
                        noerror_id += 1
                        if len(val['time']):
                            c_t = val['time'][-1]

                    if len(val['time']):
                        val['diff'].append(int((c_t - val['time'][-1]).total_seconds()))

                    val['time'].append(c_t)
                    current = ''
                    for ii in range(1, 3):
                        if x + ii < len(meta) and ": " in meta[x + ii]:
                            xx = meta[x + ii].split(": ")[1]
                            tem = re.split('[( deg )(\')"]', xx)
                            try:
                                c = str(round(float(tem[0]) + float(tem[5]) / 60 + float(tem[7]) / 3600, 6))
                                c += '0' * (9 - len(c))
                                # Add the hemisphere indicator (N/S for latitude, E/W for longitude)
                                current += xx[-1] + c
                            except (IndexError, ValueError) as e:
                                print(f"GPS parsing error: {e}")
                                current = 'N0.00000E0.00000'
                        else:
                            current = 'N0.00000E0.00000'

                    val['gps'].append(current)
                    
                    try:
                        if x + 3 < len(meta) and ": " in meta[x + 3]:
                            val['speed'].append(int(float(meta[x + 3].split(': ')[1])))
                        else:
                            val['speed'].append(0)
                    except (IndexError, ValueError):
                        val['speed'].append(0)
            
            val['diff'].append(0)
            df = pd.DataFrame(val)

            # Clean and prepare data for interpolation
            final_df = {'Frame': [], 'Position': [], 'Speed': []}

            required_sec = int(np.ceil(total_frames / fps))
            sec_available = len(df)
            print(f'Required seconds: {required_sec}, Seconds available: {sec_available}')
            
            # Mark invalid GPS values as NaN for proper interpolation
            df['valid_gps'] = df['gps'].apply(self.is_valid_gps)
            df.loc[~df['valid_gps'], 'gps'] = np.nan
            
            # Create frame indices
            frames = np.arange(0, total_frames, 2)
            frame_seconds = frames / fps
            
            # Extract lat/lon for interpolation
            df['latitude'] = np.nan
            df['longitude'] = np.nan
            
            # Only process valid GPS points
            valid_indices = df['valid_gps']
            if valid_indices.sum() >= 2:  # We need at least 2 valid points for interpolation
                for idx in df.index[valid_indices]:
                    lat, lon = self.split(df.loc[idx, 'gps'])
                    df.loc[idx, 'latitude'] = lat
                    df.loc[idx, 'longitude'] = lon
                
                # Create time indices for valid points (seconds from start)
                valid_df = df[valid_indices].copy()
                time_indices = np.arange(len(valid_df))
                
                # Perform cubic spline interpolation if we have enough points
                if len(time_indices) >= 4:  # Cubic spline needs at least 4 points
                    try:
                        lat_spline = CubicSpline(time_indices, valid_df['latitude'])
                        lon_spline = CubicSpline(time_indices, valid_df['longitude'])
                        
                        # Interpolate for all frames
                        all_indices = np.linspace(0, max(time_indices), len(frame_seconds))
                        interp_lat = lat_spline(all_indices)
                        interp_lon = lon_spline(all_indices)
                    except Exception as e:
                        print(f"Spline interpolation error: {e}")
                        # Fallback to linear interpolation
                        lat_interp = interp1d(time_indices, valid_df['latitude'], 
                                            bounds_error=False, fill_value='extrapolate')
                        lon_interp = interp1d(time_indices, valid_df['longitude'],
                                            bounds_error=False, fill_value='extrapolate')
                        
                        all_indices = np.linspace(0, max(time_indices), len(frame_seconds))
                        interp_lat = lat_interp(all_indices)
                        interp_lon = lon_interp(all_indices)
                else:
                    # Linear interpolation for fewer points
                    lat_interp = interp1d(time_indices, valid_df['latitude'], 
                                        bounds_error=False, fill_value='extrapolate')
                    lon_interp = interp1d(time_indices, valid_df['longitude'],
                                        bounds_error=False, fill_value='extrapolate')
                    
                    all_indices = np.linspace(0, max(time_indices), len(frame_seconds))
                    interp_lat = lat_interp(all_indices)
                    interp_lon = lon_interp(all_indices)
                
                # Create interpolated positions
                interpolated_positions = []
                for lat, lon in zip(interp_lat, interp_lon):
                    pos = self.merge(lat, lon)
                    interpolated_positions.append(pos if pos else 'N0.00000E0.00000')
                
                # Interpolate speeds
                if 'speed' in df.columns:
                    speed_interp = interp1d(np.arange(len(df)), df['speed'], 
                                          bounds_error=False, fill_value='extrapolate')
                    interp_speeds = speed_interp(all_indices)
                    interp_speeds = np.round(interp_speeds).astype(int)
                else:
                    interp_speeds = np.zeros(len(frames), dtype=int)
                
                # Build final dataframe
                for i, frame in enumerate(frames):
                    if i < len(interpolated_positions):
                        final_df['Frame'].append(frame)
                        final_df['Position'].append(interpolated_positions[i])
                        final_df['Speed'].append(int(interp_speeds[i]) if i < len(interp_speeds) else 0)
            else:
                # Not enough valid GPS points for interpolation
                for frame in frames:
                    final_df['Frame'].append(frame)
                    final_df['Position'].append('N0.00000E0.00000')
                    final_df['Speed'].append(0)
    
            final_df = pd.DataFrame(final_df)
            
            # Save CSV
            if video_path.lower().endswith('.mp4'):
                csv_path = video_path[:-4] + '.csv'
                final_df.to_csv(csv_path, index=False)
                print(f"Saved GPS data to {csv_path}")
            
            return final_df, True
        
        except Exception as ex:
            print(f"GPS extraction error: {ex}")
            return 'GPS Error', False


    def combine_csv_files(self, directory_path, output_file=None, recursive=True):
        """
        Combine all CSV files in a directory and its subdirectories.
        
        Args:
            directory_path (str): Path to directory containing CSV files.
            output_file (str, optional): Path to save the combined CSV.
            recursive (bool): Whether to search subdirectories for CSV files.
        
        Returns:
            pd.DataFrame: Combined DataFrame.
        """
        all_dataframes = []

        # Find all CSV files (recursively if enabled)
        if recursive:
            csv_files = [os.path.join(root, file)
                        for root, _, files in os.walk(directory_path)
                        for file in files if file.lower().endswith(".csv")]
        else:
            csv_files = sorted(glob.glob(os.path.join(directory_path, '*.csv')))

        if not csv_files:
            print("No CSV files found in the specified directory")
            return None

        # Sorting CSVs based on extracted timestamp
        def extract_timestamp(filename):
            match = re.search(r'(\d{4})_(\d{4})_(\d{6})', filename)  # Extract YYYY_MMDD_HHMMSS
            if match:
                return int(match.group(0).replace("_", ""))  # Convert to int for proper sorting
            return float('inf')  # Push unmatched filenames to the end

        csv_files.sort(key=lambda x: extract_timestamp(os.path.basename(x)))

        # Read CSV files and combine them
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                video_name = os.path.basename(csv_file).replace('.csv', '')
                df['Video_Name'] = video_name
                all_dataframes.append(df)
            except Exception as e:
                print(f"Error loading {csv_file}: {str(e)}")

        # Combine DataFrames if there is data
        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            if output_file:
                combined_df.to_csv(output_file, index=False)
            return combined_df
        else:
            print("No dataframes to combine")
            return None


    def process_csv(self, input_file, output_file):
        """
        Process the input CSV and generate cleaned data with improved handling of invalid GPS data
        
        Args:
            input_file (str): Path to input CSV
            output_file (str): Path to save processed CSV
        
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        df = pd.read_csv(input_file)
        
        # Clean and identify valid GPS positions
        df['Valid_GPS'] = df['Position'].apply(self.is_valid_gps)
        
        # Extract latitude and longitude for all points
        # For invalid points, this will set lat/lon to 0.0, which we'll handle with interpolation
        df[['Latitude', 'Longitude']] = df['Position'].apply(lambda pos: pd.Series(self.split(pos)))
        
        # Group by Video_Name to handle each video segment separately
        video_groups = df.groupby('Video_Name')
        processed_dfs = []
        
        for video_name, video_df in video_groups:
            video_df = video_df.reset_index(drop=True)
            
            # Skip videos with no valid GPS data
            if not video_df['Valid_GPS'].any():
                print(f"Skipping {video_name} - no valid GPS data")
                continue
                
            # For videos with some valid GPS data, interpolate missing values
            if video_df['Valid_GPS'].sum() >= 2:
                # Get indices of valid GPS points
                valid_indices = np.where(video_df['Valid_GPS'])[0]
                
                # Perform spline interpolation on latitude and longitude separately
                if len(valid_indices) >= 4:  # Need at least 4 points for cubic spline
                    try:
                        # For latitude
                        lat_spline = CubicSpline(
                            valid_indices, 
                            video_df.loc[valid_indices, 'Latitude']
                        )
                        
                        # For longitude
                        lon_spline = CubicSpline(
                            valid_indices, 
                            video_df.loc[valid_indices, 'Longitude']
                        )
                        
                        # Apply interpolation to all indices
                        all_indices = np.arange(len(video_df))
                        video_df['Latitude'] = lat_spline(all_indices)
                        video_df['Longitude'] = lon_spline(all_indices)
                    except Exception as e:
                        print(f"Cubic spline interpolation failed for {video_name}: {e}")
                        # Fall back to linear interpolation
                        lat_interp = interp1d(
                            valid_indices, 
                            video_df.loc[valid_indices, 'Latitude'],
                            bounds_error=False, 
                            fill_value='extrapolate'
                        )
                        
                        lon_interp = interp1d(
                            valid_indices, 
                            video_df.loc[valid_indices, 'Longitude'],
                            bounds_error=False, 
                            fill_value='extrapolate'
                        )
                        
                        video_df['Latitude'] = lat_interp(all_indices)
                        video_df['Longitude'] = lon_interp(all_indices)
                else:
                    # Linear interpolation for fewer points
                    lat_interp = interp1d(
                        valid_indices, 
                        video_df.loc[valid_indices, 'Latitude'],
                        bounds_error=False, 
                        fill_value='extrapolate'
                    )
                    
                    lon_interp = interp1d(
                        valid_indices, 
                        video_df.loc[valid_indices, 'Longitude'],
                        bounds_error=False, 
                        fill_value='extrapolate'
                    )
                    
                    all_indices = np.arange(len(video_df))
                    video_df['Latitude'] = lat_interp(all_indices)
                    video_df['Longitude'] = lon_interp(all_indices)
                
                # Recreate the Position column from interpolated coordinates
                for idx in video_df.index:
                    lat = video_df.loc[idx, 'Latitude']
                    lon = video_df.loc[idx, 'Longitude']
                    pos = self.merge(lat, lon)
                    video_df.loc[idx, 'Position'] = pos if pos else 'N0.00000E0.00000'
                
                processed_dfs.append(video_df)
            else:
                # Not enough valid points for interpolation
                print(f"Not enough valid GPS points in {video_name} for interpolation")
        
        # Combine processed video segments
        if processed_dfs:
            df = pd.concat(processed_dfs, ignore_index=True)
        else:
            print("No videos with valid GPS data found")
            return None
        
        # Select every 30th frame
        df = df[df['Frame'] % 30 == 0].reset_index(drop=True)
        
        # Initialize columns
        df['Startframe'] = df['Frame']
        df['Endframe'] = df['Startframe'].shift(-1, fill_value=df['Startframe'].iloc[-1]) - 2
        df['Gdist'] = 0
        df['Distance'] = 0.0
        df['Chainage'] = 0.0
        
        # Get last frame number per video
        last_frames = df.groupby('Video_Name')['Frame'].max().to_dict()
      
        # Replace -2 in Endframe with the last frame of the corresponding video
        df['Endframe'] = df.apply(lambda row: last_frames[row['Video_Name']] if row['Endframe'] == -2 else row['Endframe'], axis=1)
                              
        # Calculate distances and cumulative chainage
        total_distance = 0
        for i in range(1, len(df)):
            lat1, lon1 = df.loc[i - 1, ['Latitude', 'Longitude']]
            lat2, lon2 = df.loc[i, ['Latitude', 'Longitude']]
            
            # Skip invalid GPS points
            if (pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2) or
                (lat1 == 0 and lon1 == 0) or (lat2 == 0 and lon2 == 0)):
                df.loc[i, 'Distance'] = 0
                df.loc[i, 'Chainage'] = total_distance
                continue
                
            distance = self.haversine_distance(lat1, lon1, lat2, lon2)
            
            # Skip implausible distances (GPS jumps)
            if distance > 500:  # More than 500 meters between frames likely indicates a GPS error
                print(f"Large distance detected ({distance:.2f}m) at index {i}. Likely GPS error.")
                distance = 0
                
            distance /= 1000  # Convert to kilometers
            df.loc[i, 'Distance'] = distance
            total_distance += distance
            df.loc[i, 'Chainage'] = total_distance
        
        # Save processed data
        df_output = df[['Position', 'Startframe', 'Endframe', 'Speed', 'Video_Name', 
                       'Gdist', 'Distance', 'Chainage', 'Latitude', 'Longitude']]
        df_output.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
        
        return df_output

    def delete_frames(self, file_path, video_name, start_frame, end_frame):
        """
        Delete specified frames and update the CSV
        
        Args:
            file_path (str): Path to CSV file
            video_name (str): Name of the video
            start_frame (int): Starting frame to delete
            end_frame (int): Ending frame to delete
        
        Returns:
            pd.DataFrame: Filtered DataFrame
        """
        df = pd.read_csv(file_path)
        
        # Remove frames in the specified range
        mask = ~((df['Video_Name'].str.strip() == video_name.strip()) & 
                 (df['Startframe'] >= start_frame) & 
                 (df['Startframe'] <= end_frame))
        df_filtered = df[mask].reset_index(drop=True)
        
        # Recalculate chainage after deletion
        df_filtered = self.recalculate_chainage(df_filtered)
        
        # Save updated CSV (overwrite original)
        df_filtered.to_csv(file_path, index=False)
        
        print(f"Frames {start_frame} to {end_frame} in video '{video_name}' deleted and chainage updated.")
        
        return df_filtered

    def recalculate_chainage(self, df):
        """
        Recalculate chainage after frame deletion with improved handling of invalid GPS
        
        Args:
            df (pd.DataFrame): Input DataFrame
        
        Returns:
            pd.DataFrame: DataFrame with updated chainage
        """
        total_distance = 0
        df['Chainage'] = 0.0 
        
        for i in range(1, len(df)):
            lat1, lon1 = df.loc[i - 1, ['Latitude', 'Longitude']]
            lat2, lon2 = df.loc[i, ['Latitude', 'Longitude']]
            
            # Skip invalid GPS points
            if (pd.isna(lat1) or pd.isna(lon1) or pd.isna(lat2) or pd.isna(lon2) or
                (lat1 == 0 and lon1 == 0) or (lat2 == 0 and lon2 == 0)):
                df.loc[i, 'Distance'] = 0
                df.loc[i, 'Chainage'] = total_distance
                continue
                
            distance = self.haversine_distance(lat1, lon1, lat2, lon2)
            
            # Skip implausible distances (GPS jumps)
            if distance > 500:  # More than 500 meters between frames likely indicates a GPS error
                distance = 0
                
            distance /= 1000  # Convert to kilometers
            df.loc[i, 'Distance'] = distance
            total_distance += distance
            df.loc[i, 'Chainage'] = total_distance
        
        return df

    def create_map(self, file_path, output_html="map_visualization.html"):
        """
        Create an interactive map with route visualization
        
        Args:
            file_path (str): Path to CSV file
            output_html (str, optional): Path to save HTML map
        Returns:
            str: Path to generated HTML map
        """
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['Latitude', 'Longitude'])
        
        # Filter out invalid coordinates (0,0)
        df = df[(df['Latitude'] != 0) | (df['Longitude'] != 0)]
        
        if len(df) == 0:
            print("No valid GPS coordinates found for mapping")
            return None
                
        # Initialize map
        start_coords = (df.loc[0, 'Latitude'], df.loc[0, 'Longitude'])
        m = folium.Map(location=start_coords, zoom_start=15)

        # Draw polyline
        folium.PolyLine(
            list(zip(df['Latitude'], df['Longitude'])), 
            color="blue", weight=3, opacity=0.7
        ).add_to(m)

        # Add markers only for every 100m chainage
        chainage_intervals = set([round(x, 1) for x in np.arange(0, df['Chainage'].max() + 0.1, 0.1)])

        added_markers = set()

        for _, row in df.iterrows():
            closest_chainage = min(chainage_intervals, key=lambda x: abs(x - row['Chainage']))
        
            if closest_chainage not in added_markers:
                folium.Marker(
                    location=(row['Latitude'], row['Longitude']),
                    popup=f"Chainage: {float(row['Chainage'])}km<br>Frame: {row['Startframe']}<br>Video: {row['Video_Name']}<br>GPS: {row['Position']}",
                    icon=folium.Icon(color="red", icon="info-sign")
                ).add_to(m)
                added_markers.add(closest_chainage)
        
        # First add the built-in click handler
        folium.LatLngPopup().add_to(m)
        
        # Then add a custom JavaScript to modify the popup content format
        custom_js = """
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                var map = document.querySelector('.folium-map');
                    if (map) {
                        map.addEventListener('click', function(e) {
                            setTimeout(function() {
                                var popup = document.querySelector('.leaflet-popup-content');
                                if (popup) {
                                    var latLng = popup.innerText.match(/-?\\d+\\.\\d+/g);
                                    if (latLng && latLng.length >= 2) {
                                        var lat = parseFloat(latLng[0]).toFixed(6);
                                        var lng = parseFloat(latLng[1]).toFixed(6);
                                        popup.innerText = lat + ", " + lng;
                                    }
                                }
                            }, 100);
                        });
                    }
                });
        </script>
        """
        
        # Add the custom JS to the map
        m.get_root().html.add_child(folium.Element(custom_js))
        
        # Save map
        m.save(output_html)
        print(f"Map saved as {output_html}.")
        return output_html

def process_and_generate_map(csv_path, start_frame, end_frame, video_name, gps_coords):
    interpolator = VideoGPSProcessor()
    
    # Read the CSV file and perform interpolation
    df = interpolator.interpolate_gps_in_csv(csv_path, start_frame, end_frame, video_name, gps_coords)
    
    # Recalculate chainage
    df = interpolator.recalculate_chainage(df)
    
    # Extract the directory path from the input CSV path
    output_dir = os.path.dirname(csv_path)
    
    # Define the output file paths
    updated_csv_path = os.path.join(output_dir, "updated_output.csv")
    map_html_path = os.path.join(output_dir, "updated_map.html")
    
    # Save the updated CSV
    df.to_csv(updated_csv_path, index=False)
    
    # Generate the map
    map_path = interpolator.create_map(updated_csv_path, map_html_path)
    
    return f"Interpolation completed. Updated CSV saved at {updated_csv_path}. Map generated at {map_path}"
# def process_and_generate_map(csv_path, start_frame, end_frame, video_name, gps_coords):
#     interpolator = VideoGPSProcessor()
#     df = interpolator.interpolate_gps_in_csv(csv_path, start_frame, end_frame, video_name, gps_coords)
#     df = interpolator.recalculate_chainage(df)
#     df.to_csv("final_output.csv", index=False)
#     map_path = interpolator.create_map("final_output.csv", "map.html")
#     return f"Interpolation completed. Map generated at {map_path}"
def create_gradio_interface():
    """Create Gradio interface for video GPS processing"""
    processor = VideoGPSProcessor()

    with gr.Blocks() as demo:
        gr.Markdown("# 🚗 GPS Video Processing Tool")
        
        # Status Display
        status_output = gr.Textbox(label="Status", interactive=False)
        
        # Video Processing Section
        with gr.Tab("Process Videos"):
            with gr.Row():
                directory_input = gr.Textbox(label="Video Directory Path")
                process_btn = gr.Button("Process Videos")
            
            process_btn.click(
                fn=lambda directory: processor.process_videos(directory),
                inputs=directory_input,
                outputs=status_output
            )
        
        # Frame Deletion Section
        with gr.Tab("Delete Video Frames"):
            with gr.Row():
                csv_path_del_input = gr.Textbox(label="Processed CSV Path")
                video_name_input = gr.Textbox(label="Video Name")
                start_frame_input = gr.Number(label="Start Frame", minimum=0)
                end_frame_input = gr.Number(label="End Frame", minimum=0)
                delete_frames_btn = gr.Button("Delete Frames")
            
            delete_frames_btn.click(
                fn=lambda csv_path, video_name, start_frame, end_frame: 
                    processor.delete_video_frames(csv_path, video_name, start_frame, end_frame),
                inputs=[
                    csv_path_del_input, 
                    video_name_input, 
                    start_frame_input, 
                    end_frame_input
                ],
                outputs=status_output
            )
        #Interpolate GPS Points 
        with gr.Tab("Interpolate GPS Points"):
            with gr.Row():
                csv_path_interp_input = gr.Textbox(label="Processed CSV Path")
                video_name_interp_input = gr.Textbox(label="Video Name")
                start_frame_interp_input = gr.Number(label="Start Frame", minimum=0)
                end_frame_interp_input = gr.Number(label="End Frame", minimum=0)
                gps_points_input = gr.Textbox(label="GPS Points (Format: lat1,lon1,lat2,lon2,...)")
                process_and_generate_map_btn = gr.Button("Process and Generate Map")
            
            process_and_generate_map_btn.click(
                fn=lambda csv_path, video_name, start_frame, end_frame, gps_points: 
                    process_and_generate_map(
                        csv_path, 
                        start_frame, 
                        end_frame, 
                        video_name, 
                        [float(coord) for coord in gps_points.split(',')]
                    ),
                inputs=[
                    csv_path_interp_input, 
                    video_name_interp_input, 
                    start_frame_interp_input, 
                    end_frame_interp_input, 
                    gps_points_input
                ],
                outputs=status_output
            )
    
    return demo
import os
import glob
import re
import concurrent.futures
import webbrowser

def process_videos(self, directory_path):
    """
    Comprehensive video processing workflow for videos in a directory and its subdirectories.
    
    Args:
        directory_path (str): Path to directory containing videos.
    
    Returns:
        str: Processing status message.
    """
    try:
        # Validate directory
        if not os.path.isdir(directory_path):
            return "Invalid directory path"

        # Find all MP4 files in the directory and its subdirectories
        mp4_files = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(".mp4"):
                    mp4_files.append(os.path.join(root, file))

        if not mp4_files:
            return "No MP4 videos found in the directory or subdirectories"

        # Sorting videos based on extracted timestamp
        def extract_timestamp(filename):
            match = re.search(r'(\d{4})_(\d{4})_(\d{6})', filename)  # Extract YYYY_MMDD_HHMMSS
            if match:
                return int(match.group(0).replace("_", ""))  # Convert to int for proper sorting
            return float('inf')  # Push unmatched filenames to the end

        mp4_files.sort(key=lambda x: extract_timestamp(os.path.basename(x)))

        # Process each video to generate CSVs in parallel
        processed_videos = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.get_gps, video_path): video_path for video_path in mp4_files}
            for future in concurrent.futures.as_completed(futures):
                video_path = futures[future]
                try:
                    result, success = future.result()
                    if success:
                        processed_videos += 1
                except Exception as e:
                    print(f"Error processing {video_path}: {e}")

        # Create maps folder in the given directory
        maps_folder = os.path.join(directory_path, 'maps_')
        if not os.path.exists(maps_folder):
            os.makedirs(maps_folder)

        # Combine CSV files from all directories and subdirectories
        combined_csv_path = os.path.join(maps_folder, 'combined_output.csv')
        combined_df = self.combine_csv_files(directory_path, combined_csv_path, recursive=True)

        # Process combined CSV and save it in the maps folder
        processed_csv_path = os.path.join(maps_folder, 'processed_output.csv')
        processed_df = self.process_csv(combined_csv_path, processed_csv_path)

        # Generate and open map
        map_path = os.path.join(maps_folder, 'processed_output_map.html')
        self.create_map(processed_csv_path, map_path)
        
        # Open the map in a new browser tab
        webbrowser.open(f"file://{os.path.abspath(map_path)}")
        
        return (f"Processed {processed_videos} videos. "
                f"Combined CSV: {combined_csv_path}. "
                f"Processed CSV: {processed_csv_path}. "
                f"Map generated and opened in a new tab.")
    
    except Exception as e:
        return f"Error processing videos: {str(e)}"


def delete_video_frames(self, csv_path, video_name, start_frame, end_frame):
    """
    Delete frames and provide status
    
    Args:
        csv_path (str): Path to processed CSV
        video_name (str): Name of video
        start_frame (int): Start frame to delete
        end_frame (int): End frame to delete
    
    Returns:
        str: Deletion status message
    """
    try:
        # Validate inputs
        if not os.path.exists(csv_path):
            return "CSV file does not exist"
        
        if not video_name:
            return "Please provide a video name"
        
        if start_frame > end_frame:
            return "Start frame must be less than or equal to end frame"

        # Delete frames
        self.delete_frames(csv_path, video_name, start_frame, end_frame)
        
        # Regenerate the map
        map_path = csv_path.replace('.csv', '_map.html')
        self.create_map(csv_path, map_path)
        
        # Open the map in a new browser tab
        webbrowser.open(f"file://{os.path.abspath(map_path)}")
        
        return (f"Frames {start_frame}-{end_frame} deleted from {video_name}. "
                "CSV has been updated. Map regenerated and opened in a new tab.")
    
    except Exception as e:
        return f"Error deleting frames: {str(e)}"

# Attach methods to the class
VideoGPSProcessor.process_videos = process_videos
VideoGPSProcessor.delete_video_frames = delete_video_frames

# Launch the application
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)