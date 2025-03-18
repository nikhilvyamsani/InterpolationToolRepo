Video GPS Processing Tool
This repository contains a Python-based tool for processing video files to extract and interpolate GPS data, and then visualize the data on an interactive map. The tool is built using Gradio for the user interface, Pandas for data manipulation, Folium for map visualization, and OpenCV for video processing. It also includes a specific version of ExifTool for metadata extraction.

Features
GPS Data Extraction: Extract GPS metadata from video files using a specific version of ExifTool.
GPS Interpolation: Interpolate missing GPS data using cubic spline or linear interpolation.
CSV Processing: Combine and process multiple CSV files containing GPS data.
Frame Deletion: Delete specific frames from the processed data and update the CSV.
Map Visualization: Generate interactive maps using Folium to visualize the GPS route.
Gradio Interface: User-friendly interface for processing videos, deleting frames, and generating maps.

Installation
Clone the repository: 
 git clone https://github.com/your-username/video-gps-processing-tool.git
 cd video-gps-processing-tool

Install Python dependencies:
 pip install -r requirements.txt
 
Set up ExifTool:
 chmod +x exiftool
Usage :

Process Videos:
Launch the gradio interface:
python tool_final.py

Navigate to the "Process Videos" tab.
Enter the directory path containing your video files.
Click "Process Videos" to extract GPS data and generate a combined CSV.

Delete Frames:
Navigate to the "Delete Video Frames" tab.
Enter the path to the processed CSV, video name, and frame range.
Click "Delete Frames" to remove the specified frames and update the CSV.

Interpolate GPS Points:
Navigate to the "Interpolate GPS Points" tab.
Enter the path to the processed CSV, video name, frame range, and GPS coordinates.
Click "Process and Generate Map" to interpolate GPS data and generate a new map.

View the Map:
The tool will automatically open the generated map in your default web browser.

Requirements:
Python 3.7+, Gradio, Pandas, NumPy, Folium, OpenCV, Geopy, Scipy, Specific version of ExifTool V12.22 (attached in the repo)
