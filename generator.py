#!/usr/bin/env python3

import ffmpeg
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import gpxpy
import gpxpy.gpx
from PIL import Image, ImageDraw, ImageFont
import os
from typing import List
import bisect
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import select
import sys

@dataclass
class VideoMetadata:
    """Class representing video metadata."""
    width: int
    height: int
    fps: float
    fps_string: str
    duration: float
    bitrate: int
    frame_count: int  # Exact number of frames in the video

@dataclass
class GPSTrackPoint:
    """Class representing a GPS track point."""
    latitude: float
    longitude: float
    elevation: float
    timestamp: datetime
    fix_type: str
    pdop: float

@dataclass
class FrameParams:
    """Parameters needed for generating a single frame."""
    frame_num: int
    frame_time: datetime
    track_points: List[GPSTrackPoint]
    map_size: tuple
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    font: ImageFont.FreeTypeFont

def parse_gpx_file(gpx_path: str) -> list[GPSTrackPoint]:
    """
    Parse a GPX file and extract track points.
    
    Args:
        gpx_path (str): Path to the GPX file
        
    Returns:
        list[GPSTrackPoint]: List of GPS track points containing:
            - latitude: Latitude in decimal degrees
            - longitude: Longitude in decimal degrees
            - elevation: Elevation in meters
            - timestamp: UTC timestamp
            - fix_type: GPS fix type (e.g., '3d')
            - pdop: Position dilution of precision
    """
    try:
        with open(gpx_path, 'r') as gpx_file:
            gpx = gpxpy.parse(gpx_file)
            
        track_points = []
        
        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    # Extract additional data from point extensions
                    fix_type = None
                    pdop = None
                    
                    if point.extensions:
                        for extension in point.extensions:
                            if extension.tag.endswith('fix'):
                                fix_type = extension.text
                            elif extension.tag.endswith('pdop'):
                                pdop = float(extension.text)
                    
                    track_point = GPSTrackPoint(
                        latitude=point.latitude,
                        longitude=point.longitude,
                        elevation=point.elevation,
                        timestamp=point.time,
                        fix_type=fix_type or 'unknown',
                        pdop=pdop or 0.0
                    )
                    track_points.append(track_point)
        
        return track_points
    
    except Exception as e:
        raise Exception(f"Error parsing GPX file: {str(e)}")

def extract_video_metadata(video_path):
    """
    Extract metadata from an MP4 video file.
    
    Args:
        video_path (str): Path to the MP4 video file
        
    Returns:
        VideoMetadata: Object containing video metadata including:
            - width: Video width in pixels
            - height: Video height in pixels
            - fps: Frames per second
            - duration: Video duration in seconds
            - bitrate: Video bitrate in bits per second
            - frame_count: Exact number of frames in the video
    """
    try:
        # Get video stream information
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        if video_stream is None:
            raise ValueError("No video stream found in the file")
        
        # Get exact frame count if available, otherwise calculate it
        frame_count = int(video_stream.get('nb_frames', 0))
        if frame_count == 0:
            frame_count = int(float(probe['format']['duration']) * eval(video_stream['r_frame_rate']))
        
        # Create and return VideoMetadata object
        return VideoMetadata(
            width=int(video_stream['width']),
            height=int(video_stream['height']),
            fps=eval(video_stream['r_frame_rate']),  # Convert fraction string to float
            fps_string=video_stream['r_frame_rate'],
            duration=float(probe['format']['duration']),
            bitrate=int(probe['format']['bit_rate']),
            frame_count=frame_count
        )
    
    except ffmpeg.Error as e:
        raise Exception(f"Error processing video file: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

def generate_and_save_frame(params: FrameParams, output_dir: str) -> None:
    """
    Generate a single frame with GPS visualization and save it to disk.
    
    Args:
        params: FrameParams object containing all necessary parameters
        output_dir: Directory to save the frame
    """
    def lat_lon_to_pixel(lat: float, lon: float) -> tuple:
        """Convert latitude/longitude to pixel coordinates."""
        x = int((lon - params.min_lon) / (params.max_lon - params.min_lon) * params.map_size[0])
        y = int((1 - (lat - params.min_lat) / (params.max_lat - params.min_lat)) * params.map_size[1])
        return (x, y)
    
    # Find the closest GPS point to this timestamp using bisect
    timestamps = [p.timestamp for p in params.track_points]
    closest_idx = bisect.bisect_left(timestamps, params.frame_time)
    
    # Handle edge cases
    if closest_idx == len(params.track_points):
        closest_idx = len(params.track_points) - 1
    else:
        # Compare with previous point to find closest
        if abs((params.track_points[closest_idx].timestamp - params.frame_time).total_seconds()) >= \
           abs((params.track_points[closest_idx-1].timestamp - params.frame_time).total_seconds()):
            closest_idx = closest_idx - 1

    closest_point = params.track_points[closest_idx]
    
    # Create a new image for the map
    map_image = Image.new('RGBA', params.map_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(map_image)
    
    # Draw the complete track
    track_points_pixels = [lat_lon_to_pixel(p.latitude, p.longitude) for p in params.track_points]
    draw.line(track_points_pixels, fill='blue', width=2)
    
    # Draw the current position
    current_pos = lat_lon_to_pixel(closest_point.latitude, closest_point.longitude)
    draw.ellipse(
        (current_pos[0]-8, current_pos[1]-8, current_pos[0]+8, current_pos[1]+8),
        fill='red',
        outline='white',
        width=2
    )
    
    # Add GPS information
    info_text = [
        f"Lat: {closest_point.latitude:.6f}°",
        f"Lon: {closest_point.longitude:.6f}°",
        f"Elev: {closest_point.elevation:.1f}m",
        f"Speed: {calculate_speed(closest_idx, params.track_points):.1f} km/h",
        f"Time: {closest_point.timestamp.strftime('%H:%M:%S')}"
    ]
    
    # Draw text with black outline for better visibility
    for i, text in enumerate(info_text):
        y_pos = 10 + i * 30
        # Draw black outline
        for offset_x, offset_y in [(-1,-1), (-1,1), (1,-1), (1,1)]:
            draw.text((12 + offset_x, y_pos + offset_y), text, font=params.font, fill='black')
        # Draw white text
        draw.text((12, y_pos), text, font=params.font, fill='white')
    
    # Save the frame
    map_image.save(os.path.join(output_dir, f"frame_{params.frame_num:06d}.png"))

def generate_map_frames(
    video_metadata: VideoMetadata,
    track_points: List[GPSTrackPoint],
    output_dir: str,
    map_size: tuple = (800, 600)
) -> None:
    """
    Generate PNG frames with GPS path visualization using parallel processing.
    
    Args:
        video_metadata: Video metadata containing resolution and fps
        track_points: List of GPS track points
        output_dir: Directory to save generated frames
        map_size: Size of the visualization in pixels (default: 800x600)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate frame duration
    frame_duration = 1.0 / video_metadata.fps
    
    # Calculate bounds of the track
    min_lat = min(p.latitude for p in track_points)
    max_lat = max(p.latitude for p in track_points)
    min_lon = min(p.longitude for p in track_points)
    max_lon = max(p.longitude for p in track_points)
    
    # Add some padding to the bounds
    lat_padding = (max_lat - min_lat) * 0.1
    lon_padding = (max_lon - min_lon) * 0.1
    min_lat -= lat_padding
    max_lat += lat_padding
    min_lon -= lon_padding
    max_lon += lon_padding
    
    # Load font once
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Prepare frame parameters for parallel processing
    total_frames = video_metadata.frame_count
    frame_params = []
    for frame_num in range(total_frames):
        frame_time = track_points[0].timestamp + timedelta(seconds=frame_num * frame_duration)
        params = FrameParams(
            frame_num=frame_num,
            frame_time=frame_time,
            track_points=track_points,
            map_size=map_size,
            min_lat=min_lat,
            max_lat=max_lat,
            min_lon=min_lon,
            max_lon=max_lon,
            font=font
        )
        frame_params.append(params)
    
    # Use multiprocessing to generate and save frames
    num_processes = max(1, cpu_count() - 1)  # Leave one CPU free
    print(f"Generating frames using {num_processes} processes...")
    
    with Pool(num_processes) as pool:
        # Create a partial function with the output directory
        generate_and_save_with_dir = partial(generate_and_save_frame, output_dir=output_dir)
        
        # Process frames in parallel
        completed_frames = 0
        for _ in pool.imap_unordered(generate_and_save_with_dir, frame_params):
            completed_frames += 1
            progress = completed_frames / total_frames * 100
            print(f"\rProgress: {completed_frames}/{total_frames} ({progress:.1f}%)", end='', flush=True)
    
    print()  # Add newline after progress is complete

def calculate_speed(closes_idx: int, all_points: List[GPSTrackPoint]) -> float:
    """Calculate speed in km/h based on the current point and previous points."""
    # Find the previous point
    if closes_idx == 0:
        return 0.0
        
    closest_point = all_points[closes_idx]
    prev_point = all_points[closes_idx - 1]
    
    # Calculate time difference in hours
    time_diff = (closest_point.timestamp - prev_point.timestamp).total_seconds() / 3600
    
    if time_diff == 0:
        return 0.0
        
    # Calculate distance using haversine formula
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1 = radians(prev_point.latitude), radians(prev_point.longitude)
    lat2, lon2 = radians(closest_point.latitude), radians(closest_point.longitude)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    
    # Calculate speed in km/h
    speed = distance / time_diff
    return speed

def composite_video_with_map(video_path: str, map_frames_dir: str, output_path: str) -> None:
    """
    Composite the original video with the map frames using ffmpeg.
    
    Args:
        video_path: Path to the original video file
        map_frames_dir: Directory containing the map frames
        output_path: Path where the final video will be saved
    """
    try:
        # Get the frame pattern for the map frames
        frame_pattern = os.path.join(map_frames_dir, "frame_%06d.png")
        
        # Get video duration for progress calculation
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        
        # Verify input files exist
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(map_frames_dir):
            raise FileNotFoundError(f"Map frames directory not found: {map_frames_dir}")
            
        # Check if we have any map frames
        map_frames = [f for f in os.listdir(map_frames_dir) if f.startswith('frame_') and f.endswith('.png')]
        if not map_frames:
            raise FileNotFoundError(f"No map frames found in {map_frames_dir}")
            
        print(f"Found {len(map_frames)} map frames")
        
        # Construct ffmpeg command to overlay map frames on video
        stream = (
            ffmpeg
            .input(video_path)
            .overlay(
                ffmpeg.input(frame_pattern, thread_queue_size=512),  # Increase queue size for frame reading
                x=10,  # 10px margin from left
                y=10   # 10px margin from top
            )
            .output(
                output_path,
                progress='pipe:1',  # Output progress to stdout
                loglevel='warning',  # Show warnings and errors
            )
            .overwrite_output()
        )
        
        # Run the ffmpeg command and capture progress
        process = stream.run_async(pipe_stdout=True, pipe_stderr=True)
        
        # Process ffmpeg output to show progress
        last_progress = 0
        stall_count = 0
        
        # Set up non-blocking reads
        stdout_fd = process.stdout.fileno()
        stderr_fd = process.stderr.fileno()
        
        # Set up line buffers
        stdout_buffer = ""
        stderr_buffer = ""
        
        while True:
            # Check if process has finished
            if process.poll() is not None:
                break
                
            # Use select to check for available data
            reads = [stdout_fd, stderr_fd]
            ret = select.select(reads, [], [], 0.1)  # 0.1 second timeout
            
            if not ret[0]:  # No data available
                continue
                
            # Read from stdout
            if stdout_fd in ret[0]:
                data = os.read(stdout_fd, 1024).decode('utf8', errors='replace')
                if not data:  # EOF
                    continue
                stdout_buffer += data
                
                # Process complete lines
                while '\n' in stdout_buffer:
                    line, stdout_buffer = stdout_buffer.split('\n', 1)
                    if 'time=' in line:
                        try:
                            time_str = line.split('time=')[1].split()[0]
                            h, m, s = time_str.split(':')
                            current_time = float(h) * 3600 + float(m) * 60 + float(s)
                            progress = (current_time / duration) * 100
                            
                            # Check for stalled progress
                            if progress == last_progress:
                                stall_count += 1
                                if stall_count > 10:  # If stalled for 10 updates
                                    print(f"\nWarning: Progress stalled at {progress:.1f}%")
                            else:
                                stall_count = 0
                                
                            last_progress = progress
                            print(f"\rCompositing video: {progress:.1f}%", end='', flush=True)
                        except Exception as e:
                            print(f"\nError parsing progress: {str(e)}")
            
            # Read from stderr
            if stderr_fd in ret[0]:
                data = os.read(stderr_fd, 1024).decode('utf8', errors='replace')
                if not data:  # EOF
                    continue
                stderr_buffer += data
                
                # Process complete lines
                while '\n' in stderr_buffer:
                    line, stderr_buffer = stderr_buffer.split('\n', 1)
                    print(f"\nFFmpeg (err): {line.strip()}")
        
        # Process any remaining data in buffers
        if stdout_buffer:
            print(f"\nFFmpeg (out): {stdout_buffer.strip()}")
        if stderr_buffer:
            print(f"\nFFmpeg (err): {stderr_buffer.strip()}")
        
        # Wait for process to complete and check return code
        return_code = process.wait()
        if return_code != 0:
            raise Exception(f"FFmpeg process failed with return code {return_code}")
            
        print(f"\nComposited video saved to: {output_path}")
        
    except ffmpeg.Error as e:
        print(f"\nFFmpeg error output: {e.stderr.decode() if e.stderr else 'No error output'}")
        raise Exception(f"Error compositing video: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error during video composition: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract metadata from an MP4 video file and GPS data')
    parser.add_argument('--video-file', required=True, help='Path to the MP4 video file')
    parser.add_argument('--gps-file', required=True, help='Path to the GPX file')
    parser.add_argument('--output-dir', required=True, help='Directory to save generated frames')
    parser.add_argument('--skip-generation', action='store_true', default=False, help='Skip generation of frames')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        metadata = extract_video_metadata(args.video_file)
        track_points = parse_gpx_file(args.gps_file)
        
        map_output_dir = os.path.join(args.output_dir, "map")
        map_size = (800, 600)

        # Generate frames
        if not args.skip_generation:
            generate_map_frames(
                video_metadata=metadata,
                track_points=track_points,
                output_dir=map_output_dir,
                map_size=map_size
            )
        
        # Composite the video with map frames
        output_video = os.path.join(args.output_dir, "output_with_map.mp4")
        composite_video_with_map(
            video_path=args.video_file,
            map_frames_dir=map_output_dir,
            output_path=output_video
        )
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
