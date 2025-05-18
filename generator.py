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
            print("Warn: calculating frame count from duration and framerate")
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

def generate_map_video(
    video_metadata: VideoMetadata,
    track_points: List[GPSTrackPoint],
    output_path: str,
    map_size: tuple = (800, 600)
) -> None:
    """
    Generate a transparent video with GPS path visualization using parallel processing.
    
    Args:
        video_metadata: Video metadata containing resolution and fps
        track_points: List of GPS track points
        output_path: Path to save the generated video
        map_size: Size of the visualization in pixels (default: 800x600)
    """
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

    # Start FFmpeg process to encode the video
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgba', s=f'{map_size[0]}x{map_size[1]}', r=video_metadata.fps)
        .output(output_path, vcodec='prores_ks', profile='4444', pix_fmt='yuva444p10le')
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    try:
        # Generate frames sequentially and write them to the pipe
        total_frames = video_metadata.frame_count
        for frame_num in range(total_frames):
            frame_time = track_points[0].timestamp + timedelta(seconds=frame_num * frame_duration)
            
            # Find the closest GPS point to this timestamp using bisect
            timestamps = [p.timestamp for p in track_points]
            closest_idx = bisect.bisect_left(timestamps, frame_time)
            
            # Handle edge cases
            if closest_idx == len(track_points):
                closest_idx = len(track_points) - 1
            else:
                # Compare with previous point to find closest
                if abs((track_points[closest_idx].timestamp - frame_time).total_seconds()) >= \
                   abs((track_points[closest_idx-1].timestamp - frame_time).total_seconds()):
                    closest_idx = closest_idx - 1

            closest_point = track_points[closest_idx]
            
            # Create a new image for the map
            map_image = Image.new('RGBA', map_size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(map_image)
            
            def lat_lon_to_pixel(lat: float, lon: float) -> tuple:
                """Convert latitude/longitude to pixel coordinates."""
                x = int((lon - min_lon) / (max_lon - min_lon) * map_size[0])
                y = int((1 - (lat - min_lat) / (max_lat - min_lat)) * map_size[1])
                return (x, y)
            
            # Draw the complete track
            track_points_pixels = [lat_lon_to_pixel(p.latitude, p.longitude) for p in track_points]
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
                f"Speed: {calculate_speed(closest_idx, track_points):.1f} km/h",
                f"Time: {closest_point.timestamp.strftime('%H:%M:%S')}"
            ]
            
            # Draw text with black outline for better visibility
            for i, text in enumerate(info_text):
                y_pos = 10 + i * 30
                # Draw black outline
                for offset_x, offset_y in [(-1,-1), (-1,1), (1,-1), (1,1)]:
                    draw.text((12 + offset_x, y_pos + offset_y), text, font=font, fill='black')
                # Draw white text
                draw.text((12, y_pos), text, font=font, fill='white')
            
            # Write the frame to the pipe
            process.stdin.write(map_image.tobytes())
            
            # Print progress
            progress = (frame_num + 1) / total_frames * 100
            print(f"\rGenerating overlay video: {progress:.1f}%", end='', flush=True)
        
        print()  # Add newline after progress is complete
        
        # Close the pipe and wait for FFmpeg to finish
        process.stdin.close()
        process.wait()
        
        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg process exited with code {process.returncode}")
            
    except Exception as e:
        process.stdin.close()
        process.wait()
        raise Exception(f"Error generating overlay video: {str(e)}")

def composite_video_with_map(metadata: VideoMetadata, video_path: str, overlay_video_path: str, output_path: str, max_duration: float = None, offset_seconds: float = 0.0) -> None:
    """
    Composite the original video with the overlay video using ffmpeg.

    Args:
        metadata: Metadata of the original video
        video_path: Path to the original video file
        overlay_video_path: Path to the overlay video file
        output_path: Path where the final video will be saved
        max_duration: Maximum duration of the output video in seconds (optional)
        offset_seconds: Time offset (in seconds) for the overlay layer (positive = delay, negative = advance)
    """
    try:
        # Duration of the final composition (may be limited by max_duration)
        duration = min(metadata.duration, max_duration) if max_duration else metadata.duration

        # Sanity checks
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(overlay_video_path):
            raise FileNotFoundError(f"Overlay video file not found: {overlay_video_path}")

        # Build FFmpeg filter graph
        video_in = ffmpeg.input(video_path)
        overlay_in = ffmpeg.input(overlay_video_path)

        # Positive offset (delay): duplicate the first frame so that it "fills" the gap
        if offset_seconds > 0:
            overlay_in = overlay_in.filter('tpad', start_duration=offset_seconds, start_mode='clone')
        # Negative offset (advance): shift PTS earlier so the overlay starts sooner
        elif offset_seconds < 0:
            overlay_in = overlay_in.filter('setpts', f"PTS+{offset_seconds}/TB")

        # Now overlay the two streams
        composed = ffmpeg.overlay(video_in, overlay_in, x=10, y=10)

        # Final output command
        stream = (
            ffmpeg
            .output(
                composed,
                output_path,
                t=duration,
                progress='pipe:1',  # Print progress to stdout
                loglevel='warning'   # Show warnings and errors only
            )
            .overwrite_output()
        )

        # Run ffmpeg asynchronously so we can parse the progress
        process = stream.run_async(pipe_stdout=True, pipe_stderr=True)

        last_progress = -1.0  # Track progress percentage
        stall_counter = 0

        # Non-blocking I/O setup
        fds = {process.stdout.fileno(): process.stdout, process.stderr.fileno(): process.stderr}
        buffers = {fd: "" for fd in fds}

        while True:
            if process.poll() is not None:
                break  # Process finished

            # Wait briefly for data on either descriptor
            ready_fds, _, _ = select.select(list(fds.keys()), [], [], 0.1)
            if not ready_fds:
                continue

            for fd in ready_fds:
                data = os.read(fd, 1024).decode('utf-8', errors='replace')
                if not data:
                    continue  # EOF
                buffers[fd] += data

                # Process complete lines
                while '\n' in buffers[fd]:
                    line, buffers[fd] = buffers[fd].split('\n', 1)
                    line = line.strip()

                    # stdout contains key=value pairs when -progress pipe is used
                    if fd == process.stdout.fileno() and line.startswith('out_time_ms'):
                        try:
                            # out_time_ms is the current time of the output in microseconds
                            time_us = int(line.split('=')[1])
                            current_sec = time_us / 1_000_000.0
                            progress_pct = (current_sec / duration) * 100.0

                            if progress_pct == last_progress:
                                stall_counter += 1
                                if stall_counter > 20:
                                    print(f"\nWarning: ffmpeg progress seems stalled at {progress_pct:.1f}%")
                            else:
                                stall_counter = 0
                            last_progress = progress_pct
                            print(f"\rCompositing video: {progress_pct:.1f}%", end='', flush=True)
                        except Exception:
                            # Ignore parsing errors, just continue
                            pass
                    elif fd == process.stderr.fileno():
                        # Print ffmpeg warnings/errors verbosely
                        if line:
                            print(f"\nFFmpeg (err): {line}")

        # Flush remaining buffer content
        for fd, buf in buffers.items():
            if buf.strip():
                channel = "out" if fd == process.stdout.fileno() else "err"
                print(f"\nFFmpeg ({channel}): {buf.strip()}")

        ret_code = process.wait()
        if ret_code != 0:
            raise RuntimeError(f"FFmpeg exited with status {ret_code}")

        print(f"\nComposited video saved to: {output_path}")

    except Exception as exc:
        raise Exception(f"Unexpected error during video composition: {exc}") from exc

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract metadata from an MP4 video file and GPS data')
    parser.add_argument('--video-file', required=True, help='Path to the MP4 video file')
    parser.add_argument('--gps-file', required=True, help='Path to the GPX file')
    parser.add_argument('--output-dir', required=True, help='Directory to save generated files')
    parser.add_argument('--skip-generation', action='store_true', default=False, help='Skip generation of overlay video')
    parser.add_argument('--max-duration', type=float, help='Maximum duration of the output video in seconds')
    parser.add_argument('--offset-seconds', type=float, default=0, help='Offset in seconds for the overlay layer (positive = delay, negative = advance)')
    # Parse arguments
    args = parser.parse_args()
    
    try:
        metadata = extract_video_metadata(args.video_file)
        track_points = parse_gpx_file(args.gps_file)
        
        map_size = (800, 600)
        overlay_video_path = os.path.join(args.output_dir, "overlay.mov")

        # Generate overlay video
        if not args.skip_generation:
            generate_map_video(
                video_metadata=metadata,
                track_points=track_points,
                output_path=overlay_video_path,
                map_size=map_size
            )
        
        # Composite the video with overlay
        output_video = os.path.join(args.output_dir, "output_with_map.mp4")
        composite_video_with_map(
            metadata=metadata,
            video_path=args.video_file,
            overlay_video_path=overlay_video_path,
            output_path=output_video,
            max_duration=args.max_duration,
            offset_seconds=args.offset_seconds
        )
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
