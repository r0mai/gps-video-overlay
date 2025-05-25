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


def calculate_speed(
    closes_idx: int,
    all_points: List[GPSTrackPoint],
    window_seconds: float = 5.0,
) -> float:
    """Return average speed (km/h) over a *window_seconds* look-back period.

    The function walks backwards from *closes_idx* until the accumulated
    timespan reaches or exceeds *window_seconds*, then computes the total
    travelled distance across those segments divided by the actual elapsed
    time.

    Parameters
    ----------
    closes_idx : int
        Index of the track-point considered the *current* position.
    all_points : list[GPSTrackPoint]
        The full ordered list of track points.
    window_seconds : float, default 5.0
        Duration (in seconds) over which to average the speed.
    """

    if closes_idx == 0 or window_seconds <= 0:
        return 0.0

    # Identify the earliest point that is still within the window.
    current_point = all_points[closes_idx]
    start_idx = closes_idx

    while (
        start_idx > 0 and
        (current_point.timestamp - all_points[start_idx - 1].timestamp).total_seconds() <= window_seconds
    ):
        start_idx -= 1

    # If no movement (timestamps equal), bail out.
    time_diff_sec = (current_point.timestamp - all_points[start_idx].timestamp).total_seconds()
    if time_diff_sec == 0:
        return 0.0

    # Sum segment distances between consecutive points from start_idx to closes_idx.
    from math import radians, sin, cos, sqrt, atan2

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # km
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))

    distance_km = 0.0
    for i in range(start_idx, closes_idx):
        p1, p2 = all_points[i], all_points[i + 1]
        distance_km += haversine(p1.latitude, p1.longitude, p2.latitude, p2.longitude)

    speed_kmh = distance_km / (time_diff_sec / 3600.0)
    return speed_kmh

# ---------------------------------------------------------------------------
# Video overlay generation (transparent MOV)                                |
# ---------------------------------------------------------------------------

def generate_map_video(
    track_points: List[GPSTrackPoint],
    output_path: str,
    map_size: tuple = (800, 600),
    overlay_fps: float = 5.0,
    speed_window: float = 5.0,
) -> None:
    """
    Generate a transparent video with GPS path visualization.
    
    Args:
        video_metadata: Video metadata containing resolution and fps
        track_points: List of GPS track points
        output_path: Path to save the generated video
        map_size: Size of the visualization in pixels (default: 800x600)
        overlay_fps: Frame rate (frames per second) for the generated overlay video
        speed_window: Window duration in seconds for calculating speed
    """
    # Calculate frame duration based on *overlay* fps (not the base video fps)
    if overlay_fps <= 0:
        raise ValueError("overlay_fps must be greater than zero")

    frame_duration = 1.0 / overlay_fps
    
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

    # Calculate the duration of the GPS telemetry
    if len(track_points) < 2:
        raise ValueError("GPS track must have at least 2 points")
    
    gps_duration = (track_points[-1].timestamp - track_points[0].timestamp).total_seconds()
    
    # Determine how many frames to generate so that the overlay spans the full
    # duration of the GPS telemetry (rounded up so we cover the tail).
    total_frames = int(gps_duration * overlay_fps + 0.5)

    # Start FFmpeg process to encode the overlay video at the requested fps
    process = (
        ffmpeg
        .input(
            'pipe:',
            format='rawvideo',
            pix_fmt='rgba',
            s=f'{map_size[0]}x{map_size[1]}',
            r=overlay_fps,
        )
        .output(
            output_path,
            vcodec='prores_ks',
            profile='4444',
            pix_fmt='yuva444p10le',
            r=overlay_fps,
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    try:
        # Generate frames sequentially and write them to the pipe
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
                f"Speed: {calculate_speed(closest_idx, track_points, window_seconds=speed_window):.1f} km/h",
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

def composite_video_with_overlay(metadata: VideoMetadata, video_path: str, overlay_video_path: str, output_path: str, max_duration: float = None, offset_seconds: float = 0.0) -> None:
    """
    Composite the original video with the overlay video using FFmpeg.

    The function will place the (usually smaller) overlay video on top of the
    bottom-left corner of the original clip while preserving its alpha channel.

    Parameters
    ----------
    metadata : VideoMetadata
        Metadata of the *original* video.  Currently only used for sanity
        checks; supplying it keeps the call-site symmetrical with the other
        helper functions.
    video_path : str
        Path to the original/base video.
    overlay_video_path : str
        Path to the previously generated transparent overlay video.
    output_path : str
        Path where the composited file will be written.
    max_duration : float | None, optional
        If provided the output will be trimmed/clamped to this duration (in
        seconds).
    offset_seconds : float, default 0.0
        Positive values delay the overlay relative to the base video, negative
        values advance it.  Implemented via a `setpts` filter on the overlay
        stream.
    """

    # Build FFmpeg input streams
    try:
        # Base/original clip (keep audio if present).
        base_in = ffmpeg.input(video_path)

        # Overlay clip – video only (usually has no audio).  If the caller
        # wants the overlay to start later/earlier we adjust its PTS.
        overlay_in = ffmpeg.input(overlay_video_path)

        overlay_video = overlay_in.video  # isolate the video stream

        # If an offset is requested we either:
        #  • Positive offset  -> freeze the first frame for `offset_seconds`
        #  • Negative offset  -> start the overlay earlier by shifting PTS
        # In both cases we keep the last frame visible afterwards by using
        # `eof_action=repeat` later in the overlay filter.
        if offset_seconds > 0:
            # Clone the first frame so that it is displayed until the overlay
            # actually starts moving.
            overlay_video = overlay_video.filter(
                "tpad",
                start_duration=offset_seconds,
                start_mode="clone",
            )
        elif offset_seconds < 0:
            # Advance the overlay timeline; frames that would fall before
            # t=0 are discarded automatically by FFmpeg.
            overlay_video = overlay_video.filter(
                "setpts",
                f"PTS+{offset_seconds}/TB",  # note: offset_seconds is negative here
            )

        # Position the overlay 10px from the left and 10px from the bottom.
        # `main_h-overlay_h-10` keeps the overlay anchored to the bottom.
        composited = ffmpeg.overlay(
            base_in.video,
            overlay_video,
            x=10,
            y=10,
            # y="main_h-overlay_h-10",
            format="auto",  # let FFmpeg choose the correct format (keeps alpha)
            eof_action="repeat",  # when overlay ends, repeat the last frame
        )

        # Detect whether the base video contains an audio stream – if not we
        # must *not* try to map it in the output, otherwise FFmpeg will error
        # out with "Stream map 'a:0' matches no streams".
        try:
            probe = ffmpeg.probe(video_path)
            has_audio = any(s.get("codec_type") == "audio" for s in probe["streams"])
        except ffmpeg.Error:
            # In the unlikely event probing fails, assume audio exists so we
            # at least attempt to preserve it.
            has_audio = True

        # Assemble output arguments.  We re-encode the video with H.264 for
        # broad compatibility and copy the original audio stream (if present).
        output_kwargs = {
            "vcodec": "libx264",
            "pix_fmt": "yuv420p",
            "preset": "veryfast",
            "crf": 23,
            "acodec": "copy",  # keep original audio untouched
            "movflags": "+faststart",
        }

        # Trim/limit duration if requested.
        if max_duration is not None and max_duration > 0:
            output_kwargs["t"] = max_duration

        # Build the final graph – video from the composited stream and, if
        # present, audio from the base input.
        if has_audio:
            out_stream = (
                ffmpeg
                .output(
                    composited,
                    base_in.audio,
                    output_path,
                    **output_kwargs,
                )
                .overwrite_output()
            )
        else:
            # No audio in source – only map the video stream.
            # Remove the `acodec` setting to avoid superfluous arguments.
            output_kwargs_no_audio = {k: v for k, v in output_kwargs.items() if k != "acodec"}

            out_stream = (
                ffmpeg
                .output(
                    composited,
                    output_path,
                    **output_kwargs_no_audio,
                )
                .overwrite_output()
            )

        # Execute FFmpeg and capture its output so that any error bubbles up.
        ffmpeg.run(out_stream);

    except ffmpeg.Error as e:
        # Decode the error for easier debugging before propagating upwards.
        err_msg = e.stderr.decode() if hasattr(e, "stderr") else str(e)
        raise RuntimeError(f"Error compositing video with overlay: {err_msg}") from e

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract metadata from an MP4 video file and GPS data')
    parser.add_argument('--video-file', required=True, help='Path to the MP4 video file')
    parser.add_argument('--gps-file', required=True, help='Path to the GPX file')
    parser.add_argument('--output-dir', required=True, help='Directory to save generated files')
    parser.add_argument('--skip-generation', action='store_true', default=False, help='Skip generation of overlay video')
    parser.add_argument('--max-duration', type=float, help='Maximum duration of the output video in seconds')
    parser.add_argument('--offset-seconds', type=float, default=0, help='Offset in seconds for the overlay layer (positive = delay, negative = advance)')
    parser.add_argument('--speed-window', type=float, default=5.0, help='Time window (in seconds) over which to average the displayed speed')
    parser.add_argument('--overlay-fps', type=float, default=5.0, help='Frame rate (frames per second) for the generated overlay video')
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
                track_points=track_points,
                output_path=overlay_video_path,
                map_size=map_size,
                overlay_fps=args.overlay_fps,
                speed_window=args.speed_window,
            )
        
        # Composite the video with overlay
        output_video = os.path.join(args.output_dir, "output_with_map.mp4")
        composite_video_with_overlay(
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
