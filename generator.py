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
from gps_track_point import GPSTrackPoint
from overlay import generate_map_video, generate_heightmap_video, generate_speedometer_video

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
class OverlaySettings:
    video_path: str
    x: int
    y: int

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



# ---------------------------------------------------------------------------
# Video overlay generation (transparent MOV)                                |
# ---------------------------------------------------------------------------

def composite_video_with_overlays(
    video_path: str,
    overlay_settings: List[OverlaySettings],
    output_path: str,
    max_duration: float = None,
    offset_seconds: float = 0.0) -> None:
    """
    Composite the original video with multiple overlay videos using FFmpeg.

    The function will place overlay videos on top of the original clip at their
    specified positions while preserving their alpha channels.

    Parameters
    ----------
    video_path : str
        Path to the original/base video.
    overlay_settings : List[OverlaySettings]
        List of overlay settings, each containing video path and positioning info.
    output_path : str
        Path where the composited file will be written.
    max_duration : float | None, optional
        If provided the output will be trimmed/clamped to this duration (in
        seconds).
    offset_seconds : float, default 0.0
        Positive values delay the overlays relative to the base video, negative
        values advance them.  Implemented via a `setpts` filter on the overlay
        streams.
    """

    # Build FFmpeg input streams
    try:
        # Base/original clip (keep audio if present).
        base_in = ffmpeg.input(video_path)
        
        # Start with the base video
        current_video = base_in.video
        
        # Process each overlay
        for overlay_setting in overlay_settings:
            # Load overlay video
            overlay_in = ffmpeg.input(overlay_setting.video_path)
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

            # Apply overlay at the specified position
            current_video = ffmpeg.overlay(
                current_video,
                overlay_video,
                x=overlay_setting.x,
                y=overlay_setting.y,
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
                    current_video,
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
                    current_video,
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
    parser.add_argument('--tile-style', default='none', 
                       choices=['none', 'osm', 'cyclosm', 'humanitarian', 'osmfr', 'opentopomap', 'stamen-toner', 'stamen-watercolor'],
                       help='Map tile style to use (default: none)')
    
    args = parser.parse_args()
    
    try:
        track_points = parse_gpx_file(args.gps_file)
        metadata = extract_video_metadata(args.video_file)
        
        map_size = (800, 600)
        heightmap_size = (800, 400)
        speedometer_size = (400, 400)
        map_overlay_video_path = os.path.join(args.output_dir, "map.mov")
        heightmap_overlay_video_path = os.path.join(args.output_dir, "heightmap.mov")
        speedometer_overlay_video_path = os.path.join(args.output_dir, "speedometer.mov")

        # Prepare overlay settings list
        overlay_settings_list = []
        
        map_overlay_settings = OverlaySettings(
            video_path=map_overlay_video_path,
            x=(metadata.width - map_size[0]) // 2,  # Center horizontally
            y=metadata.height - map_size[1] - 10,  # Keep at bottom
        )
        
        heightmap_overlay_settings = OverlaySettings(
            video_path=heightmap_overlay_video_path,
            x=metadata.width - heightmap_size[0] - 10,  # Keep at right
            y=metadata.height - heightmap_size[1] - 10,  # Keep at bottom
        )

        speedometer_overlay_settings = OverlaySettings(
            video_path=speedometer_overlay_video_path,
            x=10,  # Move to left
            y=metadata.height - speedometer_size[1] - 10,  # Move to bottom
        )

        overlay_settings_list.append(map_overlay_settings)
        overlay_settings_list.append(heightmap_overlay_settings)
        overlay_settings_list.append(speedometer_overlay_settings)

        # Generate overlay videos
        if not args.skip_generation:
            # Generate map overlay
            generate_map_video(
                track_points=track_points,
                output_path=map_overlay_video_path,
                map_size=map_size,
                overlay_fps=args.overlay_fps,
                speed_window=args.speed_window,
                tile_style=args.tile_style,
            )
            
            # Generate heightmap overlay if enabled
            generate_heightmap_video(
                track_points=track_points,
                output_path=heightmap_overlay_video_path,
                chart_size=heightmap_size,
                overlay_fps=args.overlay_fps,
            )

            # Generate speedometer overlay
            generate_speedometer_video(
                track_points=track_points,
                output_path=speedometer_overlay_video_path,
                size=speedometer_size,
                overlay_fps=args.overlay_fps,
                speed_window=args.speed_window,
            )
        
        # Composite the video with overlays
        output_video = os.path.join(args.output_dir, "output.mp4")
        composite_video_with_overlays(
            video_path=args.video_file,
            overlay_settings=overlay_settings_list,
            output_path=output_video,
            max_duration=args.max_duration,
            offset_seconds=args.offset_seconds
        )
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
