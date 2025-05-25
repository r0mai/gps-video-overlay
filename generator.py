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
from overlay import generate_map_video

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
    parser.add_argument('--tile-style', default='none', 
                       choices=['none', 'osm', 'cyclosm', 'cyclosm-lite', 'humanitarian', 'osmfr', 'opentopomap', 'stamen-toner', 'stamen-watercolor'],
                       help='Map tile style to use (default: none)')
    
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
                tile_style=args.tile_style,
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
