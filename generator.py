#!/usr/bin/env python3

import ffmpeg
import argparse
from dataclasses import dataclass

@dataclass
class VideoMetadata:
    """Class representing video metadata."""
    width: int
    height: int
    fps: float
    duration: float
    bitrate: int

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
    """
    try:
        # Get video stream information
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        if video_stream is None:
            raise ValueError("No video stream found in the file")
        
        # Create and return VideoMetadata object
        return VideoMetadata(
            width=int(video_stream['width']),
            height=int(video_stream['height']),
            fps=eval(video_stream['r_frame_rate']),  # Convert fraction string to float
            duration=float(probe['format']['duration']),
            bitrate=int(probe['format']['bit_rate'])
        )
    
    except ffmpeg.Error as e:
        raise Exception(f"Error processing video file: {str(e)}")
    except Exception as e:
        raise Exception(f"Unexpected error: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract metadata from an MP4 video file')
    parser.add_argument('--video-path', required=True, help='Path to the MP4 video file')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        metadata = extract_video_metadata(args.video_path)
        print("Video Metadata:")
        print(f"Resolution: {metadata.width}x{metadata.height}")
        print(f"Frame Rate: {metadata.fps} fps")
        print(f"Duration: {metadata.duration:.2f} seconds")
        print(f"Bitrate: {metadata.bitrate / 1000:.2f} kbps")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
