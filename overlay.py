from gps_track_point import GPSTrackPoint
from typing import List
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import ffmpeg
import bisect
import cairo
import numpy as np
from map_tiles import MapTileProvider
import traceback

def interpolate_gps_point(
    point1: GPSTrackPoint,
    point2: GPSTrackPoint,
    target_time: datetime
) -> GPSTrackPoint:
    """
    Interpolate GPS data between two track points based on a target timestamp.
    
    Args:
        point1: First GPS track point
        point2: Second GPS track point
        target_time: Target timestamp to interpolate to
        
    Returns:
        GPSTrackPoint: Interpolated GPS track point
    """
    # Calculate the interpolation factor (0.0 to 1.0)
    time_diff_total = (point2.timestamp - point1.timestamp).total_seconds()
    time_diff_target = (target_time - point1.timestamp).total_seconds()
    
    # Handle edge cases
    if time_diff_total == 0:
        return point1
    
    factor = time_diff_target / time_diff_total
    factor = max(0.0, min(1.0, factor))  # Clamp to [0, 1]
    
    # Interpolate latitude, longitude, and elevation
    interpolated_lat = point1.latitude + (point2.latitude - point1.latitude) * factor
    interpolated_lon = point1.longitude + (point2.longitude - point1.longitude) * factor
    interpolated_elev = point1.elevation + (point2.elevation - point1.elevation) * factor
    
    # For non-numeric fields, use the value from the closest point
    if factor < 0.5:
        fix_type = point1.fix_type
        pdop = point1.pdop
    else:
        fix_type = point2.fix_type
        pdop = point2.pdop
    
    return GPSTrackPoint(
        latitude=interpolated_lat,
        longitude=interpolated_lon,
        elevation=interpolated_elev,
        timestamp=target_time,
        fix_type=fix_type,
        pdop=pdop
    )

def get_interpolated_gps_point(
    all_points: List[GPSTrackPoint],
    frame_time: datetime,
) -> GPSTrackPoint:
    """
    Get the interpolated GPS point for a given frame time.
    """
    if len(all_points) < 2:
        return all_points[0]
    
    timestamps = [p.timestamp for p in all_points]
    idx = bisect.bisect_left(timestamps, frame_time)

    if idx == 0:
        return all_points[0]
    elif idx >= len(all_points):
        return all_points[-1]
    else:
        return interpolate_gps_point(all_points[idx - 1], all_points[idx], frame_time)


def calculate_speed_interpolated(
    interpolated_point: GPSTrackPoint,
    all_points: List[GPSTrackPoint],
    window_seconds: float = 5.0,
) -> float:
    """
    Calculate average speed (km/h) over a window period using an interpolated position.
    
    Args:
        interpolated_point: The interpolated GPS point at current time
        all_points: List of all GPS track points
        window_seconds: Time window for speed calculation
        
    Returns:
        float: Speed in km/h
    """
    if len(all_points) < 2 or window_seconds <= 0:
        return 0.0
    
    # Find the points within the time window
    target_time = interpolated_point.timestamp
    start_time = target_time - timedelta(seconds=window_seconds)
    
    # Find all points within the window
    points_in_window = []
    for point in all_points:
        if start_time <= point.timestamp <= target_time:
            points_in_window.append(point)
    
    # Add the start point if we need to interpolate at the beginning of the window
    if points_in_window and points_in_window[0].timestamp > start_time:
        # Find the point just before the window
        for i, point in enumerate(all_points):
            if point.timestamp > start_time:
                if i > 0:
                    # Interpolate a point at the exact start of the window
                    start_point = interpolate_gps_point(
                        all_points[i-1], 
                        all_points[i], 
                        start_time
                    )
                    points_in_window.insert(0, start_point)
                break
    
    # Add the interpolated point at the end
    points_in_window.append(interpolated_point)
    
    if len(points_in_window) < 2:
        return 0.0
    
    # Calculate total distance traveled
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # km
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
        return 2 * R * atan2(sqrt(a), sqrt(1 - a))
    
    distance_km = 0.0
    for i in range(len(points_in_window) - 1):
        p1, p2 = points_in_window[i], points_in_window[i + 1]
        distance_km += haversine(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
    
    # Calculate actual time elapsed
    time_diff_sec = (points_in_window[-1].timestamp - points_in_window[0].timestamp).total_seconds()
    if time_diff_sec == 0:
        return 0.0
    
    speed_kmh = distance_km / (time_diff_sec / 3600.0)
    return speed_kmh

def generate_map_video(
    track_points: List[GPSTrackPoint],
    output_path: str,
    map_size: tuple = (800, 600),
    overlay_fps: float = 5.0,
    speed_window: float = 5.0,
    tile_style: str = "none",
) -> None:
    """Generate overlay using Cairo for advanced graphics with optional map background."""
    
    frame_duration = 1.0 / overlay_fps
    
    # Calculate bounds
    min_lat = min(p.latitude for p in track_points)
    max_lat = max(p.latitude for p in track_points)
    min_lon = min(p.longitude for p in track_points)
    max_lon = max(p.longitude for p in track_points)
    
    # Add padding
    lat_padding = (max_lat - min_lat) * 0.1
    lon_padding = (max_lon - min_lon) * 0.1
    min_lat -= lat_padding
    max_lat += lat_padding
    min_lon -= lon_padding
    max_lon += lon_padding
    
    # Download map tiles if requested
    map_image = None
    if tile_style != "none":
        print("Downloading map tiles...")
        tile_provider = MapTileProvider(style=tile_style)
        try:
            pil_map, map_info = tile_provider.create_map_image(
                min_lat, max_lat, min_lon, max_lon,
                map_size[0], map_size[1]
            )
            # Convert PIL image to numpy array for Cairo
            map_array = np.array(pil_map)
            # Add alpha channel if not present
            if map_array.shape[2] == 3:
                map_array = np.dstack([map_array, np.full((map_size[1], map_size[0]), 255, dtype=np.uint8)])
            # Convert RGB to BGR for Cairo (BGRA format)
            map_array = map_array[:, :, [2, 1, 0, 3]]
            # Ensure the array is C-contiguous
            map_image = np.ascontiguousarray(map_array)
            print("Map tiles downloaded successfully")
        except Exception as e:
            print(f"Warning: Could not download map tiles: {e}")
            print(traceback.format_exc())
            print("Continuing without map background")
    
    gps_duration = (track_points[-1].timestamp - track_points[0].timestamp).total_seconds()
    total_frames = int(gps_duration * overlay_fps + 0.5)
    
    # Start FFmpeg process
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgba', 
               s=f'{map_size[0]}x{map_size[1]}', r=overlay_fps)
        .output(output_path, vcodec='prores_ks', profile='4444', 
                pix_fmt='yuva444p10le', r=overlay_fps)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    
    try:
        # Create a persistent map surface if we have map tiles
        map_surface = None
        if map_image is not None:
            # Create the map surface once, outside the loop
            map_surface = cairo.ImageSurface.create_for_data(
                map_image, cairo.FORMAT_ARGB32, 
                map_size[0], map_size[1],
                map_size[0] * 4  # stride = width * 4 bytes per pixel
            )
        
        for frame_num in range(total_frames):
            frame_time = track_points[0].timestamp + timedelta(seconds=frame_num * frame_duration)
            interpolated_point = get_interpolated_gps_point(track_points, frame_time)
            
            # Create Cairo surface
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, map_size[0], map_size[1])
            ctx = cairo.Context(surface)
            
            # Draw map background if available
            if map_surface is not None:
                ctx.set_source_surface(map_surface, 0, 0)
                ctx.paint()
                
                # Add semi-transparent overlay to make the track more visible
                ctx.set_source_rgba(0, 0, 0, 0.2)
                ctx.paint()
            else:
                # Clear with transparent background
                ctx.set_source_rgba(0, 0, 0, 0)
                ctx.paint()
            
            def lat_lon_to_pixel(lat: float, lon: float) -> tuple:
                x = (lon - min_lon) / (max_lon - min_lon) * map_size[0]
                y = (1 - (lat - min_lat) / (max_lat - min_lat)) * map_size[1]
                return (x, y)
            
            # Draw track with enhanced visibility
            ctx.set_line_width(4)
            
            # Draw white outline first
            ctx.set_source_rgba(1, 1, 1, 0.8)
            for i, point in enumerate(track_points):
                x, y = lat_lon_to_pixel(point.latitude, point.longitude)
                if i == 0:
                    ctx.move_to(x, y)
                else:
                    ctx.line_to(x, y)
            ctx.stroke()
            
            # Draw colored track on top
            ctx.set_line_width(3)
            ctx.set_source_rgb(0.2, 0.4, 1.0)
            for i, point in enumerate(track_points):
                x, y = lat_lon_to_pixel(point.latitude, point.longitude)
                if i == 0:
                    ctx.move_to(x, y)
                else:
                    ctx.line_to(x, y)
            ctx.stroke()
            
            # Draw current position with glow effect
            current_pos = lat_lon_to_pixel(interpolated_point.latitude, interpolated_point.longitude)
            
            # Glow effect
            for radius in [20, 15, 10]:
                ctx.arc(current_pos[0], current_pos[1], radius, 0, 2 * 3.14159)
                alpha = 0.1 if radius == 20 else 0.2 if radius == 15 else 0.4
                ctx.set_source_rgba(1, 0, 0, alpha)
                ctx.fill()
            
            # Main dot
            ctx.arc(current_pos[0], current_pos[1], 6, 0, 2 * 3.14159)
            ctx.set_source_rgba(1, 1, 1, 1)
            ctx.fill()
            
            # Add text with background for better readability
            ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(20)
            
            info_text = [
                f"Lat: {interpolated_point.latitude:.6f}°",
                f"Lon: {interpolated_point.longitude:.6f}°",
                f"Elev: {interpolated_point.elevation:.1f}m",
                f"Speed: {calculate_speed_interpolated(interpolated_point, track_points, speed_window):.1f} km/h",
                f"Time: {interpolated_point.timestamp.strftime('%H:%M:%S')}"
            ]
            
            # Draw background box for text
            ctx.set_source_rgba(0, 0, 0, 0.7)
            ctx.rectangle(5, 5, 350, 140)
            ctx.fill()
            
            for i, text in enumerate(info_text):
                y_pos = 30 + i * 25
                # Main text
                ctx.move_to(12, y_pos)
                ctx.set_source_rgba(1, 1, 1, 1)
                ctx.show_text(text)
            
            # Convert Cairo surface to bytes
            buf = surface.get_data()
            img_array = np.ndarray(shape=(map_size[1], map_size[0], 4), 
                                 dtype=np.uint8, buffer=buf)
            # Cairo uses BGRA, convert to RGBA
            img_array = img_array[:, :, [2, 1, 0, 3]]
            
            # Write frame
            process.stdin.write(img_array.tobytes())
            
            progress = (frame_num + 1) / total_frames * 100
            print(f"\rGenerating overlay video: {progress:.1f}%", end='', flush=True)
        
        print()
        
    except Exception as e:
        raise Exception(f"Error generating overlay video: {str(e)}")
    finally:
        process.stdin.close()
        process.wait()

def generate_heightmap_video(
    track_points: List[GPSTrackPoint],
    output_path: str,
    chart_size: tuple = (800, 400),
    overlay_fps: float = 5.0,
) -> None:
    """Generate elevation profile overlay showing height vs time with current position indicator."""
    
    frame_duration = 1.0 / overlay_fps
    
    # Calculate elevation bounds
    elevations = [p.elevation for p in track_points if p.elevation is not None]
    if not elevations:
        raise ValueError("No elevation data found in track points")
    
    min_elevation = min(elevations)
    max_elevation = max(elevations)
    
    # Add padding to elevation range
    elevation_range = max_elevation - min_elevation
    if elevation_range == 0:
        elevation_range = 100  # Default range if all elevations are the same
    elevation_padding = elevation_range * 0.1
    min_elevation -= elevation_padding
    max_elevation += elevation_padding
    
    # Calculate time bounds
    start_time = track_points[0].timestamp
    end_time = track_points[-1].timestamp
    total_duration = (end_time - start_time).total_seconds()
    
    gps_duration = (track_points[-1].timestamp - track_points[0].timestamp).total_seconds()
    total_frames = int(gps_duration * overlay_fps + 0.5)
    
    # Chart margins
    margin_left = 80
    margin_right = 40
    margin_top = 40
    margin_bottom = 60
    chart_width = chart_size[0] - margin_left - margin_right
    chart_height = chart_size[1] - margin_top - margin_bottom
    
    # Start FFmpeg process
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgba', 
               s=f'{chart_size[0]}x{chart_size[1]}', r=overlay_fps)
        .output(output_path, vcodec='prores_ks', profile='4444', 
                pix_fmt='yuva444p10le', r=overlay_fps)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    
    try:
        for frame_num in range(total_frames):
            frame_time = track_points[0].timestamp + timedelta(seconds=frame_num * frame_duration)
            interpolated_point = get_interpolated_gps_point(track_points, frame_time)
            
            # Create Cairo surface
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, chart_size[0], chart_size[1])
            ctx = cairo.Context(surface)
            
            # Clear with semi-transparent background
            ctx.set_source_rgba(0, 0, 0, 0.8)
            ctx.paint()
            
            # Helper functions for coordinate conversion
            def time_to_x(timestamp: datetime) -> float:
                time_offset = (timestamp - start_time).total_seconds()
                return margin_left + (time_offset / total_duration) * chart_width
            
            def elevation_to_y(elevation: float) -> float:
                normalized = (elevation - min_elevation) / (max_elevation - min_elevation)
                return margin_top + chart_height - (normalized * chart_height)
            
            # Draw grid lines
            ctx.set_line_width(1)
            ctx.set_source_rgba(0.3, 0.3, 0.3, 0.8)
            
            # Vertical grid lines (time)
            for i in range(6):  # 5 intervals
                x = margin_left + (i / 5) * chart_width
                ctx.move_to(x, margin_top)
                ctx.line_to(x, margin_top + chart_height)
                ctx.stroke()
            
            # Horizontal grid lines (elevation)
            for i in range(6):  # 5 intervals
                y = margin_top + (i / 5) * chart_height
                ctx.move_to(margin_left, y)
                ctx.line_to(margin_left + chart_width, y)
                ctx.stroke()
            
            # Draw elevation profile
            ctx.set_line_width(3)
            ctx.set_source_rgba(0.2, 0.8, 0.2, 1.0)  # Green line
            
            for i, point in enumerate(track_points):
                if point.elevation is None:
                    continue
                    
                x = time_to_x(point.timestamp)
                y = elevation_to_y(point.elevation)
                
                if i == 0:
                    ctx.move_to(x, y)
                else:
                    ctx.line_to(x, y)
            ctx.stroke()
            
            # Draw current position indicator
            current_x = time_to_x(interpolated_point.timestamp)
            current_y = elevation_to_y(interpolated_point.elevation)
            
            # Vertical line at current time
            ctx.set_line_width(2)
            ctx.set_source_rgba(1, 0, 0, 0.8)  # Red line
            ctx.move_to(current_x, margin_top)
            ctx.line_to(current_x, margin_top + chart_height)
            ctx.stroke()
            
            # Current position dot
            ctx.arc(current_x, current_y, 6, 0, 2 * 3.14159)
            ctx.set_source_rgba(1, 0, 0, 1)  # Red dot
            ctx.fill()
            
            # White outline for dot
            ctx.arc(current_x, current_y, 6, 0, 2 * 3.14159)
            ctx.set_source_rgba(1, 1, 1, 1)
            ctx.set_line_width(2)
            ctx.stroke()
            
            # Draw axes
            ctx.set_line_width(2)
            ctx.set_source_rgba(1, 1, 1, 1)
            
            # Y-axis
            ctx.move_to(margin_left, margin_top)
            ctx.line_to(margin_left, margin_top + chart_height)
            ctx.stroke()
            
            # X-axis
            ctx.move_to(margin_left, margin_top + chart_height)
            ctx.line_to(margin_left + chart_width, margin_top + chart_height)
            ctx.stroke()
            
            # Add labels
            ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(12)
            ctx.set_source_rgba(1, 1, 1, 1)
            
            # Y-axis labels (elevation)
            for i in range(6):
                elevation = min_elevation + (i / 5) * (max_elevation - min_elevation)
                y = margin_top + chart_height - (i / 5) * chart_height
                label = f"{elevation:.0f}m"
                
                text_extents = ctx.text_extents(label)
                ctx.move_to(margin_left - text_extents.width - 10, y + text_extents.height / 2)
                ctx.show_text(label)
            
            # X-axis labels (time)
            for i in range(6):
                time_offset = (i / 5) * total_duration
                timestamp = start_time + timedelta(seconds=time_offset)
                x = margin_left + (i / 5) * chart_width
                label = timestamp.strftime('%H:%M')
                
                text_extents = ctx.text_extents(label)
                ctx.move_to(x - text_extents.width / 2, margin_top + chart_height + 20)
                ctx.show_text(label)
            
            # Current info
            
            # Current elevation info
            ctx.set_font_size(14)
            current_info = f"Current: {interpolated_point.elevation:.1f}m at {interpolated_point.timestamp.strftime('%H:%M:%S')}"
            text_extents = ctx.text_extents(current_info)
            ctx.move_to(chart_size[0] - text_extents.width - 10, chart_size[1] - 10)
            ctx.show_text(current_info)
            
            # Convert Cairo surface to bytes
            buf = surface.get_data()
            img_array = np.ndarray(shape=(chart_size[1], chart_size[0], 4), 
                                 dtype=np.uint8, buffer=buf)
            # Cairo uses BGRA, convert to RGBA
            img_array = img_array[:, :, [2, 1, 0, 3]]
            
            # Write frame
            process.stdin.write(img_array.tobytes())
            
            progress = (frame_num + 1) / total_frames * 100
            print(f"\rGenerating heightmap video: {progress:.1f}%", end='', flush=True)
        
        print()
        
    except Exception as e:
        raise Exception(f"Error generating heightmap video: {str(e)}")
    finally:
        process.stdin.close()
        process.wait()
