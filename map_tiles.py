import math
import requests
import os
from PIL import Image
from typing import Tuple, List
import time

class MapTileProvider:
    """Handle downloading and caching of map tiles from various providers."""
    
    # Available tile styles
    TILE_STYLES = {
        'osm': {
            'name': 'OpenStreetMap Standard',
            'url': 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
            'attribution': '© OpenStreetMap contributors',
            'max_zoom': 19
        },
        'cyclosm': {
            'name': 'CyclOSM (Cycling)',
            'url': 'https://a.tile-cyclosm.openstreetmap.fr/cyclosm/{z}/{x}/{y}.png',
            'attribution': '© CyclOSM, © OpenStreetMap contributors',
            'max_zoom': 20
        },
        'humanitarian': {
            'name': 'Humanitarian',
            'url': 'https://a.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png',
            'attribution': '© Humanitarian OpenStreetMap Team, © OpenStreetMap contributors',
            'max_zoom': 20
        },
        'osmfr': {
            'name': 'OSM France',
            'url': 'https://a.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png',
            'attribution': '© OpenStreetMap France, © OpenStreetMap contributors',
            'max_zoom': 20
        },
        'opentopomap': {
            'name': 'OpenTopoMap (Topographic)',
            'url': 'https://a.tile.opentopomap.org/{z}/{x}/{y}.png',
            'attribution': '© OpenTopoMap (CC-BY-SA), © OpenStreetMap contributors',
            'max_zoom': 17
        },
        'stamen-toner': {
            'name': 'Stamen Toner (Black & White)',
            'url': 'https://tiles.stadiamaps.com/tiles/stamen_toner/{z}/{x}/{y}.png',
            'attribution': '© Stamen Design, © Stadia Maps, © OpenStreetMap contributors',
            'max_zoom': 20
        },
        'stamen-watercolor': {
            'name': 'Stamen Watercolor (Artistic)',
            'url': 'https://tiles.stadiamaps.com/tiles/stamen_watercolor/{z}/{x}/{y}.jpg',
            'attribution': '© Stamen Design, © Stadia Maps, © OpenStreetMap contributors',
            'max_zoom': 18
        }
    }
    
    def __init__(self, cache_dir: str = "map_cache", style: str = "osm"):
        self.cache_dir = cache_dir
        self.tile_size = 256
        self.style = style
        
        if style not in self.TILE_STYLES:
            print(f"Warning: Unknown style '{style}', using 'osm' instead")
            self.style = "osm"
        
        self.style_config = self.TILE_STYLES[self.style]
        self.base_url = self.style_config['url']
        self.max_zoom = self.style_config['max_zoom']
        
        self.headers = {
            'User-Agent': 'GPS-Video-Overlay/1.0'  # Required by tile usage policies
        }
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"Using tile style: {self.style_config['name']}")
        print(f"Attribution: {self.style_config['attribution']}")
    
    def lat_lon_to_tile(self, lat: float, lon: float, zoom: int) -> Tuple[int, int]:
        """Convert latitude/longitude to tile coordinates."""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (x, y)
    
    def tile_to_lat_lon(self, x: int, y: int, zoom: int) -> Tuple[float, float]:
        """Convert tile coordinates to latitude/longitude (NW corner)."""
        n = 2.0 ** zoom
        lon = x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat = math.degrees(lat_rad)
        return (lat, lon)
    
    def get_tile(self, x: int, y: int, zoom: int) -> Image.Image:
        """Download a tile or retrieve from cache."""
        # Clamp zoom to max supported level
        zoom = min(zoom, self.max_zoom)
        
        cache_path = os.path.join(self.cache_dir, f"{self.style}_{zoom}_{x}_{y}.png")
        
        if os.path.exists(cache_path):
            return Image.open(cache_path)
        
        # Download tile
        url = self.base_url.format(z=zoom, x=x, y=y)
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            
            # Be respectful to tile servers
            time.sleep(0.1)
            
            return Image.open(cache_path)
        except Exception as e:
            print(f"Error downloading tile {x},{y} at zoom {zoom}: {e}")
            # Return a blank tile
            return Image.new('RGB', (self.tile_size, self.tile_size), (200, 200, 200))
    
    def calculate_zoom_level(self, min_lat: float, max_lat: float, 
                           min_lon: float, max_lon: float, 
                           map_width: int, map_height: int) -> int:
        """Calculate appropriate zoom level for the given bounds and map size."""
        # Calculate zoom level based on the larger dimension
        max_zoom_to_try = min(18, self.max_zoom)
        for zoom in range(max_zoom_to_try, 0, -1):
            # Get tile bounds at this zoom
            min_x, max_y = self.lat_lon_to_tile(max_lat, min_lon, zoom)
            max_x, min_y = self.lat_lon_to_tile(min_lat, max_lon, zoom)
            
            tiles_x = max_x - min_x + 1
            tiles_y = max_y - min_y + 1
            
            pixel_width = tiles_x * self.tile_size
            pixel_height = tiles_y * self.tile_size
            
            if pixel_width <= map_width * 1.5 and pixel_height <= map_height * 1.5:
                return zoom
        
        return 10  # Default zoom level
    
    def create_map_image(self, min_lat: float, max_lat: float,
                        min_lon: float, max_lon: float,
                        map_width: int, map_height: int) -> Tuple[Image.Image, dict]:
        """Create a map image for the given bounds."""
        zoom = self.calculate_zoom_level(min_lat, max_lat, min_lon, max_lon, 
                                       map_width, map_height)
        
        # Get tile bounds
        # Note: In tile coordinates, y increases from north to south
        # So max_lat corresponds to smaller y values
        top_left_x, top_left_y = self.lat_lon_to_tile(max_lat, min_lon, zoom)
        bottom_right_x, bottom_right_y = self.lat_lon_to_tile(min_lat, max_lon, zoom)
        
        # Ensure coordinates are properly ordered
        min_x = min(top_left_x, bottom_right_x)
        max_x = max(top_left_x, bottom_right_x)
        min_y = min(top_left_y, bottom_right_y)
        max_y = max(top_left_y, bottom_right_y)
        
        # Create composite image
        tiles_x = max_x - min_x + 1
        tiles_y = max_y - min_y + 1
        
        # Ensure we have at least 1 tile in each dimension
        tiles_x = max(1, tiles_x)
        tiles_y = max(1, tiles_y)
        
        composite = Image.new('RGB', (tiles_x * self.tile_size, tiles_y * self.tile_size))
        
        # Download and composite tiles
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                tile = self.get_tile(x, y, zoom)
                composite.paste(tile, ((x - min_x) * self.tile_size, 
                                     (y - min_y) * self.tile_size))
        
        # Calculate bounds of the tile grid
        nw_lat, nw_lon = self.tile_to_lat_lon(min_x, min_y, zoom)
        se_lat, se_lon = self.tile_to_lat_lon(max_x + 1, max_y + 1, zoom)
        
        # Crop to exact bounds and resize to target size
        # Calculate pixel positions for the requested bounds
        tile_lat_range = nw_lat - se_lat
        tile_lon_range = se_lon - nw_lon
        
        left = int((min_lon - nw_lon) / tile_lon_range * composite.width)
        top = int((nw_lat - max_lat) / tile_lat_range * composite.height)
        right = int((max_lon - nw_lon) / tile_lon_range * composite.width)
        bottom = int((nw_lat - min_lat) / tile_lat_range * composite.height)
        
        cropped = composite.crop((left, top, right, bottom))
        final_map = cropped.resize((map_width, map_height), Image.Resampling.LANCZOS)
        
        # Return map info for coordinate conversion
        map_info = {
            'min_lat': min_lat,
            'max_lat': max_lat,
            'min_lon': min_lon,
            'max_lon': max_lon,
            'width': map_width,
            'height': map_height,
            'zoom': zoom
        }
        
        return final_map, map_info 