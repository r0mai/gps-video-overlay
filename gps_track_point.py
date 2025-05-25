from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class GPSTrackPoint:
    """Class representing a GPS track point."""
    latitude: float
    longitude: float
    elevation: float
    timestamp: datetime
    fix_type: str
    pdop: float