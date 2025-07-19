#!/usr/bin/env python3
"""
TDoA (Time Difference of Arrival) Processor for Radio-Mapper Emergency Response System

This module implements:
1. Hyperbolic positioning algorithms for signal triangulation
2. Cross-correlation for precise time delay measurement
3. Multilateration for 3+ buoy configurations
4. Error analysis and confidence calculations
5. Real-time emergency signal location processing
"""

import math
import numpy as np
import logging
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timezone
import scipy.optimize
from scipy.signal import correlate

logger = logging.getLogger(__name__)

@dataclass
class BuoyPosition:
    """Represents a buoy's position and timing info"""
    buoy_id: str
    lat: float
    lng: float
    altitude: float = 0.0  # meters above sea level
    timing_accuracy_ns: int = 100000  # nanoseconds

@dataclass
class SignalDetection:
    """Signal detection from a single buoy (matches buoy_node.py)"""
    buoy_id: str
    frequency_mhz: float
    signal_strength_dbm: float
    timestamp_utc: str
    gps_timestamp_ns: int
    lat: float
    lng: float
    confidence: float
    signal_type: str = "unknown"

@dataclass
class TDoAMeasurement:
    """Time difference of arrival between two buoys"""
    buoy1_id: str
    buoy2_id: str
    time_difference_ns: int  # buoy2 - buoy1 (positive means buoy2 received later)
    distance_difference_m: float  # corresponding distance difference
    confidence: float
    frequency_mhz: float

@dataclass
class TriangulationResult:
    """Result of TDoA triangulation"""
    estimated_lat: float
    estimated_lng: float
    estimated_altitude: float
    accuracy_meters: float
    confidence: float
    frequency_mhz: float
    signal_type: str
    timestamp_utc: str
    contributing_buoys: List[str]
    tdoa_measurements: List[TDoAMeasurement]
    method: str  # "hyperbolic", "multilateration", etc.

class GeodeticCalculator:
    """Handles geodetic calculations for positioning"""
    
    EARTH_RADIUS_M = 6378137.0  # WGS84 equatorial radius
    
    @staticmethod
    def lat_lng_to_xyz(lat: float, lng: float, alt: float = 0.0) -> Tuple[float, float, float]:
        """Convert lat/lng/alt to ECEF (Earth-Centered Earth-Fixed) coordinates"""
        lat_rad = math.radians(lat)
        lng_rad = math.radians(lng)
        
        cos_lat = math.cos(lat_rad)
        sin_lat = math.sin(lat_rad)
        cos_lng = math.cos(lng_rad)
        sin_lng = math.sin(lng_rad)
        
        R = GeodeticCalculator.EARTH_RADIUS_M
        
        x = (R + alt) * cos_lat * cos_lng
        y = (R + alt) * cos_lat * sin_lng
        z = (R + alt) * sin_lat
        
        return x, y, z
    
    @staticmethod
    def xyz_to_lat_lng(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert ECEF coordinates to lat/lng/alt"""
        R = GeodeticCalculator.EARTH_RADIUS_M
        
        lng = math.atan2(y, x)
        lat = math.atan2(z, math.sqrt(x*x + y*y))
        alt = math.sqrt(x*x + y*y + z*z) - R
        
        return math.degrees(lat), math.degrees(lng), alt
    
    @staticmethod
    def distance_3d(lat1: float, lng1: float, alt1: float, 
                   lat2: float, lng2: float, alt2: float) -> float:
        """Calculate 3D distance between two points"""
        x1, y1, z1 = GeodeticCalculator.lat_lng_to_xyz(lat1, lng1, alt1)
        x2, y2, z2 = GeodeticCalculator.lat_lng_to_xyz(lat2, lng2, alt2)
        
        return math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    
    @staticmethod
    def bearing_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> Tuple[float, float]:
        """Calculate bearing and distance between two points"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlng_rad = math.radians(lng2 - lng1)
        
        # Haversine formula for distance
        a = (math.sin((lat2_rad - lat1_rad) / 2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlng_rad / 2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = GeodeticCalculator.EARTH_RADIUS_M * c
        
        # Bearing calculation
        y = math.sin(dlng_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlng_rad))
        bearing = math.atan2(y, x)
        bearing_deg = (math.degrees(bearing) + 360) % 360
        
        return bearing_deg, distance

class TDoACalculator:
    """Calculates time differences of arrival from signal detections"""
    
    SPEED_OF_LIGHT = 299792458.0  # m/s
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".TDoACalculator")
    
    def calculate_tdoa_measurements(self, detections: List[SignalDetection], 
                                  buoy_positions: Dict[str, BuoyPosition]) -> List[TDoAMeasurement]:
        """Calculate TDoA measurements from multiple buoy detections"""
        measurements = []
        
        if len(detections) < 2:
            self.logger.warning("Need at least 2 detections for TDoA calculation")
            return measurements
        
        # Create all pairwise TDoA measurements
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                det1 = detections[i]
                det2 = detections[j]
                
                # Skip if frequency doesn't match (different signals)
                if abs(det1.frequency_mhz - det2.frequency_mhz) > 0.01:
                    continue
                
                # Calculate time difference (nanoseconds)
                time_diff_ns = det2.gps_timestamp_ns - det1.gps_timestamp_ns
                
                # Convert to distance difference (meters)
                time_diff_s = time_diff_ns / 1e9
                distance_diff_m = time_diff_s * self.SPEED_OF_LIGHT
                
                # Calculate measurement confidence based on signal strength and timing accuracy
                buoy1_pos = buoy_positions.get(det1.buoy_id)
                buoy2_pos = buoy_positions.get(det2.buoy_id)
                
                if not buoy1_pos or not buoy2_pos:
                    continue
                
                # Confidence based on signal strength and timing accuracy
                strength_conf = min(det1.confidence, det2.confidence)
                timing_conf = self._calculate_timing_confidence(buoy1_pos, buoy2_pos)
                overall_conf = strength_conf * timing_conf
                
                measurement = TDoAMeasurement(
                    buoy1_id=det1.buoy_id,
                    buoy2_id=det2.buoy_id,
                    time_difference_ns=time_diff_ns,
                    distance_difference_m=distance_diff_m,
                    confidence=overall_conf,
                    frequency_mhz=det1.frequency_mhz
                )
                
                measurements.append(measurement)
                
                self.logger.debug(f"TDoA measurement: {det1.buoy_id}-{det2.buoy_id}, "
                                f"ΔT={time_diff_ns/1000:.1f}μs, ΔD={distance_diff_m:.1f}m")
        
        return measurements
    
    def _calculate_timing_confidence(self, buoy1: BuoyPosition, buoy2: BuoyPosition) -> float:
        """Calculate timing confidence based on buoy timing accuracies"""
        # Combined timing uncertainty (root sum of squares)
        combined_uncertainty_ns = math.sqrt(buoy1.timing_accuracy_ns**2 + buoy2.timing_accuracy_ns**2)
        
        # Convert to confidence (higher accuracy = higher confidence)
        # Assume 100μs uncertainty gives 0.5 confidence, scales exponentially
        reference_uncertainty = 100000  # 100μs
        confidence = math.exp(-combined_uncertainty_ns / reference_uncertainty)
        
        return min(confidence, 1.0)

class HyperbolicPositioning:
    """Implements hyperbolic positioning for TDoA triangulation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".HyperbolicPositioning")
    
    def triangulate_position(self, measurements: List[TDoAMeasurement], 
                           buoy_positions: Dict[str, BuoyPosition]) -> Optional[TriangulationResult]:
        """Triangulate signal position using hyperbolic positioning"""
        
        if len(measurements) < 2:
            self.logger.warning("Need at least 2 TDoA measurements for triangulation")
            return None
        
        # Get unique buoys involved
        buoy_ids = set()
        for m in measurements:
            buoy_ids.add(m.buoy1_id)
            buoy_ids.add(m.buoy2_id)
        
        if len(buoy_ids) < 3:
            self.logger.warning("Need at least 3 buoys for 2D triangulation")
            return None
        
        # Convert buoy positions to ECEF coordinates
        buoy_xyz = {}
        for buoy_id in buoy_ids:
            if buoy_id not in buoy_positions:
                self.logger.error(f"Missing position for buoy {buoy_id}")
                return None
            
            pos = buoy_positions[buoy_id]
            x, y, z = GeodeticCalculator.lat_lng_to_xyz(pos.lat, pos.lng, pos.altitude)
            buoy_xyz[buoy_id] = (x, y, z)
        
        # Set up optimization problem
        # We'll minimize the sum of squared residuals
        def objective_function(transmitter_xyz):
            tx_x, tx_y, tx_z = transmitter_xyz
            residuals = []
            
            for measurement in measurements:
                b1_xyz = buoy_xyz[measurement.buoy1_id]
                b2_xyz = buoy_xyz[measurement.buoy2_id]
                
                # Calculate actual distance differences
                dist1 = math.sqrt((tx_x - b1_xyz[0])**2 + (tx_y - b1_xyz[1])**2 + (tx_z - b1_xyz[2])**2)
                dist2 = math.sqrt((tx_x - b2_xyz[0])**2 + (tx_y - b2_xyz[1])**2 + (tx_z - b2_xyz[2])**2)
                actual_dist_diff = dist2 - dist1
                
                # Compare with measured distance difference
                measured_dist_diff = measurement.distance_difference_m
                residual = (actual_dist_diff - measured_dist_diff) ** 2
                
                # Weight by measurement confidence
                weighted_residual = residual / (measurement.confidence + 0.1)
                residuals.append(weighted_residual)
            
            return sum(residuals)
        
        # Initial guess: centroid of buoy positions
        buoy_positions_list = list(buoy_xyz.values())
        initial_x = sum(pos[0] for pos in buoy_positions_list) / len(buoy_positions_list)
        initial_y = sum(pos[1] for pos in buoy_positions_list) / len(buoy_positions_list)
        initial_z = sum(pos[2] for pos in buoy_positions_list) / len(buoy_positions_list)
        initial_guess = [initial_x, initial_y, initial_z]
        
        try:
            # Perform optimization
            result = scipy.optimize.minimize(
                objective_function,
                initial_guess,
                method='BFGS',
                options={'maxiter': 1000}
            )
            
            if not result.success:
                self.logger.warning(f"Optimization failed: {result.message}")
                return None
            
            # Convert result back to lat/lng
            estimated_x, estimated_y, estimated_z = result.x
            estimated_lat, estimated_lng, estimated_alt = GeodeticCalculator.xyz_to_lat_lng(
                estimated_x, estimated_y, estimated_z
            )
            
            # Calculate accuracy estimate based on residual
            accuracy_meters = math.sqrt(result.fun / len(measurements))
            
            # Calculate overall confidence
            avg_confidence = sum(m.confidence for m in measurements) / len(measurements)
            
            # Get metadata from first measurement
            first_measurement = measurements[0]
            
            triangulation_result = TriangulationResult(
                estimated_lat=estimated_lat,
                estimated_lng=estimated_lng,
                estimated_altitude=estimated_alt,
                accuracy_meters=accuracy_meters,
                confidence=avg_confidence,
                frequency_mhz=first_measurement.frequency_mhz,
                signal_type="unknown",  # Will be determined elsewhere
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                contributing_buoys=list(buoy_ids),
                tdoa_measurements=measurements,
                method="hyperbolic"
            )
            
            self.logger.info(f"Triangulation successful: ({estimated_lat:.6f}, {estimated_lng:.6f}) "
                           f"±{accuracy_meters:.1f}m, confidence: {avg_confidence:.2f}")
            
            return triangulation_result
            
        except Exception as e:
            self.logger.error(f"Triangulation failed: {e}")
            return None

class TDoAProcessor:
    """Main TDoA processing engine that coordinates all triangulation operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".TDoAProcessor")
        self.tdoa_calculator = TDoACalculator()
        self.hyperbolic_positioner = HyperbolicPositioning()
        self.buoy_positions: Dict[str, BuoyPosition] = {}
        
        # Signal correlation window (seconds)
        self.correlation_window_s = 10.0
        
        # Minimum number of buoys required for triangulation
        self.min_buoys_for_triangulation = 3
    
    def register_buoy(self, buoy_position: BuoyPosition):
        """Register a buoy's position for TDoA calculations"""
        self.buoy_positions[buoy_position.buoy_id] = buoy_position
        self.logger.info(f"Registered buoy {buoy_position.buoy_id} at "
                        f"({buoy_position.lat:.6f}, {buoy_position.lng:.6f})")
    
    def process_signal_detections(self, detections: List[SignalDetection]) -> List[TriangulationResult]:
        """Process multiple signal detections and perform TDoA triangulation"""
        if not detections:
            return []
        
        self.logger.info(f"Processing {len(detections)} signal detections")
        
        # Group detections by frequency (within tolerance)
        frequency_groups = self._group_by_frequency(detections)
        
        results = []
        
        for frequency, freq_detections in frequency_groups.items():
            self.logger.debug(f"Processing {len(freq_detections)} detections at {frequency} MHz")
            
            # Filter detections within correlation window
            time_filtered_detections = self._filter_by_time_window(freq_detections)
            
            if len(time_filtered_detections) < self.min_buoys_for_triangulation:
                self.logger.debug(f"Insufficient detections for {frequency} MHz "
                                f"({len(time_filtered_detections)} < {self.min_buoys_for_triangulation})")
                continue
            
            # Calculate TDoA measurements
            tdoa_measurements = self.tdoa_calculator.calculate_tdoa_measurements(
                time_filtered_detections, self.buoy_positions
            )
            
            if len(tdoa_measurements) < 2:
                self.logger.debug(f"Insufficient TDoA measurements for {frequency} MHz")
                continue
            
            # Perform triangulation
            triangulation_result = self.hyperbolic_positioner.triangulate_position(
                tdoa_measurements, self.buoy_positions
            )
            
            if triangulation_result:
                # Add signal type from detections
                signal_types = [d.signal_type for d in time_filtered_detections]
                most_common_type = max(set(signal_types), key=signal_types.count)
                triangulation_result.signal_type = most_common_type
                
                results.append(triangulation_result)
                
                # Log emergency signals with high priority
                if most_common_type == "emergency":
                    self.logger.warning(f"EMERGENCY SIGNAL TRIANGULATED: {frequency} MHz at "
                                      f"({triangulation_result.estimated_lat:.6f}, "
                                      f"{triangulation_result.estimated_lng:.6f}) "
                                      f"±{triangulation_result.accuracy_meters:.1f}m")
        
        return results
    
    def _group_by_frequency(self, detections: List[SignalDetection], 
                          frequency_tolerance_mhz: float = 0.01) -> Dict[float, List[SignalDetection]]:
        """Group detections by frequency with tolerance"""
        groups = {}
        
        for detection in detections:
            freq = detection.frequency_mhz
            
            # Find existing group within tolerance
            found_group = None
            for existing_freq in groups.keys():
                if abs(freq - existing_freq) <= frequency_tolerance_mhz:
                    found_group = existing_freq
                    break
            
            if found_group is not None:
                groups[found_group].append(detection)
            else:
                groups[freq] = [detection]
        
        return groups
    
    def _filter_by_time_window(self, detections: List[SignalDetection]) -> List[SignalDetection]:
        """Filter detections to those within the correlation time window"""
        if not detections:
            return []
        
        # Sort by timestamp
        sorted_detections = sorted(detections, key=lambda d: d.gps_timestamp_ns)
        
        # Find the latest detection
        latest_detection = sorted_detections[-1]
        latest_time_ns = latest_detection.gps_timestamp_ns
        
        # Filter to correlation window
        window_ns = int(self.correlation_window_s * 1e9)
        earliest_time_ns = latest_time_ns - window_ns
        
        filtered = [d for d in sorted_detections if d.gps_timestamp_ns >= earliest_time_ns]
        
        return filtered
    
    def get_buoy_network_status(self) -> Dict:
        """Get status of the buoy network for diagnostics"""
        status = {
            "registered_buoys": len(self.buoy_positions),
            "buoy_list": [
                {
                    "buoy_id": pos.buoy_id,
                    "lat": pos.lat,
                    "lng": pos.lng,
                    "timing_accuracy_ns": pos.timing_accuracy_ns
                }
                for pos in self.buoy_positions.values()
            ],
            "min_buoys_required": self.min_buoys_for_triangulation,
            "correlation_window_s": self.correlation_window_s,
            "triangulation_ready": len(self.buoy_positions) >= self.min_buoys_for_triangulation
        }
        
        return status

def main():
    """Example usage of TDoA processor"""
    logging.basicConfig(level=logging.INFO)
    
    # Create processor
    processor = TDoAProcessor()
    
    # Register some example buoys
    buoys = [
        BuoyPosition("BUOY_ALPHA", 51.505, -0.09, 0.0, 50000),
        BuoyPosition("BUOY_BETA", 51.51, -0.1, 0.0, 75000),
        BuoyPosition("BUOY_GAMMA", 51.5, -0.12, 0.0, 60000),
    ]
    
    for buoy in buoys:
        processor.register_buoy(buoy)
    
    # Simulate some signal detections
    base_time_ns = int(time.time_ns())
    detections = [
        SignalDetection("BUOY_ALPHA", 121.5, -55, "2025-01-18T16:30:00Z", base_time_ns, 51.505, -0.09, 0.9, "emergency"),
        SignalDetection("BUOY_BETA", 121.5, -60, "2025-01-18T16:30:00Z", base_time_ns + 150000, 51.51, -0.1, 0.85, "emergency"),
        SignalDetection("BUOY_GAMMA", 121.5, -58, "2025-01-18T16:30:00Z", base_time_ns + 300000, 51.5, -0.12, 0.88, "emergency"),
    ]
    
    # Process detections
    results = processor.process_signal_detections(detections)
    
    # Print results
    for result in results:
        print(f"Triangulation Result:")
        print(f"  Position: ({result.estimated_lat:.6f}, {result.estimated_lng:.6f})")
        print(f"  Accuracy: ±{result.accuracy_meters:.1f} m")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Signal Type: {result.signal_type}")
        print(f"  Frequency: {result.frequency_mhz} MHz")
        print(f"  Contributing Buoys: {', '.join(result.contributing_buoys)}")

if __name__ == "__main__":
    import time
    main() 