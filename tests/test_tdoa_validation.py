#!/usr/bin/env python3
"""
TDoA Validation Test - Radio-Mapper Emergency Response System

This test validates the Time Difference of Arrival (TDoA) triangulation system
by simulating known signal sources and measuring accuracy.
"""

import sys
import os
import time
import math
import numpy as np
from datetime import datetime, timezone
from typing import List, Tuple, Dict

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tdoa_processor import (
    TDoAProcessor, 
    SignalDetection, 
    BuoyPosition,
    GeodeticCalculator
)

class TDoAValidationTest:
    """Comprehensive TDoA system validation"""
    
    def __init__(self):
        self.processor = TDoAProcessor()
        self.speed_of_light = 299792458.0  # m/s
        
        # OKC Metro buoy network (theoretical positions)
        self.buoys = [
            BuoyPosition("BUOY_NORTH", 35.5276, -97.5164, 0.0, 1000),  # Edmond
            BuoyPosition("BUOY_WEST", 35.4676, -97.6164, 0.0, 1000),   # West OKC  
            BuoyPosition("BUOY_EAST", 35.4676, -97.4164, 0.0, 1000),   # East OKC
            BuoyPosition("BUOY_SOUTH", 35.3776, -97.5164, 0.0, 1000),  # Moore
        ]
        
        # Register buoys with processor
        for buoy in self.buoys:
            self.processor.register_buoy(buoy)
            
        print(f"✅ Registered {len(self.buoys)} buoys for TDoA testing")
        
    def test_timing_accuracy_impact(self):
        """Test how timing errors affect triangulation accuracy"""
        print("\n" + "="*60)
        print("📊 TIMING ACCURACY IMPACT ANALYSIS")
        print("="*60)
        
        # Known transmitter position (OKC downtown)
        true_lat, true_lng = 35.4676, -97.5164
        
        timing_errors = [0, 100, 1000, 10000, 100000, 1000000]  # nanoseconds
        
        print(f"True transmitter position: {true_lat:.6f}, {true_lng:.6f}")
        print(f"{'Timing Error':<12} {'Distance Error':<15} {'Position Error':<15}")
        print("-" * 45)
        
        for timing_error_ns in timing_errors:
            # Simulate signal detections with timing error
            detections = self._simulate_signal_detections(
                true_lat, true_lng, 121.5, timing_error_ns
            )
            
            # Perform triangulation
            results = self.processor.process_signal_detections(detections)
            
            if results:
                result = results[0]
                # Calculate position error
                position_error_m = GeodeticCalculator.distance_3d(
                    true_lat, true_lng, 0,
                    result.estimated_lat, result.estimated_lng, 0
                )
                
                # Calculate equivalent distance error from timing
                distance_error_m = (timing_error_ns / 1e9) * self.speed_of_light
                
                print(f"{timing_error_ns/1000:>8.1f} μs   {distance_error_m:>10.1f} m      {position_error_m:>10.1f} m")
            else:
                print(f"{timing_error_ns/1000:>8.1f} μs   {'FAILED':>10s}        {'FAILED':>10s}")
    
    def test_buoy_geometry_impact(self):
        """Test how buoy positioning affects triangulation accuracy"""
        print("\n" + "="*60)
        print("📐 BUOY GEOMETRY IMPACT ANALYSIS")
        print("="*60)
        
        # Test different buoy configurations
        configurations = [
            ("Triangle", self.buoys[:3]),
            ("Square", self.buoys),
            ("Linear", [self.buoys[0], self.buoys[1], 
                      BuoyPosition("BUOY_LINEAR", 35.4676, -97.3164, 0.0, 1000)])
        ]
        
        true_lat, true_lng = 35.4676, -97.5164  # OKC downtown
        
        print(f"True transmitter position: {true_lat:.6f}, {true_lng:.6f}")
        print(f"{'Configuration':<12} {'Buoys':<6} {'Accuracy':<12} {'Confidence':<12}")
        print("-" * 50)
        
        for config_name, buoy_set in configurations:
            # Temporarily set processor buoys
            temp_processor = TDoAProcessor()
            for buoy in buoy_set:
                temp_processor.register_buoy(buoy)
            
            # Simulate signal detections
            detections = self._simulate_signal_detections_for_buoys(
                true_lat, true_lng, 121.5, buoy_set, timing_error_ns=1000
            )
            
            # Perform triangulation
            results = temp_processor.process_signal_detections(detections)
            
            if results:
                result = results[0]
                position_error_m = GeodeticCalculator.distance_3d(
                    true_lat, true_lng, 0,
                    result.estimated_lat, result.estimated_lng, 0
                )
                
                print(f"{config_name:<12} {len(buoy_set):<6} {position_error_m:>8.1f} m    {result.confidence:>8.2f}")
            else:
                print(f"{config_name:<12} {len(buoy_set):<6} {'FAILED':<12} {'N/A':<12}")
    
    def test_signal_correlation_accuracy(self):
        """Test signal correlation and matching across buoys"""
        print("\n" + "="*60)
        print("🔗 SIGNAL CORRELATION ACCURACY TEST")
        print("="*60)
        
        # Test different frequency offsets and timing spreads
        test_cases = [
            {"name": "Perfect Match", "freq_offset": 0.0, "time_spread": 0},
            {"name": "Small Freq Drift", "freq_offset": 0.005, "time_spread": 0},
            {"name": "Large Freq Drift", "freq_offset": 0.02, "time_spread": 0},
            {"name": "Time Spread", "freq_offset": 0.0, "time_spread": 500000},  # 500μs
            {"name": "Both Issues", "freq_offset": 0.01, "time_spread": 1000000},  # 1ms
        ]
        
        true_lat, true_lng = 35.4676, -97.5164
        base_frequency = 121.5
        
        print(f"{'Test Case':<18} {'Detections':<12} {'Correlated':<12} {'Success':<8}")
        print("-" * 55)
        
        for test_case in test_cases:
            detections = []
            base_time_ns = int(time.time_ns())
            
            for i, buoy in enumerate(self.buoys):
                # Calculate expected arrival time based on distance
                distance = GeodeticCalculator.distance_3d(
                    true_lat, true_lng, 0,
                    buoy.lat, buoy.lng, 0
                )
                travel_time_ns = int((distance / self.speed_of_light) * 1e9)
                
                # Add frequency offset and time spread
                freq = base_frequency + (test_case["freq_offset"] * (i % 2))
                arrival_time = base_time_ns + travel_time_ns + (i * test_case["time_spread"])
                
                detection = SignalDetection(
                    buoy_id=buoy.buoy_id,
                    frequency_mhz=freq,
                    signal_strength_dbm=-65.0,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    gps_timestamp_ns=arrival_time,
                    lat=buoy.lat,
                    lng=buoy.lng,
                    confidence=0.9,
                    signal_type="emergency"
                )
                detections.append(detection)
            
            # Test correlation
            results = self.processor.process_signal_detections(detections)
            
            correlated = len(results) > 0
            success = "✅" if correlated else "❌"
            
            print(f"{test_case['name']:<18} {len(detections):<12} {len(results):<12} {success:<8}")
    
    def test_real_world_scenarios(self):
        """Test realistic emergency signal scenarios"""
        print("\n" + "="*60)
        print("🚨 REAL-WORLD EMERGENCY SCENARIOS")
        print("="*60)
        
        scenarios = [
            {
                "name": "Aviation Emergency (121.5 MHz)",
                "frequency": 121.5,
                "signal_strength": -75,  # Weaker signal
                "location": (35.4000, -97.6000),  # Will Rogers Airport area
                "signal_type": "emergency"
            },
            {
                "name": "Marine Emergency (156.8 MHz)", 
                "frequency": 156.8,
                "signal_strength": -70,
                "location": (35.5000, -97.4000),  # Lake area
                "signal_type": "emergency"
            },
            {
                "name": "PLB Beacon (406.025 MHz)",
                "frequency": 406.025,
                "signal_strength": -80,  # Very weak
                "location": (35.3500, -97.5500),  # Remote area
                "signal_type": "emergency"
            }
        ]
        
        print(f"{'Scenario':<25} {'True Position':<20} {'Estimated Position':<20} {'Error':<10}")
        print("-" * 80)
        
        for scenario in scenarios:
            true_lat, true_lng = scenario["location"]
            
            # Simulate realistic signal detection
            detections = self._simulate_signal_detections(
                true_lat, true_lng, scenario["frequency"], 
                timing_error_ns=10000,  # 10μs realistic GPS timing
                signal_strength=scenario["signal_strength"]
            )
            
            # Add some realistic noise/variation
            for detection in detections:
                detection.signal_strength_dbm += np.random.uniform(-5, 5)
                detection.confidence *= np.random.uniform(0.8, 1.0)
            
            results = self.processor.process_signal_detections(detections)
            
            if results:
                result = results[0]
                error_m = GeodeticCalculator.distance_3d(
                    true_lat, true_lng, 0,
                    result.estimated_lat, result.estimated_lng, 0
                )
                
                true_pos = f"{true_lat:.4f}, {true_lng:.4f}"
                est_pos = f"{result.estimated_lat:.4f}, {result.estimated_lng:.4f}"
                
                print(f"{scenario['name']:<25} {true_pos:<20} {est_pos:<20} {error_m:>6.0f} m")
            else:
                print(f"{scenario['name']:<25} {true_lat:.4f}, {true_lng:.4f}     {'FAILED':<20} {'N/A':<10}")
    
    def _simulate_signal_detections(self, true_lat: float, true_lng: float, 
                                  frequency: float, timing_error_ns: int = 0,
                                  signal_strength: float = -65.0) -> List[SignalDetection]:
        """Simulate signal detections from all buoys for a known transmitter"""
        return self._simulate_signal_detections_for_buoys(
            true_lat, true_lng, frequency, self.buoys, timing_error_ns, signal_strength
        )
    
    def _simulate_signal_detections_for_buoys(self, true_lat: float, true_lng: float,
                                            frequency: float, buoys: List[BuoyPosition],
                                            timing_error_ns: int = 0,
                                            signal_strength: float = -65.0) -> List[SignalDetection]:
        """Simulate signal detections for specific buoy set"""
        detections = []
        base_time_ns = int(time.time_ns())
        
        for buoy in buoys:
            # Calculate distance and travel time
            distance = GeodeticCalculator.distance_3d(
                true_lat, true_lng, 0,
                buoy.lat, buoy.lng, 0
            )
            
            # Calculate signal travel time
            travel_time_ns = int((distance / self.speed_of_light) * 1e9)
            
            # Add timing error (simulates GPS timing inaccuracy)
            arrival_time_ns = base_time_ns + travel_time_ns + np.random.randint(-timing_error_ns, timing_error_ns + 1)
            
            # Calculate signal strength with distance attenuation
            # Free space path loss approximation
            distance_km = distance / 1000
            path_loss_db = 20 * math.log10(distance_km) + 20 * math.log10(frequency) - 27.55
            received_strength = signal_strength - path_loss_db
            
            detection = SignalDetection(
                buoy_id=buoy.buoy_id,
                frequency_mhz=frequency,
                signal_strength_dbm=received_strength,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                gps_timestamp_ns=arrival_time_ns,
                lat=buoy.lat,
                lng=buoy.lng,
                confidence=0.9,
                signal_type="emergency"
            )
            detections.append(detection)
        
        return detections
    
    def run_all_tests(self):
        """Run complete TDoA validation test suite"""
        print("🔬 TDoA VALIDATION TEST SUITE")
        print("Radio-Mapper Emergency Response System")
        print("=" * 60)
        
        print(f"Network Configuration:")
        for buoy in self.buoys:
            print(f"  {buoy.buoy_id}: {buoy.lat:.4f}, {buoy.lng:.4f}")
        
        # Run all validation tests
        self.test_timing_accuracy_impact()
        self.test_buoy_geometry_impact()
        self.test_signal_correlation_accuracy()
        self.test_real_world_scenarios()
        
        print("\n" + "="*60)
        print("📋 SUMMARY AND RECOMMENDATIONS")
        print("="*60)
        print("1. ⚠️  CRITICAL: Current system uses system time, not GPS timing")
        print("2. ⚠️  CRITICAL: Only 1 buoy deployed - need minimum 3 for triangulation")
        print("3. 📈 Timing accuracy directly impacts position accuracy:")
        print("   - 1μs timing error → ~300m position error")
        print("   - Current 100μs target → ~30km position error (unusable)")
        print("4. 🎯 Target: <1μs GPS timing for <100m position accuracy")
        print("5. 📡 Deploy 4-buoy network around OKC metro for coverage")
        
        print("\n🚀 NEXT STEPS:")
        print("   1. Implement GPS PPS timing synchronization")
        print("   2. Deploy additional buoys at strategic locations")
        print("   3. Test with known signal sources for validation")

def main():
    """Run TDoA validation tests"""
    try:
        validator = TDoAValidationTest()
        validator.run_all_tests()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 