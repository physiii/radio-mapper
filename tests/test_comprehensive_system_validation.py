#!/usr/bin/env python3
"""
Comprehensive Radio-Mapper System Validation
This test validates the complete system and explains expected behavior
"""

import requests
import time
import json
from collections import Counter, defaultdict
from datetime import datetime

class ComprehensiveSystemValidator:
    def __init__(self):
        self.central_url = "http://localhost:5001"
        self.webapp_url = "http://localhost:5000"
        
    def test_frequency_cycling_system(self):
        """Test that the frequency cycling system works correctly"""
        print("🔄 Testing Frequency Cycling System")
        print("=" * 50)
        
        # Expected frequencies in the 35-second cycle
        expected_frequencies = [105.7, 121.5, 243.0, 156.8, 101.9]
        frequency_history = []
        
        print("Monitoring frequency changes over 45 seconds...")
        start_time = time.time()
        
        while time.time() - start_time < 45:
            response = requests.get(f"{self.central_url}/api/detections")
            detections = response.json()
            
            if detections:
                current_freqs = list(set([d['frequency_mhz'] for d in detections]))
                if current_freqs:
                    primary_freq = max(set([d['frequency_mhz'] for d in detections]), 
                                     key=[d['frequency_mhz'] for d in detections].count)
                    frequency_history.append((time.time() - start_time, primary_freq))
                    
            time.sleep(2)
        
        # Analyze frequency changes
        unique_freqs = list(set([freq for _, freq in frequency_history]))
        print(f"\n📊 Frequency Cycling Results:")
        print(f"Expected frequencies: {sorted(expected_frequencies)}")
        print(f"Detected frequencies: {sorted(unique_freqs)}")
        
        coverage = len(set(unique_freqs) & set(expected_frequencies))
        print(f"Frequency coverage: {coverage}/{len(expected_frequencies)} ({coverage/len(expected_frequencies)*100:.1f}%)")
        
        if coverage >= 3:
            print("✅ PASS: Frequency cycling system operational")
            return True
        else:
            print("❌ FAIL: Insufficient frequency coverage")
            return False
    
    def test_real_time_search_functionality(self):
        """Test real-time search for currently active frequencies"""
        print("\n🔍 Testing Real-Time Search Functionality")
        print("=" * 50)
        
        # Monitor and test search for 30 seconds
        start_time = time.time()
        search_tests = []
        
        while time.time() - start_time < 30:
            # Get current detections
            response = requests.get(f"{self.central_url}/api/detections")
            detections = response.json()
            
            if detections:
                # Find most common frequency (current scan target)
                freq_counts = Counter([d['frequency_mhz'] for d in detections])
                current_freq = freq_counts.most_common(1)[0][0]
                
                # Test search for this frequency
                search_response = requests.get(f"{self.webapp_url}/api/search-signals?frequency={current_freq}")
                search_results = search_response.json()
                
                search_tests.append({
                    'frequency': current_freq,
                    'detections_available': freq_counts[current_freq],
                    'search_results': len(search_results),
                    'success': len(search_results) > 0
                })
                
                if len(search_results) > 0:
                    print(f"✅ Search for {current_freq} MHz: Found {len(search_results)} results")
                    break
                else:
                    print(f"⏳ Search for {current_freq} MHz: 0 results (may be frequency transition)")
                    
            time.sleep(3)
        
        successful_searches = [t for t in search_tests if t['success']]
        
        if successful_searches:
            print(f"\n✅ PASS: Search functionality working ({len(successful_searches)} successful searches)")
            sample = successful_searches[0]
            print(f"Sample successful search: {sample['frequency']} MHz with {sample['search_results']} results")
            return True
        else:
            print(f"\n❌ FAIL: No successful searches in 30 seconds")
            return False
    
    def test_signal_persistence_and_quality(self):
        """Test signal data persistence and quality"""
        print("\n📊 Testing Signal Data Persistence and Quality")
        print("=" * 50)
        
        response = requests.get(f"{self.central_url}/api/detections")
        detections = response.json()
        
        if not detections:
            print("❌ FAIL: No detections available")
            return False
        
        # Analyze signal quality
        signal_strengths = [d['signal_strength_dbm'] for d in detections]
        confidences = [d['confidence'] for d in detections]
        
        avg_strength = sum(signal_strengths) / len(signal_strengths)
        avg_confidence = sum(confidences) / len(confidences)
        
        print(f"Signal quality metrics:")
        print(f"  - Total detections: {len(detections)}")
        print(f"  - Average signal strength: {avg_strength:.1f} dBm")
        print(f"  - Average confidence: {avg_confidence:.2f}")
        print(f"  - Signal strength range: {min(signal_strengths):.1f} to {max(signal_strengths):.1f} dBm")
        
        # Check for reasonable signal quality
        quality_ok = (
            len(detections) >= 10 and
            avg_confidence > 0.5 and
            avg_strength > -100
        )
        
        if quality_ok:
            print("✅ PASS: Signal quality is acceptable")
            return True
        else:
            print("❌ FAIL: Signal quality issues detected")
            return False
    
    def test_buoy_connectivity(self):
        """Test buoy connectivity and status"""
        print("\n🛰️ Testing Buoy Connectivity")
        print("=" * 50)
        
        response = requests.get(f"{self.central_url}/api/nodes")
        nodes = response.json()
        
        if not nodes:
            print("❌ FAIL: No buoys connected")
            return False
        
        print(f"Connected buoys: {len(nodes)}")
        for node in nodes:
            print(f"  - {node['id']}: {node['status']} (last seen: {node['lastSeen']})")
            
        active_nodes = [n for n in nodes if n['status'] == 'active']
        
        if active_nodes:
            print(f"✅ PASS: {len(active_nodes)} active buoys")
            return True
        else:
            print("❌ FAIL: No active buoys")
            return False
    
    def assess_triangulation_readiness(self):
        """Assess readiness for triangulation"""
        print("\n📐 Assessing Triangulation Readiness")
        print("=" * 50)
        
        # Get current nodes
        nodes_response = requests.get(f"{self.central_url}/api/nodes")
        nodes = nodes_response.json()
        
        # Get current detections
        detections_response = requests.get(f"{self.central_url}/api/detections")
        detections = detections_response.json()
        
        active_nodes = [n for n in nodes if n['status'] == 'active']
        
        print(f"Triangulation requirements:")
        print(f"  - Active buoys: {len(active_nodes)} (need ≥3 for triangulation)")
        print(f"  - Signal detections: {len(detections)} available")
        
        if detections:
            freq_counts = Counter([d['frequency_mhz'] for d in detections])
            print(f"  - Frequencies with signals: {list(freq_counts.keys())}")
        
        triangulation_ready = len(active_nodes) >= 3 and len(detections) > 0
        
        if triangulation_ready:
            print("✅ READY: System ready for triangulation")
        else:
            print("⏳ NOT READY: Need more buoys for triangulation")
            print("   Recommendation: Deploy 2 more buoys for full triangulation capability")
        
        return triangulation_ready
    
    def explain_system_behavior(self):
        """Explain the expected system behavior"""
        print("\n📚 System Behavior Explanation")
        print("=" * 50)
        
        print("🔄 FREQUENCY SCANNING CYCLE (35 seconds):")
        print("  0-5s:   105.7 MHz (FM Commercial)")
        print("  5-15s:  121.5 MHz (Emergency)")  
        print("  15-25s: 243.0 MHz (Emergency)")
        print("  25-30s: 156.8 MHz (Emergency)")
        print("  30-35s: 101.9 MHz (FM Commercial)")
        print()
        print("🔍 SEARCH BEHAVIOR:")
        print("  - Search returns signals for CURRENTLY ACTIVE frequency")
        print("  - 'No signals found' means that frequency isn't being scanned right now")
        print("  - Wait 35 seconds to see all frequencies in the cycle")
        print()
        print("📊 EXPECTED SEARCH RESULTS:")
        print("  - Emergency frequencies: More detections (longer scan windows)")
        print("  - Commercial frequencies: Fewer detections (shorter scan windows)")
        print("  - Search results change every 5-10 seconds as frequencies change")
        print()
        print("🎯 FOR TESTING 105.7 MHz:")
        print("  1. Wait for frequency cycle to reach 105.7 MHz (0-5 second window)")
        print("  2. Search immediately when 105.7 MHz signals appear")
        print("  3. Results will disappear when cycle moves to next frequency")
        
    def run_comprehensive_validation(self):
        """Run all validation tests"""
        print("🧪 Radio-Mapper Comprehensive System Validation")
        print("=" * 70)
        
        tests = [
            self.test_buoy_connectivity,
            self.test_signal_persistence_and_quality,
            self.test_frequency_cycling_system,
            self.test_real_time_search_functionality,
            self.assess_triangulation_readiness
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
                else:
                    # Continue even if some tests fail
                    pass
            except Exception as e:
                print(f"❌ Test error: {e}")
        
        print("\n" + "=" * 70)
        print(f"🧪 VALIDATION RESULTS: {passed}/{total} tests passed")
        
        if passed >= 3:
            print("✅ SYSTEM OPERATIONAL: Core functionality working")
            print("🎯 READY FOR: Signal detection and frequency monitoring")
            if passed == total:
                print("🚀 READY FOR: Full triangulation deployment")
        else:
            print("❌ SYSTEM ISSUES: Core problems need resolution")
        
        # Always show behavior explanation
        self.explain_system_behavior()
        
        return passed >= 3

if __name__ == "__main__":
    validator = ComprehensiveSystemValidator()
    validator.run_comprehensive_validation() 