
# TDoA System Analysis and Findings

## 1. Executive Summary

### Current Status: **NOT OPERATIONAL FOR TRIANGULATION**

The Radio-Mapper's Time Difference of Arrival (TDoA) system is a foundational component for locating emergency signals. However, comprehensive testing reveals that while individual signal detection is functional, the core triangulation capability is **not operational**. This document outlines the critical issues, provides a detailed analysis, and defines a clear path forward.

**Key Issues:**
- **Algorithm Instability:** The `scipy.optimize.minimize` function frequently fails, preventing successful triangulation.
- **Single Buoy Limitation:** The system is deployed with only one buoy, making triangulation physically impossible (minimum of 3 is required).
- **Inaccurate GPS Timing:** The system uses `time.time_ns()` (system clock) instead of true GPS Pulse-Per-Second (PPS) timing, leading to errors that make triangulation unviable.

**Path Forward:**
The immediate focus is to stabilize the core optimization algorithm and simulate multi-buoy data for testing. The next phase involves deploying a multi-buoy network with true GPS PPS hardware for sub-microsecond timing accuracy.

---

## 2. TDoA Architecture

### 2.1. GPS Timing Synchronization
- **Current Method:** `int(time.time_ns())`
- **Problem:** Relies on system time, which is not synchronized between buoys and is prone to drift. A 1ms error can result in a 300km position error.
- **Required Solution:** Implement true GPS PPS (Pulse Per Second) synchronization to achieve <1µs accuracy.

### 2.2. Signal Detection and Timestamping
- **Process:** Buoys scan a synchronized frequency schedule, and upon detection, a timestamp is captured.
- **Schedule:** A 35-second cycle covers emergency and commercial frequencies.
  ```python
  sync_schedule = [
      {"frequency": 105.7, "duration": 5, "type": "testing"},
      {"frequency": 121.5, "duration": 10, "type": "emergency"}, 
      {"frequency": 243.0, "duration": 10, "type": "emergency"},
      {"frequency": 156.8, "duration": 5, "type": "emergency"},
      {"frequency": 101.9, "duration": 5, "type": "testing"}
  ]
  ```

### 2.3. Triangulation Algorithm
- **Method:** Uses hyperbolic positioning, minimizing the sum of squared residuals with the `scipy.optimize.minimize` (BFGS) algorithm.
- **Formula:** `Distance Difference = (ΔT_nanoseconds / 1e9) × 299,792,458 m/s`
- **Problem:** The optimization algorithm is numerically unstable and frequently fails.

---

## 3. Critical Findings and Analysis

### 3.1. Optimization Algorithm Failures
- **Symptom:** "Optimization failed" errors from `scipy`.
- **Root Cause:** Poor numerical conditioning in the hyperbolic equations, and potentially a poor initial guess for the solver.
- **Impact:** The vast majority of triangulation attempts fail.

### 3.2. Timing Accuracy vs. Position Error
| Timing Error | Distance Error | Position Error | Status |
|--------------|----------------|----------------|--------|
| 1.0 µs | 299.8 m | 284.8 m | ✅ **Only working case** |
| 100.0 µs | 29,979 m | 11,947 m | ⚠️ Unacceptable Error |

This table clearly shows that **sub-microsecond timing accuracy is mandatory** for the system to be effective.

### 3.3. Buoy Geometry
- **Optimal Configuration:** A 4-buoy square configuration provides the best accuracy (86.1m).
- **Minimum Configuration:** A 3-buoy triangle is the minimum for 2D triangulation (210.7m accuracy).

### 3.4. Signal Correlation
- The current algorithm, which uses a simple frequency tolerance, is too strict and fails to correlate signals with minor frequency drift. A more robust cross-correlation approach is needed.

---

## 4. Action Plan and Roadmap

### Phase 1: Algorithm and Simulation (Immediate Focus)
1.  **Stabilize Optimization Algorithm:**
    -   Implement a more robust initial guess for the solver.
    -   Introduce optimization bounds to constrain the search space.
    -   Investigate alternative optimization methods.
2.  **Improve Signal Correlation:**
    -   Increase frequency tolerance to ±50kHz.
    -   Implement a cross-correlation analysis for more robust signal matching.
3.  **Develop a Multi-Buoy Simulator:**
    -   Create a testing framework that can simulate data from multiple buoys to validate the algorithm without requiring full hardware deployment.

### Phase 2: Multi-Buoy Deployment (Next 1-2 Months)
1.  **Hardware Procurement:**
    -   Acquire at least 3 RTL-SDR devices with GPS modules that support PPS output.
2.  **Strategic Deployment:**
    -   Position the buoys in a triangular or square geometry around the target area (e.g., OKC metro).
    -   Suggested locations:
        ```
        North (Edmond): 35.5276, -97.5164
        West: 35.4676, -97.6164
        East: 35.4676, -97.4164  
        South (Moore): 35.3776, -97.5164
        ```

### Phase 3: GPS PPS Integration (Next 1-2 Months)
1.  **Hardware Integration:**
    -   Interface the GPS PPS output with the system to trigger hardware timestamps.
2.  **Software Integration:**
    -   Modify the `GPSTimeSource` to use the PPS-synchronized timestamps.

---

## 5. Success Criteria

### Minimum Viable Product (MVP)
- **3+ buoys** deployed and operational.
- **<1µs** GPS timing synchronization.
- **<500m** position accuracy for emergency signals.

### Target Performance
- **4-buoy network** for improved accuracy and redundancy.
- **<100m** position accuracy.
- **<5 second** response time from signal detection to triangulation. 