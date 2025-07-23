# Radio-Mapper Project Roadmap

## 1. Core Mission

To develop a professional-grade, distributed network of GPS-synchronized "buoy" nodes capable of detecting and triangulating the position of radio signals in real-time. The system is designed for emergency response scenarios where locating a signal source quickly and accurately is critical.

### **Primary Use Case: Emergency Signal Triangulation**
1.  **Detect:** An emergency signal (e.g., aviation, marine) is transmitted.
2.  **Synchronize:** Multiple buoy nodes detect the signal and capture GPS-synchronized, sub-microsecond timestamps.
3.  **Triangulate:** The central processor uses Time Difference of Arrival (TDoA) algorithms to calculate the transmitter's location.
4.  **Visualize:** The location is displayed on the web interface in real-time.

---

## 2. Current System Status: **Partially Operational**

-   ✅ **Signal Detection:** The system can successfully detect signals on a single buoy across a pre-defined frequency schedule.
-   ✅ **Web Interface:** A functional UI displays detected signals and system status.
-   ❌ **Triangulation:** **NOT OPERATIONAL.** The core TDoA functionality is currently non-functional due to algorithm instability and the lack of a multi-buoy network.
-   ❌ **GPS Timing:** The system uses system time, not true GPS PPS time, which is insufficient for accurate triangulation.

---

## 3. Immediate Priorities: Stabilize and Validate

The immediate focus is to address the critical issues preventing triangulation and to validate the system's core algorithms.

### **Phase 1: Algorithm Stabilization & Simulation (Current Focus)**
1.  **Fix TDoA Optimization Algorithm:**
    -   [ ] **In Progress:** Implement a more robust initial guess and bounds for the `scipy.optimize.minimize` solver to prevent numerical instability.
    -   [ ] Investigate alternative, more stable optimization methods.
2.  **Improve Signal Correlation:**
    -   [ ] **In Progress:** Move from a strict frequency match to a more robust cross-correlation algorithm to handle frequency drift and noise.
3.  **Develop a Multi-Buoy Simulator:**
    -   [ ] Create a test environment to simulate data from 3-4 buoys.
    -   [ ] Use this simulator to validate the TDoA algorithm fixes without requiring full hardware deployment.

### **Phase 2: Hardware and Deployment (1-2 Months)**
1.  **Deploy a Multi-Buoy Network:**
    -   [ ] Procure and deploy a minimum of **3 buoys** in the Oklahoma City metro area.
    -   [ ] Target a 4-buoy square configuration for optimal accuracy.
2.  **Implement GPS PPS Timing:**
    -   [ ] Integrate GPS receivers with PPS (Pulse Per Second) output.
    -   [ ] Modify the system to use hardware-level, sub-microsecond timestamps for all signal detections.

---

## 4. Future Enhancements

Once the core triangulation functionality is operational and validated, the following enhancements will be prioritized:

-   **Mobile Source Tracking:** Implement Kalman filtering to track moving signal sources.
-   **Signal Fingerprinting:** Use IQ sample analysis to identify and classify different types of signals.
-   **Expanded Frequency Coverage:** Add more frequency bands to the scanning schedule.
-   **User-Requested Triangulation:** Allow users to submit a frequency or signal pattern for on-demand triangulation.

---

## 5. Success Metrics

### **Minimum Viable Product (MVP)**
-   ✅ 3+ buoys deployed and operational.
-   ✅ <1µs GPS timing synchronization.
-   ✅ <500m position accuracy for emergency signals.

### **Target Performance**
-   🎯 4-buoy network providing redundancy and improved accuracy.
-   🎯 <100m position accuracy.
-   🎯 <5-second response time from signal detection to a triangulated position on the map. 