# Radio-Mapper Project Roadmap

This document outlines the development roadmap for the Radio-Mapper project. The **core mission** is to create a distributed network of "buoy" nodes that can detect arbitrary radio signals (especially emergency transmissions) and triangulate their position using Time Difference of Arrival (TDoA) techniques.

## Core Objectives

**Primary Use Case**: When an emergency radio keys up, the system should:
1. Detect the signal across multiple distributed buoy nodes
2. Capture precise timestamps for each detection
3. Use TDoA algorithms to triangulate the transmitter's location
4. Display the result on a public-facing web interface

**Secondary Use Case**: User-requested signal triangulation:
1. User provides signal pattern/frequency they want to locate
2. System searches historical data from all buoy nodes for matching signals
3. TDoA triangulation shows where that signal was transmitted from
4. Used for finding interference sources, rogue transmitters, etc.

**Key Requirements**:
- High-precision timing synchronization (sub-microsecond accuracy)
- Real IQ data streaming from distributed nodes to central processor
- Arbitrary signal detection and pattern matching
- Geographic focus: Oklahoma City metro area for initial deployment
- Real-time triangulation and alerting capabilities

## Phase 1: Real IQ Data Streaming and Local Testing (Current Focus)

### Steps:

1.  **Geographic Setup for Oklahoma City:**
    *   [x] Update map center to Oklahoma City (35.4676, -97.5164)
    *   [x] Configure realistic buoy positions around OKC metro area
    *   [ ] Plan actual deployment locations (need 4+ nodes minimum for good coverage)

2.  **Real IQ Data Client:**
    *   [ ] **NEW**: Create IQ streaming client that connects to central server
    *   [ ] Implement continuous IQ data capture from local RTL-SDR
    *   [ ] Add signal detection and timestamp correlation
    *   [ ] Stream detected signals with precise GPS timestamps to central processor
    *   [ ] Test with single local SDR first, then expand to multiple nodes

3.  **Central Signal Processing Server:**
    *   [ ] **NEW**: Implement central server that receives IQ streams from multiple clients
    *   [ ] Add real-time signal correlation and pattern matching
    *   [ ] Implement user-requested signal search ("find this frequency/pattern")
    *   [ ] Store historical signal data for post-analysis triangulation
    *   [ ] Real TDoA processing based on actual signal signatures and timestamps

4.  **Minimum Node Requirements:**
    *   [ ] **Analysis**: Determine minimum nodes needed for OKC coverage
    *   [ ] **Estimate**: 4-6 nodes for urban area, 8-12 for metro coverage
    *   [ ] Geographic distribution planning for optimal TDoA accuracy
    *   [ ] Test TDoA accuracy vs number of nodes

## Phase 2: Multi-Node OKC Deployment

### Steps:

1.  **Local Multi-Node Testing:**
    *   [ ] Set up 3-4 local test nodes around immediate area
    *   [ ] Test real signal triangulation with known transmitter positions
    *   [ ] Validate TDoA accuracy with actual hardware and timing
    *   [ ] Measure and optimize timing synchronization

2.  **User Signal Request System:**
    *   [ ] Web interface for users to request "find this signal"
    *   [ ] Upload signal samples or specify frequency/pattern
    *   [ ] Historical data search and triangulation
    *   [ ] Results display with confidence intervals

3.  **Production OKC Deployment:**
    *   [ ] Deploy nodes at strategic locations around OKC metro
    *   [ ] Test emergency frequency monitoring (aviation, public safety)
    *   [ ] Validate against known transmitter locations
    *   [ ] Performance monitoring and accuracy validation

## Technical Questions to Address:

### Node Count for OKC:
- **Minimum for 2D positioning**: 3 nodes 
- **Recommended for accuracy**: 4-6 nodes for urban OKC
- **Optimal coverage**: 8-12 nodes for full metro area
- **Node spacing**: 5-15 km apart for optimal TDoA geometry

### Current TDoA Logic Status:
✅ **Mathematical algorithms implemented** - Hyperbolic positioning with optimization
✅ **Timestamp correlation** - Nanosecond precision GPS timing
✅ **Signal matching** - Cross-correlation for multi-node detection
⚠️ **Needs real data testing** - Currently using simulated timestamps
⚠️ **Needs signal signature matching** - Pattern recognition for arbitrary signals

## Immediate Development Priority:

1. **Fix geographic center** to Oklahoma City
2. **Create IQ streaming client** for real SDR data
3. **Test single-node signal detection** with local SDR
4. **Add user signal request interface**
5. **Test multi-node simulation** with realistic OKC positions 