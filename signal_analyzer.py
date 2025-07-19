#!/usr/bin/env python3
"""
Signal Analysis Tool for RTL-SDR Captured IQ Data
This script analyzes IQ data files and extracts signal information.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import signal
from datetime import datetime

def load_iq_data(filename, sample_rate=2048000):
    """
    Load IQ data from binary file
    
    Args:
        filename: Path to the IQ data file
        sample_rate: Sample rate used during capture
    
    Returns:
        complex_samples: Complex IQ samples
        sample_rate: Sample rate
    """
    try:
        # Load binary data (8-bit unsigned integers)
        raw_data = np.fromfile(filename, dtype=np.uint8)
        
        # Convert to float and center around zero
        raw_data = raw_data.astype(np.float32) - 127.5
        
        # Separate I and Q components and create complex samples
        i_samples = raw_data[0::2]  # Even indices are I
        q_samples = raw_data[1::2]  # Odd indices are Q
        complex_samples = i_samples + 1j * q_samples
        
        print(f"Loaded {len(complex_samples)} complex samples from {filename}")
        print(f"Duration: {len(complex_samples) / sample_rate:.2f} seconds")
        
        return complex_samples, sample_rate
        
    except Exception as e:
        print(f"Error loading IQ data: {e}")
        return None, None

def analyze_spectrum(complex_samples, sample_rate, center_freq_mhz):
    """
    Analyze the frequency spectrum of the IQ data
    
    Args:
        complex_samples: Complex IQ samples
        sample_rate: Sample rate in Hz
        center_freq_mhz: Center frequency in MHz
    
    Returns:
        frequencies: Frequency array in MHz
        power_spectrum: Power spectrum in dB
        peak_freqs: List of peak frequencies
    """
    # Calculate FFT
    N = len(complex_samples)
    fft_result = np.fft.fft(complex_samples)
    fft_shifted = np.fft.fftshift(fft_result)
    
    # Calculate power spectrum in dB
    power_spectrum = 20 * np.log10(np.abs(fft_shifted) + 1e-12)
    
    # Create frequency array in MHz
    freq_array = np.fft.fftfreq(N, 1/sample_rate)
    freq_array_shifted = np.fft.fftshift(freq_array)
    frequencies = (freq_array_shifted / 1e6) + center_freq_mhz
    
    # Find peaks in the spectrum
    peaks, _ = signal.find_peaks(power_spectrum, height=np.mean(power_spectrum) + 10)
    peak_freqs = frequencies[peaks]
    peak_powers = power_spectrum[peaks]
    
    print(f"\nSpectrum Analysis Results:")
    print(f"  Frequency range: {frequencies[0]:.2f} to {frequencies[-1]:.2f} MHz")
    print(f"  Peak frequencies found: {len(peak_freqs)}")
    
    for i, (freq, power) in enumerate(zip(peak_freqs, peak_powers)):
        print(f"    Peak {i+1}: {freq:.3f} MHz ({power:.1f} dB)")
    
    return frequencies, power_spectrum, peak_freqs

def calculate_signal_stats(complex_samples):
    """Calculate basic signal statistics"""
    
    # Signal power
    signal_power = np.mean(np.abs(complex_samples)**2)
    signal_power_db = 10 * np.log10(signal_power + 1e-12)
    
    # Peak amplitude
    peak_amplitude = np.max(np.abs(complex_samples))
    
    # RMS amplitude
    rms_amplitude = np.sqrt(np.mean(np.abs(complex_samples)**2))
    
    print(f"\nSignal Statistics:")
    print(f"  Signal Power: {signal_power_db:.2f} dB")
    print(f"  Peak Amplitude: {peak_amplitude:.2f}")
    print(f"  RMS Amplitude: {rms_amplitude:.2f}")
    print(f"  Total Samples: {len(complex_samples)}")
    
    return {
        'power_db': signal_power_db,
        'peak_amplitude': peak_amplitude,
        'rms_amplitude': rms_amplitude,
        'num_samples': len(complex_samples)
    }

def plot_spectrum(frequencies, power_spectrum, center_freq_mhz, output_file=None):
    """Create a plot of the power spectrum"""
    
    plt.figure(figsize=(12, 6))
    plt.plot(frequencies, power_spectrum)
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power (dB)')
    plt.title(f'Power Spectrum - Center Frequency: {center_freq_mhz} MHz')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at center frequency
    plt.axvline(x=center_freq_mhz, color='red', linestyle='--', alpha=0.7, label='Center Freq')
    plt.legend()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Spectrum plot saved to {output_file}")
    else:
        plt.show()
    
    plt.close()

def analyze_iq_file(filename):
    """Complete analysis of an IQ data file"""
    
    # Extract frequency from filename
    center_freq_mhz = None
    if 'MHz' in filename:
        try:
            freq_part = filename.split('_')[1]  # e.g., "100.0MHz"
            center_freq_mhz = float(freq_part.replace('MHz', ''))
        except:
            center_freq_mhz = 100.0  # Default
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {filename}")
    print(f"Center Frequency: {center_freq_mhz} MHz")
    print(f"{'='*60}")
    
    # Load IQ data
    complex_samples, sample_rate = load_iq_data(filename)
    if complex_samples is None:
        return None
    
    # Calculate basic statistics
    stats = calculate_signal_stats(complex_samples)
    
    # Analyze spectrum
    frequencies, power_spectrum, peak_freqs = analyze_spectrum(
        complex_samples, sample_rate, center_freq_mhz
    )
    
    # Create plot filename
    plot_filename = filename.replace('.bin', '_spectrum.png')
    plot_spectrum(frequencies, power_spectrum, center_freq_mhz, plot_filename)
    
    return {
        'filename': filename,
        'center_freq_mhz': center_freq_mhz,
        'stats': stats,
        'peak_frequencies': peak_freqs,
        'spectrum_plot': plot_filename
    }

if __name__ == "__main__":
    print("=== RTL-SDR IQ Data Analysis Tool ===")
    
    # Find all IQ capture files
    iq_files = [f for f in os.listdir('.') if f.startswith('iq_capture_') and f.endswith('.bin')]
    
    if not iq_files:
        print("No IQ capture files found. Run sdr_capture.py first.")
        sys.exit(1)
    
    print(f"Found {len(iq_files)} IQ data files to analyze:")
    for f in iq_files:
        print(f"  - {f}")
    
    # Analyze each file
    results = []
    for filename in iq_files:
        try:
            result = analyze_iq_file(filename)
            if result:
                results.append(result)
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
    
    print(f"\n{'='*60}")
    print("Analysis Summary:")
    print(f"{'='*60}")
    
    for result in results:
        print(f"\nFile: {result['filename']}")
        print(f"  Center Freq: {result['center_freq_mhz']} MHz")
        print(f"  Signal Power: {result['stats']['power_db']:.2f} dB")
        print(f"  Peak Frequencies: {len(result['peak_frequencies'])} found")
        print(f"  Spectrum Plot: {result['spectrum_plot']}")
    
    print("\nAnalysis complete!") 