import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Time axis
t = np.linspace(0, 10, 1000)

# Signals
ppg_finger = np.sin(2 * np.pi * 1.2 * t)
delay = 20
ppg_wrist = np.roll(ppg_finger, delay)

# Detect peaks
peaks_finger, _ = find_peaks(ppg_finger, distance=50)
peaks_wrist, _ = find_peaks(ppg_wrist, distance=50)

# --- PTT ---
min_len = min(len(peaks_finger), len(peaks_wrist))
ptt_values = []

for i in range(min_len):
    ptt = (peaks_wrist[i] - peaks_finger[i]) * (t[1] - t[0])
    ptt_values.append(ptt)

ptt_avg = np.mean(ptt_values)

# --- Heart Rate ---
time_diff = np.diff(t[peaks_finger])
hr = 60 / np.mean(time_diff)

# --- Other Features ---
peak_amp = np.mean(ppg_finger[peaks_finger])
signal_mean = np.mean(ppg_finger)
signal_std = np.std(ppg_finger)

# Feature vector
features = [ptt_avg, hr, peak_amp, signal_mean, signal_std]

# Print
print("Features:")
print("PTT:", ptt_avg)
print("Heart Rate:", hr)
print("Peak Amplitude:", peak_amp)
print("Mean:", signal_mean)
print("Std:", signal_std)

print("\nFeature Vector:", features)