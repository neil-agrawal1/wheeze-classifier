import wave
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

def process_data(file_path, type): 
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs  # Nyquist frequency
        low = lowcut / nyq  # Normalize the lowcut frequency
        high = highcut / nyq  # Normalize the highcut frequency
        b, a = butter(order, [low, high], btype='band', analog=False)
        return b, a

    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def low_pass_filter(data, cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            y = filtfilt(b, a, data)
            return y

    def high_pass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return filtfilt(b, a, data)


    with wave.open(file_path, 'rb') as wav_file:
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        fs = wav_file.getframerate()
        num_frames = wav_file.getnframes()
        duration = num_frames / fs
        frames = wav_file.readframes(num_frames)

    audio_data = np.frombuffer(frames, dtype=np.int16)

    audio_data = high_pass_filter(audio_data, 7.5, fs, 1)
    audio_data = low_pass_filter(audio_data, (fs/2)-100, fs, 8)
    audio_data = bandpass_filter(audio_data, 80, 1600, fs, 4)

    fft_values = np.fft.fft(audio_data)
    fft_freq = np.fft.fftfreq(len(audio_data), 1/fs)  # Frequency values for the x-axis

    # Take the magnitude of the FFT and plot
    magnitude = np.abs(fft_values)


    band_width = 10 # this was originally 40 Hz
    max_freq = 500  # this was originally 1600 Hz, just playing around with it though
    num_bands = int(max_freq / band_width)
    band_areas = []
    band_centers = []

    for i in range(num_bands):
        # Define band range
        band_start = i * band_width
        band_end = min((i + 1) * band_width, max_freq)
        
        # Mask to select frequency range within the band
        band_mask = (fft_freq >= band_start) & (fft_freq < band_end)
        
        # Calculate area under the magnitude within this band
        band_area = np.trapz(magnitude[band_mask], fft_freq[band_mask])
        band_areas.append(band_area)
        band_centers.append((band_start + band_end) / 2)  # Center frequency of the band

    df = pd.DataFrame(band_areas, columns=['Power'])
    df = df.T
    df['Label'] = type
    return df