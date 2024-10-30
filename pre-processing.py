import wave
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


#fs = 3000;
#sample_width = 2 (depending on number of bytes for resolution, 8-bit, 16-bit, etc)
#channels = 1

with open("audio.bin", "rb") as f:
    audio_data = f.read()


with wave.open("output_audio.wav", "w") as wav_file:
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(sample_width)
    wav_file.setframerate(fs)
    wav_file.writeframes(audio_data)

audio = np.frombuffer(audio_data, dtype=np.int16)

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


#testing commit