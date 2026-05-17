import numpy as np
import pandas as pd
from collections import deque
from scipy.fft import fft

class FeatureExtractor:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.voltage_buffer = deque(maxlen=window_size)
        self.current_buffer = deque(maxlen=window_size)
        self.temp_buffer = deque(maxlen=window_size)
        self.time_buffer = deque(maxlen=window_size)

    def update(self, voltage, current, temp, timestamp):
        self.voltage_buffer.append(voltage)
        self.current_buffer.append(current)
        self.temp_buffer.append(temp)
        self.time_buffer.append(timestamp)

    def extract_features(self):
        if len(self.voltage_buffer) < self.window_size:
            return None

        v_arr = np.array(self.voltage_buffer)
        c_arr = np.array(self.current_buffer)
        t_arr = np.array(self.temp_buffer)

        v_mean = np.mean(v_arr)
        v_var = np.var(v_arr)
        t_mean = np.mean(t_arr)
        t_var = np.var(t_arr)

        dv_dt = (v_arr[-1] - v_arr[0]) / self.window_size
        dt_dt = (t_arr[-1] - t_arr[0]) / self.window_size

        temp_spike = 1 if (t_arr[-1] - np.mean(t_arr[:-1])) > 2.0 else 0
        current_instability = np.std(c_arr)

        fft_v = np.abs(fft(v_arr))
        fft_v_mean = np.mean(fft_v[1:self.window_size//2]) if self.window_size > 2 else 0

        features = {
            "v_mean": v_mean,
            "v_var": v_var,
            "t_mean": t_mean,
            "t_var": t_var,
            "dv_dt": dv_dt,
            "dt_dt": dt_dt,
            "temp_spike": temp_spike,
            "current_instability": current_instability,
            "fft_v_mean": fft_v_mean
        }
        return features
