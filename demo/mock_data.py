# mock_data.py
import time
import numpy as np
import pandas as pd
from datetime import datetime

class MockICS:
    def __init__(self, n_sensors=4, seed=42):
        np.random.seed(seed)
        self.n = n_sensors
        # base normal values for sensors (e.g., flow, level, temp, pressure)
        self.base = np.array([50.0, 75.0, 30.0, 1.0])[:n_sensors]
        self.t = 0
        self.attack_on = False
        self.attack_type = None
        self.attack_start = None

    def step(self):
        """Return a timestamped sample (1 row, n sensors)"""
        self.t += 1
        noise = np.random.normal(scale=[0.5, 1.0, 0.2, 0.02][:self.n])
        values = self.base + 0.5*np.sin(self.t/10.0) + noise

        # If attack is on, modify readings
        if self.attack_on and self.attack_type == "sensor_spoof":
            # spoof one sensor to a high value
            values[0] += 30 + np.random.normal(scale=2.0)
        if self.attack_on and self.attack_type == "unauth_write":
            # flip actuator-like sensor (last) to 0/1 rapidly
            values[-1] = 1.0 if (self.t % 4) < 2 else 0.0

        row = {
            "time": datetime.now(),
            **{f"sensor_{i+1}": float(values[i]) for i in range(self.n)}
        }
        return row

    def start_attack(self, attack_type):
        self.attack_on = True
        self.attack_type = attack_type
        self.attack_start = datetime.now()

    def stop_attack(self):
        self.attack_on = False
        self.attack_type = None
        self.attack_start = None
